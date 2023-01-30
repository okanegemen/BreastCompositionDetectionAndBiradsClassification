import DataLoaders.config as config
import torch
from DataLoaders.scores import scores
from qqdm import qqdm, format_str
import math
import os
import sys
import imp

def focal_loss(outputs,targets, alpha=1, gamma=2):
    ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
    return focal_loss

def training(model, trainLoader, lossFunc, optimizer, valLoader=None,fold="---"):
    if config.FOCAL_LOSS:
        loss_Func = focal_loss
    metrics = {"train":[],"val":[]}

    # loop over epochs
    print("[INFO] training the network...")
    for epoch in range(config.NUM_EPOCHS):
        scores_train = scores()
        # set the model in training mode
        model.train()

        lr_scheduler = None
        # if epoch == 0:
        #     warmup_factor = 1.0 / 1000
        #     warmup_iters = min(1000, len(trainLoader) - 1)

        #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        #     )
        

        # loop over the training set

        train_loss = []
        for idx_t,traindata in enumerate(tw :=qqdm(trainLoader, desc=format_str('bold', 'Description'))):
            images,targets = traindata
            # send the input to the device
            images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss_train = lossFunc(outputs,targets)
            
            l1_regularization = 0.
            l2_regularization = 0.
            if config.L1regularization:
                for param in model.parameters():
                    l1_regularization += param.abs().sum()
            if config.L2regularization:
                for param in model.parameters():
                    l2_regularization += (param**2).sum()
            loss_train += l1_regularization + l2_regularization

            optimizer.zero_grad()

            train_loss.append(loss_train.item())
            temp_loss = sum(train_loss[-20:]) / min([len(train_loss),20])

            if not math.isfinite(loss_train):
                print(f"Loss is {loss_train}, stopping training")
                sys.exit(1)

            loss_train.backward()
            optimizer.step()

            scores_train.update(outputs,targets)
            tw.set_infos({"Fold":fold,
                            "Epoch":f"{epoch}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.7f}",
                            "loss":"%.4f"%temp_loss,
                            **scores_train.metrics()})

            if lr_scheduler is not None:
                lr_scheduler.step()
        
        n_threads = torch.get_num_threads()

        metrics["train"].append(scores_train.metric)
        
        if valLoader!=None:
            if epoch % config.VALIDATE_PER_EPOCH == 0 and (epoch != 0 or config.VALIDATE_PER_EPOCH == 1):
                print(f'Validation')
                print('--------------------------------')
                scores_val = scores()
                # set the model in evaluation mode
                model.eval()
                # loop over the validation set
                val_loss = []

                # switch off autograd
                with torch.no_grad():
                    for idx_v, valData in enumerate(tw :=qqdm(valLoader, desc=format_str('bold', 'Description'))):
                        images, targets = valData

                        # send the input to the device
                        images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        outputs = model(images)
                        loss_val = lossFunc(outputs,targets)

                        val_loss.append(loss_val.item())
                        temp_loss = sum(val_loss[-20:]) / min([len(val_loss),20])

                        scores_val.update(outputs,targets)
                        tw.set_infos({"loss":"%.4f"%temp_loss,
                                    **scores_val.metrics()})

        if epoch % config.SAVE_MODEL_PER_EPOCH == 0 or epoch == config.NUM_EPOCHS-1:
            imp.reload(config)
            name = ""+model.__class__.__name__+"_"+str(fold)+"_"+config.DATE_FOLDER
            os.makedirs(os.path.join(config.MID_FOLDER,name))
            print("\nSaving Model State Dict...")
            # torch.save(model.state_dict(), config.MID_FOLDER+"/"+name+"/"+name+".pth")

        # accumulate predictions from all images
        torch.set_num_threads(n_threads)
    return metrics



def testing(model, lossFunc, testLoader):
    print(f'TEST')
    print('--------------------------------')
    scores_test = scores()
    test_loss = []

    # switch off autograd
    with torch.no_grad():

        n_threads = torch.get_num_threads()
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set

        for idx,testdata in enumerate(tw :=qqdm(testLoader, desc=format_str('bold', 'Description'))):
            images,targets = testdata
            # send the input to the device
            images, targets = torch.stack(images).to(config.DEVICE), torch.stack(targets).view(-1).to(config.DEVICE)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            outputs = model(images)
            loss_test = lossFunc(outputs,targets)

            test_loss.append(loss_test.item())
            temp_loss = sum(test_loss[-20:]) / min([len(test_loss),20])
            
            scores_test.update(outputs,targets)
            tw.set_infos({"loss":"%.4f"%temp_loss,
                        **scores_test.metrics()})


    # accumulate predictions from all images
    torch.set_num_threads(n_threads)
    return scores_test.metric