import DataLoaders.config as config
import torch
from DataLoaders.scores import scores
from qqdm import qqdm, format_str
import math
import sys

def training(model, trainLoader, lossFunc, optimizer, valLoader=None,fold="---"):
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
                outputs = model(images)["birads"]
                loss_train = lossFunc(outputs,targets)
            
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
                        outputs = model(images)["birads"]
                        loss_val = lossFunc(outputs,targets)

                        val_loss.append(loss_val.item())
                        temp_loss = sum(val_loss[-20:]) / min([len(val_loss),20])

                        scores_val.update(outputs,targets)
                        tw.set_infos({"loss":"%.4f"%temp_loss,
                                    **scores_val.metrics()})

        if epoch % config.SAVE_MODEL_PER_EPOCH == 0 or epoch == config.NUM_EPOCHS-1:
            print("\nSaving Model State Dict...")
            torch.save(model.state_dict(), config.MODEL_PATH.strip(".pth")+"_fold"+str(fold)+"_epoch"+str(epoch)+".pth")

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

            outputs = model(images)["birads"]
            loss_test = lossFunc(outputs,targets)

            test_loss.append(loss_test.item())
            temp_loss = sum(test_loss[-20:]) / min([len(test_loss),20])
            
            scores_test.update(outputs,targets)
            tw.set_infos({"loss":"%.4f"%temp_loss,
                        **scores_test.metrics()})


    # accumulate predictions from all images
    torch.set_num_threads(n_threads)
    return scores_test.metric