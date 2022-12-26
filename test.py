import DataLoaders.config as config
import torch
from DataLoaders.scores import scores
from qqdm import qqdm, format_str

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
