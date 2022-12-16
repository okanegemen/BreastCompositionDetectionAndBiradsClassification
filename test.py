from DataLoaders.dataset import Dataset
from DataLoaders.XLS_utils import XLS
import DataLoaders.config as config
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import time
import pandas as pd
from pretty_confusion_matrix import pp_matrix_from_data

def get_model():
    return torch.load(config.MODEL_PATH) # load previous weights

def get_dataset():
    train,test = XLS().return_datasets()

    imgs_dir = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0/images"

    train = Dataset(train,imgs_dir)
    test = Dataset(test,imgs_dir)

    return train, test

def get_dataloaders(testDS):
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=1)

    return testLoader

def plot(H):
    epochs = len(H["val_loss"])
    plt.plot(epochs,H["val_loss"])
    plt.plot(epochs,H["val_acc"])
    plt.savefig("/home/alican/Documents/AnkAI/yoloV5/output/plot_test.png")
    plt.clf()
    
    outputs = [torch.argmax(i).item() for i in H["outputs"]]
    targets = [torch.argmax(i).item() for i in H["targets"]]

    df = pd.DataFrame(list(zip(outputs, targets)),columns =config.CM_COLUMNS)
    fig = pp_matrix_from_data(df)
    fig.savefig(config.PLOT_TEST)

def get_others():

    lossFunc = BCEWithLogitsLoss()

    return lossFunc

def testing(model, lossFunc, testLoader):
    # switch off autograd
    with torch.no_grad():

        n_threads = torch.get_num_threads()
        # set the model in evaluation mode
        cpu_device = torch.device("cpu")
        model.eval()
        # loop over the validation set
        H = {"loss": [], "acc": [],"outputs":[],"targets":[]}

        test_acc = 0
        test_counter = 0
        len_testLoader = len(testLoader)

        for idx,testdata in enumerate(testLoader):
            imgs, targets = testdata
            # send the input to the device
            images = torch.stack([img.to(config.DEVICE) for img in imgs])
            targets = torch.stack([target.to(config.DEVICE) for target in targets]).float()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)
            outputs = torch.stack([v.to(cpu_device) for v in outputs]).squeeze(0)
            model_time = time.time() - model_time

            targets = targets.squeeze(0).float().to(cpu_device)
            loss = lossFunc(outputs,targets)
            

            H["outputs"].append(outputs)
            H["targets"].append(targets)

            test_counter += 1
            test_acc += 1 if torch.argmax(outputs)==torch.argmax(targets) else 0
            temp_acc = test_acc/test_counter
            H["loss"].append(loss)
            H["acc"].append(temp_acc)

            evaluator_time = time.time()
            evaluator_time = time.time() - evaluator_time
            if idx % config.PRINT_FREQ == 0  and idx != 0:
                print(f"Test:  [{idx}/{len_testLoader}]  test_loss: {loss}  test_acc: {temp_acc}  evaluator_time: {evaluator_time}")

    print(f"Test:  [{idx}/{len_testLoader}]  test_loss: {loss}  test_acc: {temp_acc}  evaluator_time: {evaluator_time}")

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

    plot(H)

if __name__ == "__main__":
    test,_,imgs_dir = XLS().get_all_info()

    test = Dataset(test,imgs_dir)

    testLoader = get_dataloaders(test)
    model = get_model()

    lossFunc = get_others()

    testing(model,lossFunc,testLoader)
