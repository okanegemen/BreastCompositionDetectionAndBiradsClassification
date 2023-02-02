import json
from jsonmerge import merge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def concatJson(path1:str,path2:str,wantedPath:str=None):


    with open(path1) as j1,open(path2) as j2:
        data1 = json.load(j1)
        data2 = json.load(j2)
    j1.close()
    j2.close()
    result = merge(data1,data2)
    splitted = path1.split("/")
    print(path1[:-5])
    if wantedPath is None:

        print(path1[-5])
        if path1[-6]=="c":
            name = path1[:-5] + "c" + ".json"
        else:
            name = path1[:-5] + "_c" +".json"
    else:         
        if path1[-6]=="c":
            name = wantedPath + "/" + splitted[-1][:-5] + "c" + ".json"
        else:
            name = wantedPath + "/" + splitted[-1][:-5] + "_c" +".json"

    with open(name,"w") as f:
        json.dump(result,f)
    
    f.close()

path = "/Users/okanegemen/Desktop/yoloV5/folder/metrics_Resnet50_c_c.json"
def plot_acc(path:str):

    assert path.endswith(".json") , "You must enter json file path"
    
    dataframe = pd.read_json(path)
    train = dataframe["training"]
    test = dataframe["test"]
    train_list = list(train.iloc[0:])
    new_dict = train_list[0]
    train_acc= []
    val_acc = []
    for i in range(len(train_list)):
        buffer = train_list[i]
        try:
            valid_val = buffer["val"]
            train_val = buffer["train"]
        except KeyError:
            train_val = buffer["train"]
        for val in range(len(valid_val)):
            val_acc.append(valid_val[val]["acc"]*100)
        for train in range(len(train_val)):

            train_acc.append(train_val[train]["acc"]*100)
    

    

    plt.figure()
    plt.ylim(0,100)
    plt.yticks(np.arange(0,101,10))
    plt.xticks(np.arange(0,101,10))
    plt.plot(train_acc,label ='train accuracy')
    plt.plot(np.arange(0,100,5),val_acc,label ="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch bazlı başarı oranı grafiği")
    plt.legend()
