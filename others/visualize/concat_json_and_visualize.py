import json
from jsonmerge import merge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_value_from_list_dict(x:dict,train:bool=True):
    if train:
        a = [d for d in x["train"]]
        b = [d for d in x["val"]]
        return a,b
    else:
        a = [d for d in x]
        return a

def concatJson(path1:str,path2:str,wantedPath:str=None):
     
    j1 = open(path1)
    j2 = open(path2)

    data1 = json.load(j1)
    data2 = json.load(j2)

    training = data1["training"] + data2["training"]
    test = data1["test"] + data2["test"]

    result = {"training":training,"test":test}

    splitted = path1.split("/")
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

    print(name)
    with open(name,"w") as f:
        json.dump(result,f)
    
    f.close()

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
            val_acc.append(valid_val[val]["f1"]*100)
        for train in range(len(train_val)):

            train_acc.append(train_val[train]["f1"]*100)
    

    

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

path1 = """/home/alican/Documents/yoloV5/results_models/ConcatModel_2_4_12_57/ConcatModel.json"""
path2 = """/home/alican/Documents/yoloV5/results_models/ConcatModel_2_4_17_8/ConcatModel.json"""
concatJson(path1,path2)