from typing import Optional
import json
import matplotlib.pyplot as plt
import numpy as np

def get_value_from_list_dict(x:dict,y:str="f1",train:bool=True):
    if train:
        a = [d[y]*100 for d in x["train"]]
        if len(x["val"])>0:
            b = [d[y]*100 for d in x["val"]]
        else: b = None
        return a,b
    else:
        a = [d[y]*100 for d in x]
        return a

def plot(src,dst=None):
    
    if dst == None:
        dst = src.strip(".json")+".png"

    f = open(src)
    dicti = json.load(f)
    # print(dicti["training"])
    trains = []
    vals = []

    fold_num = len(dicti["training"])/len(dicti["test"])
    for fold in dicti["training"]:
        train,val = get_value_from_list_dict(fold)
        trains.append(train)
        vals.append(val)
        
    epoch_num = len(trains[0])

    tests = get_value_from_list_dict(dicti["test"],train=False)

    trains = sum(trains,[])

    plt.figure()
    plt.ylim(0,100)
    plt.yticks(np.arange(0,101,10))
    plt.plot(trains,label ='Train accuracy')
    plt.scatter([a*fold_num*epoch_num for a in range(1,len(tests)+1)],tests,label ="Test accuracy",c="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(src)
    plt.legend()
    plt.grid()
    plt.show()

path = "/home/alican/Documents/yoloV5/results_models/ConcatModel_2_4_12_57/ConcatModel_c.json"
plot(path)  

