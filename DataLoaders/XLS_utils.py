import os
import pandas as pd
import random

class XLS():
    def __init__(self,root,train_split=0.8,xls_name = "INbreast.xls"):
        self.root = root
        self.train_split = train_split
        self.xls_name = xls_name

        self.xls = self.open_xls(self.root,xls_filename=self.xls_name,row_end=410)

        self.df = self.eliminate_columns_of_df(self.xls)

    def return_datasets(self):
        self.idxs = [*range(len(self.df))]
        random.shuffle(self.idxs)
        self.split = int(self.train_split*len(self.idxs))
        self.train_idxs = self.idxs[:self.split]
        self.test_idxs = self.idxs[self.split:]
        train = self.df[self.df.index.isin(self.train_idxs)].sample(frac = 1).reset_index(drop=True)
        test = self.df[self.df.index.isin(self.test_idxs)].sample(frac = 1).reset_index(drop=True)
        return train,test

    @staticmethod
    def open_xls(root,xls_filename,row_end):
        xls = pd.ExcelFile(os.path.join(root,xls_filename))
        sheetX = xls.parse(0).iloc[:row_end,:]
        sheetX["File Name"] = sheetX["File Name"].apply(lambda x:int(x))
        return sheetX
    
    @staticmethod
    def eliminate_columns_of_df(df):
        eliminated_columns_names = ["Patient ID","Patient age","Other Notes","Other Annotations","Acquisition date","Pectoral Muscle Annotation","Asymmetry","Distortion","Micros","Mass ","Findings Notes (in Portuguese)","Lesion Annotation Status"]
        df = df.drop(eliminated_columns_names, axis=1)
        return df

if __name__ == "__main__":
    root = "/home/alican/Documents/AnkAI/yoloV5/INbreast Release 1.0"
    train,test = XLS(root).return_datasets()

    print(train.keys())
    print(test)