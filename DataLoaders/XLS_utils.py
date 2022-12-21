import os
import pandas as pd
import random

if __name__ == "__main__":
    import config
else:
    import DataLoaders.config as config

# importing
class XLS():
    def __init__(self):

        self.Dataset_name = config.DATASET_NAME

        if self.Dataset_name not in config.DATASET_NAMES:
            raise ValueError(f"Invalid dataset name. Expected one of: {config.DATASET_NAME}")

        if self.Dataset_name == "INBreast":
            self.df = self.INBreast()

        elif self.Dataset_name == "VinDr":
            self.df = self.VinDr_mammo()

    def get_all_info(self):
        train_set, test_set = self.return_datasets()
        image_dir = self.return_images_dir()

        return train_set, test_set, image_dir

    def return_datasets(self, test_split = config.TEST_SPLIT):

        self.idxs = [*range(len(self.df))]
        random.shuffle(self.idxs)
        self.split = int(test_split*len(self.idxs))

        self.train_idxs = self.idxs[self.split:]
        self.test_idxs = self.idxs[:self.split]

        remain_set = self.df[self.df.index.isin(self.train_idxs)].sample(frac = 1).reset_index(drop=True)
        test = self.df[self.df.index.isin(self.test_idxs)].sample(frac = 1).reset_index(drop=True)

        return remain_set, test

    def return_images_dir(self):
        if self.Dataset_name == "INBreast":
            return os.path.join(self.root,"AllDICOMs")
        elif self.Dataset_name == "VinDr":
            return os.path.join(self.root,"Dicom_images")

    def VinDr_mammo(self):

        self.root = "/Users/okanegemen/yoloV5/INbreast Release 1.0/"
        info_filename = "INbreast.csv"

        df = pd.read_csv(self.root+info_filename)

        df["ACR"] = df["breast_density"].apply(lambda x: list(x)[-1]).replace(["A","B","C","D"],[1,2,3,4])
        df['Bi-Rads'] = df['breast_birads'].apply(lambda x: int(list(x)[-1]))
        df["File Name"] = df[['study_id', 'image_id']].agg('/'.join, axis=1)

        df.rename(columns={"view_position":"View","laterality":"Laterality"},inplace=True)

        eliminated_columns_names = ["series_id","split","height","width","breast_birads","breast_density","image_id","study_id"]
        df = df.drop(eliminated_columns_names, axis=1)

        return df

    def INBreast(self,row_end=410):
        self.root = "/home/alican/Documents/Datasets/INBreast"
        info_filename = "INbreast.xls"

        xls = pd.ExcelFile(os.path.join(self.root,info_filename))
        df = xls.parse(0).iloc[:row_end,:]

        df["File Name"] = df["File Name"].apply(lambda x:str(int(x)))
        df['Bi-Rads'] = df['Bi-Rads'].replace(['4a', '4b', '4c'], 4)

        if config.CONVERT_BI_RADS:
            df['Bi-Rads'] = df['Bi-Rads'].replace([3,4, 5], [2,3,3])
            df = df[df["Bi-Rads"]!= 6]

        if config.ONLY_CC:
            df = df[df["View"] == "CC"]

        eliminated_columns_names = ["Patient ID","Patient age","Other Notes","Other Annotations","Acquisition date","Pectoral Muscle Annotation","Asymmetry","Distortion","Micros","Mass ","Findings Notes (in Portuguese)","Lesion Annotation Status"]
        df = df.drop(eliminated_columns_names, axis=1)

        return df

if __name__ == "__main__":
    train,test = XLS().return_datasets()

    print(train["File Name"][0])
    print(test)