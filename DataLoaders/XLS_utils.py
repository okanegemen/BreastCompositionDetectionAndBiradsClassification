import os
import pandas as pd
import random
import numpy as np

if __name__ == "__main__":
    import config
else:
    # import config
    import DataLoaders.config as config

# importing
class XLS():
    def __init__(self,img_folder):
        self.img_folder = img_folder
        self.Dataset_name = config.DATASET_NAME

        if self.Dataset_name == "VinDr":
            self.df = self.VinDr_mammo()

    def get_all_info(self):
        dataset = self.df
        image_dir = self.return_images_dir()

        return dataset, image_dir

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
        return os.path.join(self.root,self.img_folder)

    def VinDr_mammo(self):
        self.root = config.VINDR_DIR
        info_filename = "breast-level_annotations.csv"

        df = pd.read_csv(os.path.join(self.root,info_filename))

        df["ACR"] = df["breast_density"].apply(lambda x: list(x)[-1]).replace(["A","B","C","D"],[1,2,3,4])
        df['Bi-Rads'] = df['breast_birads'].apply(lambda x: int(list(x)[-1]))
        df["File Name"] = df[['study_id', 'image_id']].agg('/'.join, axis=1)

        df.rename(columns={"view_position":"View","laterality":"Laterality"},inplace=True)

        eliminated_columns_names = ["series_id","split","height","width","breast_birads","breast_density","image_id","study_id"]
        df = df.drop(eliminated_columns_names, axis=1)

        if config.CONVERT_BI_RADS:
            df = df[df["Bi-Rads"]!= 6]
            df = df[df["Bi-Rads"]!= 3]
            df['Bi-Rads'] = df['Bi-Rads'].replace([0,1,2,4,5], [0,1,1,2,2])

        return df

if __name__ == "__main__":
    train,test = XLS().return_datasets()

    print(train["File Name"][0])
    print(test)