import os
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import glob
import config
# if __name__ == "__main__":
#     import config
# else:
#     import DataLoaders.config as config

# importing
class XLS():
    def __init__(self):
        self.root = config.TEKNOFEST
        self.columns = ["HASTANO",	
                "BIRADS KATEGORİSİ",
                "MEME KOMPOZİSYONU",	
                "KADRAN BİLGİSİ (SAĞ)",	
                "KADRAN BİLGİSİ (SOL)",	
                "Birads Skoru (EK BİLGİ OLARAK VERİLMİŞTİR, YARIŞMADA İSTENMEYECEKTİR)"
                ]
        self.df = self.teknofest_data()

    def return_datasets(self,df=None, test_split = config.TEST_SPLIT):
        if df == None:
            df = self.df
        if config.CROP_DATA>0:
            self.df, _ = train_test_split(df,test_size=config.CROP_DATA,shuffle=True,stratify=self.df["BIRADS KATEGORİSİ"],random_state=44)

        remain_set, test = train_test_split(df,test_size=test_split,shuffle=True,stratify=self.df["BIRADS KATEGORİSİ"],random_state=44)
        return remain_set, test

    def teknofest_data(self):
        info_filename = glob.glob(os.path.join(config.TEKNOFEST,'*.xlsx'))[0]

        excel_data = pd.read_excel(os.path.join(self.root,info_filename))
        df = pd.DataFrame(excel_data, columns = self.columns)

        df[self.columns[0]] = df[self.columns[0]].apply(lambda x: str(x))

        df[self.columns[1]] = df[self.columns[1]].apply(lambda x: self.birads_to_int(x))

        df[self.columns[2]] = df[self.columns[2]].apply(lambda x: self.kompozisyon_to_int(x))
        
        df[self.columns[3]] = df[self.columns[3]].fillna("[]")
        df[self.columns[4]] = df[self.columns[4]].fillna("[]")
        
        df[self.columns[3]] = df[self.columns[3]].apply(lambda x:ast.literal_eval(x))
        df[self.columns[4]] = df[self.columns[4]].apply(lambda x:ast.literal_eval(x))

        df[self.columns[3]] = df[self.columns[3]].apply(lambda x:self.kadran_to_int(x))
        df[self.columns[4]] = df[self.columns[4]].apply(lambda x:self.kadran_to_int(x))

        df[self.columns[5]] = df[self.columns[5]].apply(lambda x: int(list(x)[9])).astype("int64")

        return df
    
    @classmethod
    def kadran_to_int(cls, kadranlar:list, choices = ["ÜST DIŞ","ÜST İÇ","ALT İÇ","ALT DIŞ", "MERKEZ"]):
        if len(kadranlar)>0:
            return [choices.index(kadran) for kadran in kadranlar]
        else:
            return []
    
    @classmethod
    def kompozisyon_to_int(cls, kompozisyon:str, choices = ["A","B","C","D"]):
        return choices.index(kompozisyon)
    
    @classmethod
    def birads_to_int(cls, birads:str, choices = ["BI-RADS0","BI-RADS1-2","BI-RADS4-5"]):
        return choices.index(birads)
            
if __name__ == "__main__":
    data_class =  XLS()
    train,test =data_class.return_datasets()

    print(train)
    print(test)