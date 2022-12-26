from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall,Dice,MulticlassAccuracy,MulticlassAUROC,F1Score,MulticlassHingeLoss
from torchmetrics.classification import BinaryConfusionMatrix, BinaryPrecision, BinaryRecall,Dice,BinaryAccuracy,BinaryAUROC,F1Score,BinaryHingeLoss

import DataLoaders.config as config
import torch

class scores: # TO DO
    def __init__(self) -> None:
        self.class_num = config.NUM_CLASSES
        
        if self.class_num > 2:
            self.accuracy = MulticlassAccuracy(num_classes=self.class_num).to(config.DEVICE)
            self.f1score = F1Score("multiclass",num_classes=self.class_num).to(config.DEVICE)
            self.dice_score = Dice(num_classes=self.class_num).to(config.DEVICE)
            self.precision = MulticlassPrecision(num_classes=self.class_num).to(config.DEVICE)
            self.recall = MulticlassRecall(num_classes=self.class_num).to(config.DEVICE)
            self.auroc = MulticlassAUROC(num_classes=self.class_num).to(config.DEVICE)
            self.hingeloss = MulticlassHingeLoss(num_classes=self.class_num).to(config.DEVICE)
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.class_num).to(config.DEVICE)
        else:
            self.accuracy = BinaryAccuracy(num_classes=self.class_num).to(config.DEVICE)
            self.f1score = F1Score("multiclass",num_classes=self.class_num).to(config.DEVICE)
            self.dice_score = Dice(num_classes=self.class_num).to(config.DEVICE)
            self.precision = BinaryPrecision(num_classes=self.class_num).to(config.DEVICE)
            self.recall = BinaryRecall(num_classes=self.class_num).to(config.DEVICE)
            self.auroc = BinaryAUROC(num_classes=self.class_num).to(config.DEVICE)
            self.hingeloss = BinaryHingeLoss(num_classes=self.class_num).to(config.DEVICE)
            self.confusion_matrix = BinaryConfusionMatrix(num_classes=self.class_num).to(config.DEVICE)


    def update(self,output:torch.Tensor,target:torch.Tensor):
        
        self.accuracy.update(output,target)
        self.f1score.update(output,target)
        self.dice_score.update(output,target)
        self.precision.update(output,target)
        self.recall.update(output,target)
        self.auroc.update(output,target)
        self.hingeloss.update(output,target)
        self.confusion_matrix.update(output,target)

    def metrics(self):
        accuracy = self.accuracy.compute().item()
        f1score = self.f1score.compute().item()
        dice = self.dice_score.compute().item()
        precision = self.precision.compute().item()
        recall = self.recall.compute().item()
        auroc = self.auroc.compute().item()
        hingeloss = self.hingeloss.compute().item()
        
        conf_m = self.confusion_matrix.compute()
        confusion = self.confusion_str(conf_m)


        metrics = {
        "hingeloss":"%.4f"%hingeloss,
        "acc":"%.4f"%accuracy,
        "f1":"%.4f"%f1score,
        "dice":"%.4f"%dice,
        "precision":"%.4f"%precision,
        "recall":"%.4f"%recall,
        "auroc\n":"%.4f\n"%auroc,
        "confusion matrix":confusion}

        self.metric = {
        "hingeloss":hingeloss,
        "acc":accuracy,
        "f1":f1score,
        "dice":dice,
        "precision":precision,
        "recall":recall,
        "auroc":auroc,
        "cm":conf_m.tolist()}

        return metrics

    def confusion_str(self,cm) -> str:
        confusion = "\n".join(["".join([f"{int(j):>5}" for j in i]) for i in cm.tolist()])
        return confusion

if __name__ == "__main__":
    a = scores()

    