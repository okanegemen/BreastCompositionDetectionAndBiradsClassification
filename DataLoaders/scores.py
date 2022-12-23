from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall,Dice,MulticlassAccuracy,MulticlassAUROC,F1Score,MulticlassHingeLoss
import DataLoaders.config as config
import torch

class scores: # TO DO
    def __init__(self) -> None:
        self.class_num = config.NUM_CLASSES
        
        self.accuracy = MulticlassAccuracy(num_classes=self.class_num).to(config.DEVICE)
        self.f1score = F1Score("multiclass",num_classes=self.class_num).to(config.DEVICE)
        self.dice_score = Dice(num_classes=self.class_num).to(config.DEVICE)
        self.precision = MulticlassPrecision(num_classes=self.class_num).to(config.DEVICE)
        self.recall = MulticlassRecall(num_classes=self.class_num).to(config.DEVICE)
        self.auroc = MulticlassAUROC(num_classes=self.class_num).to(config.DEVICE)
        self.hingeloss = MulticlassHingeLoss(num_classes=self.class_num).to(config.DEVICE)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.class_num).to(config.DEVICE)

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
        self.flag = False
        
        conf_m = self.confusion_matrix.compute()
        confusion = self.confusion_str(conf_m/torch.sum(conf_m,dim=1)*100)


        metrics = {
        "hingeloss":"%.4f"%hingeloss,
        "acc":"%.4f"%accuracy,
        "f1":"%.4f"%f1score,
        "dice":"%.4f"%dice,
        "precision":"%.4f"%precision,
        "recall":"%.4f"%recall,
        "auroc":"%.4f"%auroc,
        " "*150:" "*10,
        "confusion matrix"+" "*87:confusion}

        return metrics

    def confusion_str(self,cm) -> str:
        confusion = "\n".join(["".join([f"{round_int(j):>4}" for j in i]) for i in cm.tolist()])
        return confusion

def round_int(x):
    try:
        if x in [float("-inf"),float("inf"),float("nan")]: return 0
        
        return round(x,2)
    except:
        return 0.00
if __name__ == "__main__":
    a = scores()

    