import torch
from torchvision.models import resnet18 as load_model,ResNet18_Weights 
from DataLoaders.XLS_utils import XLS
from DataLoaders.dataset import Dataset
import torchvision.transforms as T
from PIL import Image

def get_model():
    model = load_model(weights = ResNet18_Weights.IMAGENET1K_V1)
    params = model.conv1
    print(params)

    for param in params.parameters():
            param.requires_grad = False


    for id,(name,param) in enumerate(params.named_parameters()):
        print(name,param.requires_grad)
    return params

transform  = T.ToPILImage()

if __name__=="__main__":
    model = get_model()
    train, test= XLS().return_datasets()

    train = Dataset(train,True)
    test = Dataset(test,False)

    plot = Image.new("RGB", (32*8, 32*8), "white")

    image, birads  = train[0]
    images = image.unsqueeze(1)
    images = torch.cat([images,images,images],dim=1)
    print(images.size())
    output = model(images)
    print(output.size(),birads)
    point = 0
    for view in output.squeeze():
        transform(images[0]).show()
        for img in view:
            plot.paste(transform(img), (point%(32*8), 32*(point//(32*8))))
            point += 32

    plot.show()