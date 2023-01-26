""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64,32)
        self.silu = nn.SiLU(False)
        self.fc2 = nn.Linear(32,n_classes)
        self.soft_max = nn.LogSoftmax(dim = 1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = self.outc(x)
        x = self.adp(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.soft_max(x)
        return {"birads":x}





