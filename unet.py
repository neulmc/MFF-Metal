import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ConvBlock=DoubleConv):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, ConvBlock=DoubleConv,):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up = nn.ConvTranspose2d(in_channels/2, in_channels/2, kernel_size=2, stride=2),
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # class 4
        out_channels = 4
        super(OutConv, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=None)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=None)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, mode='RAW', model_c=64):
        super(UNet, self).__init__()
        c = model_c
        self.mode = mode
        ConvBlock = DoubleConv

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ConvBlock(n_channels, c)
        self.down1 = Down(c, int(c * 2), ConvBlock)
        self.down2 = Down(int(c * 2), int(c * 4), ConvBlock)
        self.down3 = Down(int(c * 4), int(c * 8), ConvBlock)
        factor = 2
        self.down4 = Down(int(c * 8), int(c * 16) // factor, ConvBlock)
        self.up1 = Up(int(c * 16), int(c * 8) // factor, ConvBlock)
        self.up2 = Up(int(c * 8), int(c * 4) // factor, ConvBlock)
        self.up3 = Up(int(c * 4), int(c * 2) // factor, ConvBlock)
        self.up4 = Up(int(c * 2), c, ConvBlock)
        self.outc = OutConv(c, n_classes)

    def forward(self, x, sample=True):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = x5
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits