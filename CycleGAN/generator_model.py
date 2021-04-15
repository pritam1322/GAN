import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs) if down else
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity()
            )
    def forward(self, x):
        return  self.conv(x)

class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()

        self.conv1 = CNN(in_channels, in_channels,kernel_size=3, padding=1)
        self.conv2 = CNN(in_channels, in_channels, kernel_size=3, padding=1, use_act=False)




    def forward(self, x):
        residual =x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()
        self.c7s1_64 = CNN(in_channels, features, kernel_size=7, stride=1 ,padding=3)
        self.down1 = CNN(features, features*2, kernel_size=3, stride=2, padding=1)
        self.down2 = CNN(features*2, features*4, kernel_size=3, stride=2, padding=1)
        self.res_block = nn.Sequential(
        *[residual_block(features*4) for i in range(6)]
        )
        self.up1 = CNN(features*4, features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = CNN(features*2, features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c7s1_3 = nn.Sequential(
            nn.Conv2d(features, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh())


    def forward(self, x):
        x = self.c7s1_64(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_block(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.c7s1_3(x)
        return x

"""
x = torch.randn((2,3,256,256))
model = Generator(in_channels=3, features=64)
pred = model(x)
print(pred.shape)
"""