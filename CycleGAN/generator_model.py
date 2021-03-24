import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, padding_mode='reflect') if down else
                nn.ConvTranspose2d(in_channels, out_channels, 3, 0.5, 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return  self.conv(x)

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.Identity()

            )

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()
        self.c7s1_64 = nn.Sequential(
            nn.Conv2d(in_channels, features, 7, 1, 3, padding_mode='reflect'),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.down1 = CNN(features, features*2)
        self.down2 = CNN(features*2, features*4)
        self.res_block = nn.Sequential(
        *[residual_block(features*4, features*4) for i in range(6)]
        )
        self.up1 = CNN(features*4, features*2)
        self.up2 = CNN(features*2, features)
        self.c7s1_3 = nn.Conv2d(features, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect')


    def forward(self, x):
        x = self.c7s1_64(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_block(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.c7s1_3(x)
        return torch.tanh(x)

"""
x = torch.randn((2,3,256,256))
model = Generator(in_channels=3, features=64)
print(model(x).shape)
"""