import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride, normalize=True ):
        super(CNN, self).__init__()
        if normalize:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels ,4 ,stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        else :
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False),
                nn.LeakyReLU(0.2),
            )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, feature=64):
        super(Discriminator, self).__init__()
        self.down1 = CNN(in_channels*2, feature, stride=2, normalize=False)
        self.down2 = CNN(feature, feature*2, stride=2)
        self.down3 = CNN(feature*2, feature*4, stride=2)
        self.down4 = CNN(feature*4, feature*8, stride=1)
        self.down5 = nn.Sequential(
            nn.Conv2d(feature*8, 1, 4, padding=1),
        )

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)
        # Concatenate image and condition image by channels to produce input
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        return x

'''
x = torch.randn((1, 3, 256, 256))
y = torch.randn((1, 3, 256, 256))
model = Discriminator(in_channels=3, feature=64)
preds = model(x, y)
print(preds.shape)
'''