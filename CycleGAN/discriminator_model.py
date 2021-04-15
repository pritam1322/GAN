import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=True):
        super(CNN, self).__init__()
        if norm:
            self.conv = nn.Sequential(

                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode='reflect'),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(discriminator, self).__init__()
        self.conv1 = CNN(in_channels, features, stride=2, norm=False)
        self.conv2 = CNN(features, features*2, stride=2)
        self.conv3 = CNN(features*2, features*4, stride=2)
        self.conv4 = CNN(features*4, features*8, stride=1)
        self.conv5 = nn.Sequential(
                        nn.Conv2d(features*8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

                    )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return torch.sigmoid(self.conv5(x))

"""
x = torch.randn((5, 3, 256, 256))
model = discriminator()
preds = model(x)
print(preds.shape)
"""