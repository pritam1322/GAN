import torch
import torch.nn as nn
import torchvision.transforms as transform


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, conv=True, relu=False, dropout = True):
        super(CNN, self).__init__()
        if normalize:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
                if conv
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2) if relu == False else nn.ReLU()
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
                if conv
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2) if relu == False else nn.ReLU()
            )
        self.dropout = dropout
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        x = self.cnn(x)
        return self.drop(x) if self.dropout else x

class GeneratorUNET(nn.Module):
    def __init__(self, in_channels=3, feature=64):
        super(GeneratorUNET, self).__init__()
        self.down_sampling = nn.ModuleList()
        self.up_sampling = nn.ModuleList()
        self.down1 = CNN(in_channels, feature, normalize=False, relu=False, dropout=False)
        self.down2 = CNN(feature, feature*2,relu=False, dropout=False)
        self.down3 = CNN(feature*2, feature * 4, relu=False, dropout=False)
        self.down4 = CNN(feature*4, feature * 8, relu=False, dropout=False)
        self.down5 = CNN(feature*8, feature*8 , relu=False, dropout=False)
        self.down6 = CNN(feature * 8, feature * 8, relu=False, dropout=False)
        self.down7 = CNN(feature * 8, feature * 8, relu=False, dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature*8, feature*8, 4, 2, 1),
            nn.ReLU()
        )

        self.up1 = CNN(feature*8, feature*8, conv=False, relu=True)
        self.up2 = CNN(feature*16, feature*8, conv=False, relu=True)
        self.up3 = CNN(feature*16, feature*8, conv=False, relu=True)
        self.up4 = CNN(feature*16, feature*8, conv=False, relu=True, dropout=False)
        self.up5 = CNN(feature * 16, feature * 4, conv=False, relu=True, dropout=False)
        self.up6 = CNN(feature * 8, feature * 2, conv=False, relu=True, dropout=False)
        self.up7 = CNN(feature * 4, feature , conv=False, relu=True, dropout=False)

        self.fcs = nn.Sequential(
            nn.ConvTranspose2d(feature*2, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.fcs(torch.cat([up7,d1], 1))

'''
x = torch.randn((1,3,256,256))
model = GeneratorUNET(in_channels=3, feature=64)
preds = model(x)
print(preds.shape)
print(x.shape)

'''