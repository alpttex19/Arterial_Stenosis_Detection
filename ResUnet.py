import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True) if ch_in != ch_out else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        if self.shortcut:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ResUnet(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down-sampling path
        self.conv1 = ResBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = ResBlock(ch_in=64, ch_out=128)
        self.conv3 = ResBlock(ch_in=128, ch_out=256)
        self.conv4 = ResBlock(ch_in=256, ch_out=512)
        self.conv5 = ResBlock(ch_in=512, ch_out=1024)

        # Up-sampling path
        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.up_conv5 = ResBlock(ch_in=1024, ch_out=512)
        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.up_conv4 = ResBlock(ch_in=512, ch_out=256)
        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.up_conv3 = ResBlock(ch_in=256, ch_out=128)
        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.up_conv2 = ResBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Down-sampling path
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # Up-sampling path
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1

if __name__ == "__main__":
    model = ResUnet(img_ch=1, output_ch=2)
    print(model)
