# Author: Zeyu Wang
# Department: Sun Yat-sen University
# Function: (LBGNN) Generate backlight

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.conv3(x)
        identity = self.bn3(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class DoubleConv2(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None, groups=1):
        if mid_channels is None:
            mid_channels = in_channels
        super(DoubleConv2, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DownSample(nn.Sequential):
    def __init__(self):
        super(DownSample, self).__init__(
            nn.MaxPool2d(2, stride=2),
        )

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False),
            nn.Tanh()
        )

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=6):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = matlab_style_gauss2D((41, 41), 10)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        Padding = nn.ReflectionPad2d(20)
        x = Padding(x)
        x = F.conv2d(x, self.weight, groups=self.channels)
        return x

class LBGNN(nn.Module):
    def __init__(self, in_channels=3, color_channels=6, base_c=16):
        super(LBGNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = color_channels
        self.RB1 = BasicBlock(in_channels, base_c)
        self.RB2 = BasicBlock(base_c, base_c * 2)
        self.RB3 = BasicBlock(base_c * 2, base_c * 4)
        self.DSC1 = DoubleConv2(base_c * 4, base_c * 2, groups=base_c * 4)
        self.DSC2 = DoubleConv2(base_c * 2, base_c * 1, groups=base_c * 2)
        self.out = OutConv(base_c * 1, color_channels)
        self.downsample = DownSample()
        # 1080*1920-->144*256-->9*16
        self.DownBilinear = nn.Upsample(size=(144, 256), mode='bilinear')
        self.Upnearest = nn.Upsample(size=(360, 640), mode='nearest')
        self.GaussF = GaussianBlurConv()

    def forward(self, x: torch.Tensor):
        x = self.DownBilinear(x)

        x1 = self.RB1(x)
        x1 = self.downsample(x1)
        x2 = self.RB2(x1)
        x2 = self.downsample(x2)
        x3 = self.RB3(x2)
        x3 = self.downsample(x3)
        x4 = self.DSC1(x3)
        x4 = self.downsample(x4)
        x5 = self.DSC2(x4)
        output = self.out(x5)

        backlightplus = self.Upnearest(output)
        backlightreal = self.GaussF(backlightplus)

        return backlightreal

model = LBGNN()
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(model))
