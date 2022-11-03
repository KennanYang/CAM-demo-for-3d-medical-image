# UNet_3d.py
# Here, the UNet_3d network structure is defined, which is used for 3d medical image segmentation

import torch
import torch.nn as nn

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(          
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet_3d(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, training):
        super(UNet_3d, self).__init__()

        self.training = training
        in_ch = 1
        out_ch = 2
        
        n1 = 32
        channels = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        #-----------------------------Encoder---------------------------
        self.Conv1 = conv_block(in_ch, channels[0])
        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv2 = conv_block(channels[0], channels[1])
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv4 = conv_block(channels[2], channels[3])

        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Conv5 = conv_block(channels[3], channels[4])

        #-----------------------------Decoder---------------------------

        # self.Up5 = up_conv(channels[4], channels[3])
        # self.Up_conv5 = conv_block(channels[4], channels[3])
        
        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[3], channels[2])

        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[2], channels[1])

        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[1], channels[0])

        self.map = nn.Sequential(
            nn.Conv3d(channels[0], out_ch, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # e5 = self.Conv5(e5)

        # d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)

        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(d5)

        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.map(d2)

        return out


def init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)

net_stage2 = UNet_3d(training=True)
net_stage2.apply(init)

# net_total_para = sum(param.numel() for param in net.parameters())
# print('Run U_Net， total parameters:', net_total_para)