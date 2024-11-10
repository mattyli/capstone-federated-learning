# This is the PyTorch implementation of XNet
# https://github.com/JosephPB/XNet/blob/master/XNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# hard to separate this into blocks because of the skip connections

class XNet(nn.Module):
    def __init__(self, n_classes: int = 2):
        super(XNet, self).__init__()
        self.conv1 = self.conv_block(1, 64)     # because grayscale (x-rays) have only one color channel
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 512)  # flat block
        self.conv6 = self.conv_block(512, 256)
        self.conv7 = self.conv_block(256, 128)
        self.conv8 = self.conv_block(128, 128)
        self.conv9 = self.conv_block(128, 256)
        self.conv10 = self.conv_block(256, 512)
        self.conv11 = self.conv_block(512, 512)
        self.conv12 = self.conv_block(512, 256)
        self.conv13 = self.conv_block(256, 128)
        self.conv14 = self.conv_block(128, 64)
        
        self.conv15 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, padding='valid')
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.UpSample_64 = nn.Upsample(size=64)
        self.UpSample_128 = nn.Upsample(size=128)
        self.UpSample_256 = nn.Upsample(size=256)
        self.SoftMax = nn.Softmax(n_classes)
        self.n_classes = n_classes
    
    def conv_block(self, in_channels: int, out_channels: int)->nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),               # need to pass the number of channels
            nn.ReLU()
        )
    
    def forward(self, x):
        act_1 = self.conv1(x)
        pool_1 = self.MaxPool(act_1)

        act_2 = self.conv2(pool_1)
        pool_2 = self.MaxPool(act_2)

        act_3 = self.conv3(pool_2)
        pool_3 = self.MaxPool(act_3)

        act_4 = self.conv4(pool_3)
        act_5 = self.conv5(act_4)

        up_6 = self.UpSample_64(act_5)
        act_6 = self.conv6(up_6) + act_3

        up7 = self.UpSample_128(act_6)
        act_7 = self.conv7(up7) + act_2

        act_8 = self.conv8(act_7)
        pool_8 = self.MaxPool(act_8)

        act_9 = self.conv9(pool_8)
        pool_9 = self.MaxPool(act_9)

        act_10 = self.conv10(pool_9)
        act_11 = self.conv11(act_10)

        up12 = self.UpSample_64(act_11)
        act_12 = self.conv12(up12) + act_9

        up13 = self.UpSample_128(act_12)
        act_13 = self.conv13(up13) + act_8

        up14 = self.UpSample_256(act_13)    
        act_14 = self.conv14(up14) + act_1

        # need a 1 x 1 convolution to reduce the depth of the tensor, spatial dimensions retained
        act_15 = self.conv15(act_14)            # this should reduce the channel depth to the number of classes specified (1x1 conv)

        # reshape and softmax
        x = act_15.view(-1, self.n_classes * 256 * 256)
        x = F.softmax(x)
        return x