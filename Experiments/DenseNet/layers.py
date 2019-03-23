import torch
from torch import nn
import torch.nn.functional as F
from math import floor

class SubBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, bottleneck, p):
        
        super(SubBlock, self).__init__()
        self.bottleneck = bottleneck
        self.p = p

        in_channels_2 = in_channels
        out_channels_2 = out_channels

        if bottleneck:
            in_channels_1 = in_channels
            out_channels_1 = out_channels * 4
            in_channels_2 = out_channels_1

            self.bn1 = nn.BatchNorm2d(in_channels_1)
            self.conv1 = nn.Conv2d(in_channels_1,
                                   out_channels_1,
                                   kernel_size=1)

        self.bn2 = nn.BatchNorm2d(in_channels_2)
        self.conv2 = nn.Conv2d(in_channels_2, 
                               out_channels_2, 
                               kernel_size=3, 
                               padding=1)

    def forward(self, x):
        
        if self.bottleneck:
            out = self.conv1(F.relu(self.bn1(x)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)
            out = self.conv2(F.relu(self.bn2(out)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)
        else:
            out = self.conv2(F.relu(self.bn2(x)))
            if self.p > 0:
                out = F.dropout(out, p=self.p, training=self.training)  
        return torch.cat((x, out), 1)

class DenseBlock(nn.Module):
    
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck, p):
        
        super(DenseBlock, self).__init__()

        # create L subblocks
        layers = []
        for i in range(num_layers):
            cumul_channels = in_channels + i * growth_rate
            layers.append(SubBlock(cumul_channels, growth_rate, bottleneck, p))

        self.block = nn.Sequential(*layers)
        self.out_channels = cumul_channels + growth_rate

    def forward(self, x):
        
        out = self.block(x)
        return out

class TransitionLayer(nn.Module):
    
    def __init__(self, in_channels, theta, p):
        
        super(TransitionLayer, self).__init__()
        self.p = p
        self.out_channels = int(floor(theta*in_channels))

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, 
                              self.out_channels, 
                              kernel_size=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.pool(self.conv(F.relu(self.bn(x))))
        if self.p > 0:
            out = F.dropout(out, p=self.p, training=self.training)
        return out