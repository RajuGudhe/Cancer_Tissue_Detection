import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import sqrt

from layers import *

class DenseNet(nn.Module):
   
    def __init__(self, 
                 num_blocks, 
                 num_layers_total, 
                 growth_rate, 
                 num_classes, 
                 bottleneck, 
                 p, 
                 theta):
        
        super(DenseNet, self).__init__()

         
        error_msg = "[!] Total number of layers must be 3*n + 4..."
        assert (num_layers_total - 4) % 3 == 0, error_msg

        # compute L, the number of layers in each dense block
        # if bottleneck, we need to adjust L by a factor of 2
        num_layers_dense = int((num_layers_total - 4) / 3)
        if bottleneck:
            num_layers_dense = int(num_layers_dense / 2)

        
        # initial convolutional layer
        out_channels = 16
        if bottleneck:
            out_channels = 2 * growth_rate
        self.conv = nn.Conv2d(3,
                              out_channels, 
                              kernel_size=3,
                              padding=1)
        
        # dense blocks and transition layers 
        blocks = []
        for i in range(num_blocks - 1):
            # dense block
            dblock = DenseBlock(num_layers_dense, 
                                out_channels, 
                                growth_rate, 
                                bottleneck, 
                                p)
            blocks.append(dblock)

            # transition block
            out_channels = dblock.out_channels
            trans = TransitionLayer(out_channels, theta, p)
            blocks.append(trans)
            out_channels = trans.out_channels
        
        # last dense block does not have transition layer
        dblock = DenseBlock(num_layers_dense, 
                            out_channels, 
                            growth_rate, 
                            bottleneck, 
                            p)
        blocks.append(dblock)
        self.block = nn.Sequential(*blocks)
        self.out_channels = dblock.out_channels
        

       
        # fully-connected layer
        self.fc = nn.Linear(self.out_channels, num_classes)
       

        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
     

    def forward(self, x):
        
        out = self.conv(x)
        out = self.block(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.out_channels)
        out = self.fc(out)
        return out