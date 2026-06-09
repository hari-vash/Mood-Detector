import torch
import torch.nn as nn
import torch.nn.functional as F

class residualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dropout_prob=0.4):
        super(residualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.drop = nn.Dropout(dropout_prob)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out+=self.shortcut(x)
        out = F.relu(out)
        out = self.drop(out)
        return out
    
class emotionModel(nn.Module):
    def __init__(self,num_classes=3):
        super(emotionModel,self).__init__()
        # Initial conv layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Block 1
        self.block1 = residualBlock(in_channels=32,out_channels=32,dropout_prob=0.4)
        
        # Block 2
        self.block2 = residualBlock(in_channels=32,out_channels=64,dropout_prob=0.4)
        
        # Block 3
        self.block3 = residualBlock(in_channels=64,out_channels=128,dropout_prob=0.4)
        
        # for Downsampling
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        # classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(128,num_classes)
        )
        
    def forward(self,x):
        x = self.init_conv(x)
        
        x = self.block1(x)
        x = self.pool(x)
        
        x = self.block2(x)
        x = self.pool(x)
        
        x = self.block3(x)
        
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x