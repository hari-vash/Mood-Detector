import torch.nn as nn
import torch

class Face_Detection(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2,1),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,1),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,1),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,1),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size =3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )

        self.classifier1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,512),
            nn.ReLU()
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU()
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(256,5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == '__main__':
    bn_model = Face_Detection()
    x = torch.randn(1,1,48,48)
    print('Shape of output = ',bn_model(x).shape)
    print('No of Parameters of the BatchNorm-CNN Model =',bn_model.count_parameters())