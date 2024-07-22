import torch
from torch import nn
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(3407)
model = nn.Sequential(OrderedDict([('Conv1', nn.Conv2d(3,32,kernel_size=(3,3),padding=(1,1))),
                                    ('Conv2', nn.Conv2d(32,64,kernel_size=(3,3),padding=(1,1))),
                                    ('ReLU1', nn.ReLU()),
                                    ('MaxPool1',nn.MaxPool2d(kernel_size=(3,3),padding=(1,1))),
                                    ('Conv3',nn.Conv2d(64,64,kernel_size=(3,3),padding=(1,1))),
                                    ('Conv4',nn.Conv2d(64,64,kernel_size=(3,3),padding=(1,1))),
                                    ('ReLU2',nn.ReLU()),
                                    ('MaxPool2',nn.MaxPool2d(kernel_size=(3,3),padding=(1,1))),
                                    ('Conv5',nn.Conv2d(64,64,kernel_size=(3,3),padding=(1,1))),
                                    ('Conv6',nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1))),
                                    ('ReLU3',nn.ReLU()),
                                    ('MaxPool3',nn.MaxPool2d(kernel_size=(3,3),padding=(1,1))),
                                    ('Flatten',nn.Flatten()),
                                    ('Layer1',nn.Linear(512,128)),
                                    ('ReLU3',nn.ReLU()),
                                    ('Layer2',nn.Linear(128,64)),
                                    ('ReLU4',nn.ReLU()),
                                    ('Layer3',nn.Linear(64,7))
                                   ])).to(device)
