import torch
import torch.nn as nn

class ConvetNet(nn.Module):
    def __init__(self):
        super(ConvetNet, self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.MaxPool2d(3,2),
            nn.ReLU(),

            nn.Conv2d(32,64,3),
            nn.MaxPool2d(3,2),
            nn.ReLU(),

            nn.Conv2d(64,128,3),
            nn.MaxPool2d(3,2),
            nn.ReLU(),)
        self.fc_layers=nn.Sequential(

        )