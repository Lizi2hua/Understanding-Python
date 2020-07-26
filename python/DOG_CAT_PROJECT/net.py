import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self,layer1_filter):
        super(ConvNet, self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,layer1_filter,3),
            nn.MaxPool2d(3,2),
            nn.ReLU(),

            nn.Conv2d(layer1_filter,64,3),
            nn.MaxPool2d(3,2),
            nn.ReLU(),

            nn.Conv2d(64,128,3),
            nn.ReLU(),)
        self.fc_layers=nn.Sequential(
            nn.Linear(128*20*20,2),
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        out=self.conv_layers(x)
        # print(out.shape)
        out=out.reshape(-1,128*20*20)

        out=self.fc_layers(out)
        return out

# x=torch.randn([64,3,100,100])
# for i in range(32,256,2):
#     net=ConvNet(layer1_filter=i)
#     y=net(x)
#     print(y)
# net=ConvNet(layer1_filter=32)
# print(net)
