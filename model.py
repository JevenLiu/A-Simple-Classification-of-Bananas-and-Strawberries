import torch.nn as nn
import torch


class MyNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 假设输入图片是100*100，channel=3
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            # 输出是16*96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 输出是16*48*48
        )

        self.layer2 = nn.Sequential(
            # 输入是16*48*48
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            # 输出是32*44*44
            nn.ReLU(inplace=True),
            # 输入是32*44*44
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 输出是32*22*22
        )
        self.fc = nn.Linear(32 * 22 * 22, num_classes)

    # 正向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x