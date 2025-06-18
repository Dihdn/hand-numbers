import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import PIL.Image as PIL


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class LeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(Block(1, 6, 3, 1, 1),
                                  Block(6, 32, 3, 1, 1),
                                  Block(32, 64, 3, 1, 1),
                                  Block(64, 128, 3, 1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(128, 256),
                                    nn.Linear(256, 512),
                                    nn.Linear(512, 10))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


# 加载模型
net = LeNet()
net.load_state_dict(torch.load(r"module\linear_10.pth"))

pipeline = transforms.Compose([transforms.ToTensor(),
                               transforms.Grayscale(),
                               transforms.Resize(size=(28, 28)),
                               transforms.Normalize(mean=0, std=1)])

# 加载图片并进行预处理
img = PIL.open(r"a.jpg")
img = pipeline(img)
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
y = net(img)
num = y.argmax(dim=1)
print(f"图中的数字是{int(num)}")
