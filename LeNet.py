import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



BATCH_SIZE = 32
EPOCHS = 10
pipeline = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=0, std=1)])
train = datasets.MNIST("data", download=True, train=True, transform=pipeline)
test = datasets.MNIST("data", download=True, train=False, transform=pipeline)

train_loader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, shuffle=True, batch_size=BATCH_SIZE)

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

# 搭建模型
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

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

x_l = []
train_l_y = []
test_l_y = []

# 训练函数
def train_func(train_loader, net, loss, optimer, epochs):
    for epoch in range(epochs):
        train_i = 0
        train_l = 0
        test_i = 0
        test_l = 0
        x_l.append(epoch+1)
        train_acc = 0
        for x, y in train_loader:
            train_i += 1
            pred = net(x)
            train_acc += accuracy(pred=pred, y=y)
            l = loss(pred, y)
            train_l += l
            optimer.zero_grad()
            l.backward()
            optimer.step()
        test_acc = 0
        for x, y in test_loader:
            test_i += 1
            pred = net(x)
            test_acc += accuracy(pred=pred, y=y)
            l = loss(pred, y)
            test_l += l
        torch.save(net.state_dict(), f"./module/linear_{epoch+1}.pth")
        train_l_y.append((train_l/train_i).detach().numpy())
        test_l_y.append((test_l/test_i).detach().numpy())
        print(f"epoch:{epoch+1}  train_loss:{train_l/train_i:2f}  train_acc:{(train_acc/train_i):2f}  test_loss:{test_l/test_i:2f}  test_acc:{(test_acc/test_i):2f}")

def accuracy(pred, y):
    pred = F.softmax(pred)
    pred = pred.argmax(dim=1)
    mask = (pred == y)
    a = mask.sum()/len(y)
    return a

# 绘图函数
def draw_func(x, y):
    np.save("./lenet_data/liner_x.npy", x)
    np.save("./lenet_data/liner_y.npy", y)
    fig, ax = plt.subplots()
    train_y, test_y = y
    ax.plot(x, train_y, label="train")
    ax.plot(x, test_y, label="test")
    ax.set_ylabel("loss")
    plt.legend()
    fig.savefig("./img/lenet.jpg")
    fig.show()

net = LeNet()
loss = nn.CrossEntropyLoss()
optimer = optim.SGD(params=net.parameters(), lr=0.2)
train_func(train_loader, net=net, loss=loss, optimer=optimer, epochs=EPOCHS)
draw_func(x_l, (train_l_y, test_l_y))
