import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

torch.manual_seed(2020) #对于可重复的实验，我们必须为任何使用随机数产生的东西设置随机种子

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./data/',train=False,download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
  plt.savefig('./test2.jpg')
plt.show()
#
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(batch_idx)  #938
#     print(data.view(-1,28,28).unsqueeze(1).shape)  #torch.Size([64, 1, 28, 28])/torch.Size([32, 1, 28, 28])
#     print(target.shape)  #torch.size([64])/torch.size([32])
#
# for index,(data, target) in enumerate(test_loader):
#     print(index)  # 10
#     print(data.shape)  # torch.Size([1000, 1, 28, 28])
#     print(target.shape)  # torch.Size([1000])
#
# for data,target in test_loader:
#     print(data.shape)  #torch.Size([1000, 1, 28, 28])
#     print(target.shape)  #torch.Size([1000])
#


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  #(1,28,28)-->(10,12,12)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  #(10,12,12)-->(20,4,4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)  #20*4*4=320
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  #为了dropout能起作用，training状态一直是默认值False得改成training=self.training
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net()
print(network)
from torch.utils.tensorboard import SummaryWriter
#input = torch.rand(12,1,28,28)
# writer = SummaryWriter('./data')
# writer.add_graph(network,(input,))
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
#设置学习率衰减
scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8) #在scheduler设置学习率衰减的策略
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
print(test_counter) #[0, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000, 540000, 600000, 660000, 720000, 780000, 840000, 900000, 960000, 1020000, 1080000, 1140000, 1200000, 1260000, 1320000, 1380000, 1440000, 1500000, 1560000, 1620000, 1680000, 1740000, 1800000, 1860000, 1920000, 1980000, 2040000, 2100000, 2160000, 2220000, 2280000, 2340000, 2400000, 2460000, 2520000, 2580000, 2640000, 2700000, 2760000, 2820000, 2880000, 2940000, 3000000, 3060000, 3120000, 3180000, 3240000, 3300000, 3360000, 3420000, 3480000, 3540000, 3600000, 3660000, 3720000, 3780000, 3840000, 3900000, 3960000, 4020000, 4080000, 4140000, 4200000, 4260000, 4320000, 4380000, 4440000, 4500000, 4560000, 4620000, 4680000, 4740000, 4800000, 4860000, 4920000, 4980000, 5040000, 5100000, 5160000, 5220000, 5280000, 5340000, 5400000, 5460000, 5520000, 5580000, 5640000, 5700000, 5760000, 5820000, 5880000, 5940000, 6000000]

print(train_loader.dataset)
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import cv2
from PIL import Image
def train(epoch):
    network.train()
    '''
    model.train和model.eval区别  model.train(训练用)启用 BatchNormalization 和 Dropout  model.eval(测试用) 不启用 BatchNormalization 和 Dropout
    '''
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        '''
        batch_idx:1,2,3...973
        data:张量 数字组成的矩阵
        target：这个数字代表什么
        '''
        '''
        1.模型梯度清零
        2.输入数据 产生结果
        3.与目标值对比（这里的任务是 模型经过softmax生成的标签值与真实target对比） nll_loss损失函数
        4.损失反向传播
        5.学习率更新
        '''
        optimizer.zero_grad()  #每次训练都梯度清零
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()  #反向传播
        optimizer.step()
        if batch_idx % log_interval == 0: #每十次打印数据
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())  #item是得到一个元素张量里面的元素值   把每次训练损失保存下来
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model/model.pth')
            torch.save(optimizer.state_dict(), './model/optimizer.pth')
            #神经网络模块以及优化器能够使用.state_dict()保存和加载它们的内部状态。这样，如果需要，
            #我们就可以继续从以前保存的状态dict中进行训练——只需调用.load_state_dict(state_dict)


'''change for '''