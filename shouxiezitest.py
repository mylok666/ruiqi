import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from shouxiezitrain import *


def test():
    network.eval()  #Dropout与BatchNormalization 会让模型的梯度发生变化 所以测试之前先model.eval
    test_loss = 0
    correct = 0
    with torch.no_grad():  #测试时不进行梯度计算
        for data, target in test_loader:
            output = network(data)
            print("output:",output.shape)
            print('target:',target.shape)
            test_loss += F.nll_loss(output, target, size_average=False).item() #加了itemm之前 输出的是：loss= tensor(X) 有了item之后：loss=X 就是把tensor变回数据
            print('test_loss',test_loss)
            pred = output.data.max(1, keepdim=True)[1] ## 找到概率最大的下标

            correct += pred.eq(target.data.view_as(pred)).sum()  #eq()函数功能是判断两个东西是不是一个东西，会返回true和false 然后通过.sum函数把正确的加起来返回给correct
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


