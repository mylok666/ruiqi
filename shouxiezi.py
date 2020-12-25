from shouxiezitest import test
from shouxiezitrain import *

for epoch in range(1,n_epochs+1):
    train(epoch) #如果不训练直接test 就等于用没训练的模型产生数据 先train再test test里的model.eval用的就是训练好的模型
    test()
    from torch.utils.tensorboard import SummaryWriter

    #input = torch.rand(12, 1, 28, 28)
    #writer = SummaryWriter('./data')
    #writer.add_graph(network, (input,))  #模型可视化

import matplotlib.pyplot as plt
fig = plt.figure()
print(len(train_counter))
print(len(train_loader))
print(len(test_counter))
print(len(test_losses))
plt.plot(train_counter, train_losses, color='blue')
#plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

#可视化结果
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
  output = network(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()