import torch
x = torch.arange(1,4).view(2,2)
print(x[1,1])
print(x[1,1].item())
y = torch.rand(5,3)

#print(x+y)
