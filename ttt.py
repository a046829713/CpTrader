import torch
import torch.nn as nn
import math

entroy=nn.CrossEntropyLoss()
input=torch.Tensor([[-0.7715, -0.6205,-0.2562]])
print(input)
target = torch.tensor([0])
output = entroy(input, target)
print(output)
#根据公式计算

