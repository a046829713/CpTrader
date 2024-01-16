import torch.nn as nn
import torch.nn.functional as F
import time


class NeuralNetwork(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) neural network.
    
    Args:
        nn.Module (class): PyTorch base class for all neural network modules.
        input_size (int): The size of the input features.
        hidden_size (int, optional): The size of the hidden layers. Defaults to 500.
        output_size (int): The size of the output layer.
    """
    def __init__(self, input_size, hidden_size=500, output_size=3):
        super(NeuralNetwork, self).__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x),dim=1) # Use only if appropriate
        return x

    
# class MAMLModel(nn.Module):
#     """
#         Modal-Agnostic Meta-Learning (MAML)
#     Args:
#         nn (_type_): _description_
#     """
    
#     def __init__(self,output_size=3):
#         super(MAMLModel, self).__init__()
#         # 定义四个具有500个单元的层
#         self.layer1 = nn.Linear(in_features=..., out_features=500)  # 根据您的输入特征大小调整in_features
#         self.layer2 = nn.Linear(500, 500)
#         self.layer3 = nn.Linear(500, 500)
#         self.layer4 = nn.Linear(500, output_size)  # 根据您的输出特征大小调整out_features

#     def forward(self, x, params=None):
#         # 实现前向传播
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         x = torch.relu(self.layer3(x))        
#         x = F.softmax(self.layer4(x), dim=1) # 透過模型選擇可能的動作得出reward,並且比較兩者
#         return x