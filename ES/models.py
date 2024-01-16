import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size =512, output_size=3):
        super(NeuralNetwork, self).__init__()
        # 定義全連接層
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
# class SharedNeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size=512):
#         super(SharedNeuralNetwork, self).__init__()
#         # 共享層
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x

# class TaskSpecificNeuralNetwork(nn.Module):
#     def __init__(self, shared_model, output_size=3):
#         super(TaskSpecificNeuralNetwork, self).__init__()
#         self.shared_model = shared_model
#         # 任務專屬層
#         self.fc3 = nn.Linear(512, output_size)

#     def forward(self, x):
#         x = self.shared_model(x)
#         x = F.softmax(self.fc3(x), dim=1)
#         return x

# # 實例化共享模型
# shared_model = SharedNeuralNetwork(input_size=...)

# # 為不同的任務創建專屬模型
# task1_model = TaskSpecificNeuralNetwork(shared_model, output_size=3)
# task2_model = TaskSpecificNeuralNetwork(shared_model, output_size=3)