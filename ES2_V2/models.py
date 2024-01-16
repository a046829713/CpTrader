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
        x = self.fc4(x) # Use only if appropriate
        return x

    
