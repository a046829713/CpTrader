import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import time

class LSTMFeatureExtractor(nn.Module): #A
    def __init__(self, input_size, hidden_size = 32, num_layers=3):
        """
        
            用來處理輸入的特徵
        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            num_layers (int, optional): _description_. Defaults to 1.
        """
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        _, (h_n, _) = self.lstm(x, (h0, c0))
        # 為甚麼要使用h_n這個,是為了要提取特徵
        return h_n[-1]  # shape: [batch_size, hidden_size]

class Gnet(nn.Module): #B
    def __init__(self):
        super(Gnet, self).__init__()
        self.linear1 = nn.Linear(64,256)
        self.linear2 = nn.Linear(256,3)

    def forward(self, state1,state2):
        x = torch.cat( (state1, state2) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y

class Fnet(nn.Module): #C
    def __init__(self):
        super(Fnet, self).__init__()
        self.linear1 = nn.Linear(33,256)
        self.linear2 = nn.Linear(256,32)

    def forward(self,state,action):
        action_ = action.float().unsqueeze(1)  # action_.shape = [batch_size, 1] 
        x = torch.cat( (state,action_) ,dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)        
        return y
    
    
class ICM_network():
    def __init__(self,raw_state) -> None:
        """
            raw_state(numpy.ndarray): is env to provider 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        inputs_dims = raw_state.shape[1]        
        self.encoder = LSTMFeatureExtractor(inputs_dims).to(self.device)
        self.forward_model = Fnet().to(self.device)
        self.inverse_model = Gnet().to(self.device)
        
        self.forward_loss = nn.MSELoss(reduction='none')
        self.inverse_loss = nn.CrossEntropyLoss(reduction='none')