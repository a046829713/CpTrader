import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import time


class LSTMModel(nn.Module):
    """        
    Args:
        policyNetwork 在強化學習中扮演這樣的角色
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 建立 LSTM 層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全連接層，將 LSTM 的輸出映射到類別數
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate) # 0.1
         
    def forward(self, x):
        # 初始化隱藏狀態和單元狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向傳播 LSTM
        out, _ = self.lstm(x, (h0, c0)) 
        lstm_out = out[:, -1, :]
        # 丟棄部份
        drop_out = self.dropout(lstm_out)
        out = self.fc(drop_out)  
        # 应用对数 softmax
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

class Critic(nn.Module):    
    def __init__(self,obs_len):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(obs_len,25) # 
        self.l2 = nn.Linear(25,50)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)
    
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))        
        c = F.relu(self.l3(y))
        critic = torch.tanh(self.critic_lin1(c)) #D
        return critic #E



class ActorCriticAgent():
    def __init__(self,lr ,raw_state, gamma = 1) -> None:
        """

        Args:
            lr (_type_): _description_
            inputs_dims (_type_): 特徵數
            gamma (int, optional): _description_. Defaults to 1.
            n_actions (int, optional): _description_. Defaults to 3.
        """
        self.gamma = gamma # 論文認為要尋找的是最好的整體交易策略,當gamme等於1,等於整體回報
        self.lr = lr
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型參數設定
        hidden_size = 32  # LSTM 單元數
        num_layers = 3  # LSTM 層數
        num_classes = 2 # 輸出類別數
        dropout_rate = 0.2  # Dropout 比率        
        
        inputs_dims = raw_state.shape[1]
        critic_dims = torch.from_numpy(raw_state).float().view(-1)  # 同样扁平化为 [250]
        
        self.policy = LSTMModel(inputs_dims, hidden_size, num_layers, num_classes, dropout_rate).to(self.device)
        self.critic = Critic(critic_dims.shape[0]).to(self.device)
        
    
    def choose_action(self, observation):
        """
            根據環境來選擇,動作
        Args:
            observation (_type_): _description_
        """
        
        state = torch.Tensor([observation]).to(self.policy.device)
        probabilities = F.softmax(self.policy.forward(state))
        actions_probs = torch.distributions.Categorical(probabilities)
        # 這個sample 測試的結果為隨機的
        action = actions_probs.sample()
        return action.item()   

    