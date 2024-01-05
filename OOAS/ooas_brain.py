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
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 建立 LSTM 層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全連接層，將 LSTM 的輸出映射到類別數
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隱藏狀態和單元狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向傳播 LSTM
        out, _ = self.lstm(x, (h0, c0))        
        # 取出最後一個時間步的輸出來進行分類        
        out = self.fc(out[:, -1, :])
        
        # 根據論文的說法 他應該是使用softmax來輸出
        out = nn.functional.softmax(out, dim=1)        
        return out


    
class PolicyGradientAgent():
    def __init__(self, lr, num_classes, gamma = 1) -> None:
        """

        Args:
            lr (_type_): _description_
            gamma (int, optional): _description_. Defaults to 1.
            num_classes (int, optional): 輸出類別數. Defaults to 3.
        """
        self.gamma = gamma # 論文認為要尋找的是最好的整體交易策略,當gamme等於1,等於整體回報
        self.lr = lr        
        
        # 模型參數設定
        input_size = 5  # 特徵數
        hidden_size = 64  # LSTM 單元數
        num_layers = 3  # LSTM 層數
        # dropout_rate = 0.2  # Dropout 比率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(self.device)
        

     