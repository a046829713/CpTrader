import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class DQNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, actions_n):
        """
        LSTM模型用於時間序列數據處理。

        Args:
            input_size (int): 輸入特徵的數量。
            hidden_size (int): LSTM單元的隱藏層大小。
            num_layers (int): 網絡中LSTM層的數量。
            actions_n (int): 可行動作的數量。
        """
        super(DQNLSTM, self).__init__()

        # LSTM層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 值函數網絡
        self.fc_val = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 優勢函數網絡
        self.fc_adv = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        # LSTM層輸出
        lstm_out, _ = self.lstm(x)
        # 取LSTM最後一個時間步的輸出
        lstm_out = lstm_out[:, -1, :]

        # 值函數和優勢函數的計算
        val = self.fc_val(lstm_out)
        adv = self.fc_adv(lstm_out)

        # 組合最終Q值
        return val + adv - adv.mean(dim=1, keepdim=True)