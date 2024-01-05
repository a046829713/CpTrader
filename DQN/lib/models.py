import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full(
            (out_features, in_features), sigma_init))
        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class SimpleFFDQN(nn.Module):
    """

    """

    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        # 值函數部分 (Value function):
        # 這部分是Dueling DQN中的狀態價值函數部分。
        # 給定一個狀態，它嘗試估計該狀態的值。它由三層全連接層和ReLU激活函數組成。
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 優勢函數部分 (Advantage function):
        # 這部分是Dueling DQN中的優勢函數部分。給定一個狀態，它嘗試估計每個動作的優勢。
        # 結構和值函數部分非常相似，但最後一層的輸出大小為actions_n，代表每個動作的優勢值。
        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        # Q(s,a)=V(s)+(A(s,a)−mean(A(s,a′ )))
        # 這是模型的前向傳播方法。給定一個輸入狀態x，它先計算該狀態的價值val和每個動作的優勢值adv。
        # 然後，根據Dueling DQN的公式，它結合這兩個部分來計算動作價值函數(Q函數)的估計。

        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        """捲積網路的參數詳解
            in_channels = 特徵的數量
            out_channels 出來的特徵數量(如果有100張,可以想像就是有100張小圖)
            kernel_size (濾波器的大小)
            stride = (移動步長)
            padding =補0            
            
            # 這邊指的是input 是指輸入的特徵,口訣就是批次大小,通道數,數據長度
            
            input_data 是三维的，因为卷积层（nn.Conv1d）的输入需要具有特定的形状，以便正确地进行卷积操作。这三个维度分别代表：

            批次大小（Batch Size）：

            代表一次输入到网络中的样本数量。通过一次处理多个样本（一个批次），网络可以加速训练，并且可以得到梯度的更稳定估计。
            
            通道数（Channels）：

            对于图像，通道通常代表颜色通道（例如，RGB）。在其他类型的数据中，通道可以代表不同类型的输入特征。
            在一维卷积中，通道数可能对应于输入数据的不同特征。
            
            数据长度（Data Length）：

            这是输入数据的实际长度。对于图像，这会是图像的宽度和高度。对于一维数据，如时间序列，这会是序列的长度。
        Args:
            shape (_type_): _description_
            actions_n (_type_): _description_
        """
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        """用來計算flatten後的參數個數

        Args:
            shape (_type_): _description_

        Returns:
            _type_: _description_
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # 要丟進去一般網絡之前要先flatten
        conv_out = self.conv(x).view(x.size()[0], -1) #執行flatten的動作
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)
    
    # def script_model(self, example_input):
    #     return torch.jit.script(self, example_input)

class DQNConv1DLarge(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 64, 5),
            nn.ReLU(),
            nn.Conv1d(64, 32, 5),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)


class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(CustomNet, self).__init__()
        """
            用來定義 evolution reinforcement learning 
        """
        
        # 定義全連接層
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # 通過第一層全連接層和 ReLU 激活函數
        x = F.relu(self.fc1(x))
        # 通過第二層全連接層和 ReLU 激活函數
        x = F.relu(self.fc2(x))
        # 通過第三層全連接層和 log softmax
        x = F.log_softmax(self.fc3(x), dim=0)
        return x