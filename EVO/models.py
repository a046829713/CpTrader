import torch.nn as nn
import torch.nn.functional as F


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
        """_summary_

        Args:
            x like this: tensor([[ 54.5496,   8.6316, -21.4676]], device='cuda:0', grad_fn=<ViewBackward0>)
            x size = torch.Size([1,201])
                
        Returns:
            _type_: _description_
        """
        # 通過第一層全連接層和 ReLU 激活函數
        x = F.relu(self.fc1(x))
        # 通過第二層全連接層和 ReLU 激活函數
        x = F.relu(self.fc2(x))
        # 通過第三層全連接層和 log softmax
        x = F.log_softmax(self.fc3(x), dim=1)
        return x