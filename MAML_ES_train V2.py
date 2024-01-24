""" 最終目標是透過訓練遍歷所有 股票來訓練一個可以泛化的股票模型 """
import torch
from torch import nn, optim
from ES2_V2.models import NeuralNetwork
from ES2_V2.evolution_strategies import NeuroEvolution
from ES2_V2.environment import Env
from ES2_V2.common import model_check
import time
from torch.nn import functional as F


class MAML():
    """
    用於實現整個MAML的演算法
    """
    def __init__(self,beta=0.03) -> None:
        """_summary_

        Args:
            beta (float, optional): 元更新的的學習率. Defaults to 0.001.
        """
        self.env_info_static = Env(data_type='test_data', random_ofs_on_reset = False).info_provider()      
        self.neuroEvolution = NeuroEvolution(population_size=16,mutation_rate=0.1)
        
        # GPU
        self.maml_model = NeuralNetwork(input_size=self.env_info_static['shapeOfFeature'][0]).to(self.neuroEvolution.device) # 全域權重 
        self.optimizer = optim.Adam(self.maml_model.parameters(), lr=beta) # 全域優化器
                
        self.tasks = ["BTCUSDT"]  # 定义任务分布
        self.maml_train(num_iterations=1000)
    
    # def calculate_loss(self,adapted_model):
    #     # 元策略選擇商品機率
    #     _,maml_model_probs, maml_model_actions, maml_model_rewards = self.neuroEvolution.test_model(self.maml_model,self.neuroEvolution.train_env)
    #     _,adapted_model_probs, adapted_model_actions,adapted_model_rewards = self.neuroEvolution.test_model(adapted_model,self.neuroEvolution.train_env)

        
    #     maml_model_probs = torch.stack(maml_model_probs).squeeze(1)
    #     adapted_model_actions = torch.tensor(adapted_model_actions).to(self.neuroEvolution.device)


    #     return  nn.CrossEntropyLoss()(maml_model_probs, adapted_model_actions)

    def calculate_loss(self,adapted_model,gamma=1):
        # 元策略選擇商品機率
        _, maml_model_logprobs, maml_model_rewards = self.neuroEvolution.test_model(self.maml_model,self.neuroEvolution.train_env)
        _, adapted_model_logprobs,adapted_model_rewards = self.neuroEvolution.test_model(adapted_model,self.neuroEvolution.train_env)
        # 將rewards,logprobs,的元素順序顛倒,以方便計算折扣回報陣列
        logprobs = torch.stack(maml_model_logprobs).flip(dims=(0,))
        print(k)
        rewards = torch.Tensor(adapted_model_rewards).flip(dims=(0,)).to(self.neuroEvolution.device)
        Returns = []
        ret_ = 0
        for r in range(rewards.shape[0]): #B
            ret_ = rewards[r] + gamma * ret_
            Returns.append(ret_)
        
        Returns = torch.stack(Returns).view(-1)
        Returns = F.normalize(Returns,dim=0)
        
        loss = -1* logprobs * Returns
        return loss.sum()
    
    
    def maml_train(self, num_iterations):
        """
        训练MAML模型。

        Args:            
            num_iterations (int): 訓練迭代次數。
        """
        for iteration in range(num_iterations):            
            # 定義任務分布
            for symbol in self.tasks:
                self.neuroEvolution.pretrained_model = self.maml_model
                self.neuroEvolution.reset()
                # ES 根據論文描述只要迭代一次就好 避免過度擬合單一股票
                adapted_model = self.neuroEvolution.evolve(2) # 進化一次
                loss = self.calculate_loss(adapted_model)
                self.optimizer.zero_grad()
                # # 进行元更新
                loss.backward()
                print(loss)
                # model_check("udpate-berfore:",self.maml_model)                
                self.optimizer.step()
                # model_check("udpate-after:",self.maml_model)                
                      
                
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}: Meta Loss {loss.item()}")
                             
                






if __name__ == '__main__':
    MAML()