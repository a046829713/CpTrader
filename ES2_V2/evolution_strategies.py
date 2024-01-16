"""
github link : https://github.com/huseinzol05/Stock-Prediction-Models

to tranfor pytorch and change code 

"""
from .models import NeuralNetwork
from .environment import Env
import time
import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
from utils.AppSetting import AppSetting
import copy 
from .models import NeuralNetwork
from .common import model_check
from torch.nn import functional as F


class NeuroEvolution:
    def __init__(self,
                 population_size,
                 mutation_rate                   
                 ):
        
        self.population_size = population_size # 人口數
        self.mutation_rate = mutation_rate # 變異率
        # 環境準備 細分成訓練和測試
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_env = Env(data_type='train_data', random_ofs_on_reset = False)
        self.val_env = Env(data_type='test_data', random_ofs_on_reset = False)
        self.setting = AppSetting.get_es_setting()
        self.save_path = os.path.join(self.setting['SAVES_PATH'], datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.setting['BARS_COUNT']) + 'k')        
        self.env_info_static = self.train_env.info_provider()
        self.populations = [] # 用來放置所有的神經網絡
        self.best_fitness = None        
        self.count_play_steps = 0 # 用來轉換環境(env)的alpha
        self.model_generator = NeuralNetwork
        
        
        # 紀錄
    def reset(self):
        """
            用來重置需要重置的地方
        """
        self.populations = []
        
    def _initialize_population(self):
        """
            每个神经网络都是一个潜在的解决方案，用于预测股票市场的买卖时机。
        """
        for i in range(self.population_size):
            model = copy.deepcopy(self.pretrained_model).to(self.device)
            # 對模型進行輕微的隨機變異
            # model = self.mutate(model, scale=1)  # scale 參數控制變異程度，可以根據需要進行調整            
                
            self.populations.append(model)
    
        
    def test_model(self, agent: dict, env):
        """
            測試模型在給定環境上的表現。
            Args:
                agent (nn.Module):  神經網絡
                env (Env): 要進行測試的環境，可以是訓練環境或驗證環境。
                                
        """
        done = False
        state = torch.from_numpy(env.reset()).view(1, self.env_info_static['shapeOfFeature'][0]).to(self.device)        
        rewards = []
        actions =[]
        probs =[]
        N_step = 0
        while not done:            
            self.count_play_steps +=1 # 用來轉換環境(env)的alpha
            policy = agent(state)
            probs.append(policy)
            policy_softmax = F.softmax(policy, dim=1)  # 在這裡應用softmax
            # 我在設計這個ES 策略的時候認為智能體為了要評估好壞 應該要使用max來取得最好的動作
            # 並且在最後動作並沒有特別針對有的部位歸零        
            action = policy_softmax.max(dim=1)[1].item() 
            actions.append(action)
            state_, reward, done, info = env.step(action,self.count_play_steps)
            rewards.append(reward)
            state = torch.from_numpy(state_).view(1, self.env_info_static['shapeOfFeature'][0]).to(self.device)            

            N_step +=1            
            if N_step >1000:
                break

        train_fitness = sum(rewards)  / self.setting['INITIAL_INVESTMENT']         
        return train_fitness,probs,actions,rewards
    

    
    def evaluate_population(self, env,epoch):
        """
        評估給定環境中代理人群體的適應度。
        Args:
            pop (list): 代理人群體。
            env (Env): 要進行評估的環境。

        """
        if epoch == 0:
            train_fitness = None            
            for agent in self.populations:
                if train_fitness is None:
                    train_fitness,probs,actions,rewards = self.test_model(agent, env)
                
                if env.env_data_type =='train_data':
                    agent.train_fitness = train_fitness # 最後計算出來的就是用來產生分數就是競賽條件
        else:        
            for agent in self.populations:
                train_fitness,probs,actions,rewards = self.test_model(agent, env)
                if env.env_data_type =='train_data':
                    agent.train_fitness = train_fitness # 最後計算出來的就是用來產生分數就是競賽條件
                

    
    def inherit_weights(self, parent, child1):
        child1.load_state_dict(copy.deepcopy(parent.state_dict()))        

    
    def crossover(self,parent1, parent2):
        child1 = self.model_generator(self.env_info_static['shapeOfFeature'][0]).to(self.device)
        child2 = self.model_generator(self.env_info_static['shapeOfFeature'][0]).to(self.device)        
        # 确保子代模型已经继承了父代模型的权重
        self.inherit_weights(parent1, child1)
        self.inherit_weights(parent2, child2)
        self._crossver(child1,parent2)
        self._crossver(child2,parent1)        
        return child1,child2
                        
    def _crossver(self,child,parent):
        # 为每层权重设置一个随机切割点并交换权重
        for (name1, param1) in child.named_parameters():
            if 'weight' in name1:  # 确保只交换权重矩阵，不交换偏差
                cutoff = np.random.randint(0, param1.size(0))
                for (name2, param2) in parent.named_parameters():
                    if name2 == name1:
                        param1.data[cutoff:]= param2.data[cutoff:].clone()
    
    def mutate(self, network, scale=1.0):
        with torch.no_grad():  # 确保在不跟踪梯度的情况下进行操作
            for (name, param) in network.named_parameters():
                if param.requires_grad:
                    # 直接使用PyTorch操作进行变异
                    mutation_mask = torch.bernoulli(torch.full(param.shape, self.mutation_rate, device=param.device))
                    normal_data = torch.randn(param.shape, device=param.device) * scale * mutation_mask
                    param.add_(normal_data)  # 使用 in-place 操作来更新权重                    
    
        return network

    
    def save_best_agent(self, index:int ,best_agent):        
        os.makedirs(self.save_path, exist_ok=True)
        checkpoint = {            
            'model_state_dict': best_agent.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_path ,f'best_agent_params_{index}_{best_agent.train_fitness}.pt'))
    
    
    def evolve(self, generations=20, checkpoint= 1):
        # 產生第一批神經
        self._initialize_population()
        n_winners = int(self.population_size * 0.4)
        n_parents = self.population_size - n_winners        
        
        
        offset = 1e-10
        
        for epoch in range(generations):
            self.evaluate_population(self.train_env,epoch)            
            fitnesses = [i.train_fitness for i in self.populations]
            # 取得排序強度
            sort_fitness = np.argsort(fitnesses)[::-1]
            self.populations = [self.populations[i] for i in sort_fitness]     
            fittest_individual = self.populations[0] 
            
            print(f"第幾次:{epoch},此次分數狀況:",[i.train_fitness for i in self.populations])
            if epoch == 1:
                break
            
            # 為避免零機率，給每個適應度加上一個小的正數偏移量            
            normalized_fitnesses = [abs(f) + offset for f in fitnesses] 
            next_population = [self.populations[i] for i in range(n_winners)]
            # 計算正規化後的適應度比例
            total_fitness = sum(normalized_fitnesses)
            parent_probabilities = [f / total_fitness for f in normalized_fitnesses]           

            parents = np.random.choice(self.populations, size=n_parents, p=parent_probabilities, replace=False)            
                        
            for i in np.arange(0, len(parents), 2):                
                child1, child2 = self.crossover(parents[i], parents[i+1])                
                next_population += [self.mutate(child1), self.mutate(child2)]            
            self.populations = next_population
            
            
        return fittest_individual
# if __name__ == "__main__":
#     population_size = 20
#     mutation_rate = 0.1
#     neural_evolve = NeuroEvolution(population_size, mutation_rate, NeuralNetwork) 
#     fittest_nets = neural_evolve.evolve(50)