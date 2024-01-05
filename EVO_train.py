from EVO.environment import Env
import gym
import numpy as np
import time
import os
from utils.AppSetting import AppSetting
from datetime import datetime
from EVO.models import CustomNet
import torch
import torch.nn as nn

class EVO_Train():
    def __init__(self) -> None:
        self.population_size = 20 # 每一代族群中的個體數
        self.mutation_rate = 0.01 # 突變率       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_env = Env(data_type='train_data', random_ofs_on_reset = False)
        self.val_env = Env(data_type='test_data', random_ofs_on_reset = False)
        
        self.env_info_static = self.train_env.info_provider()
        self.setting = AppSetting.get_evo_setting()        
        self.best_fitness = -float('inf')  # 追蹤最高平均分數
        self.model = CustomNet(input_size=self.env_info_static['shapeOfFeature'][0], hidden_size1=512, hidden_size2=512, output_size=3).to(self.device)
        self.save_path = os.path.join(self.setting['SAVES_PATH'], datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.setting['BARS_COUNT']) + 'k')
        self.main_train()
        
    def main_train(self):
        pop = self.spawn_population(N=self.population_size, size=368131) #C # list
        # what pop look like?  ==>  [{}]
        i = 0
        while True:
            i +=1    
            self.evaluate_population(pop,self.train_env)  # 訓練集的平均適應度 list
            self.evaluate_population(pop,self.val_env)  # 驗證集的平均適應度
            for each_net_parameter in pop:
                # 檢查是否需要更新最高平均分數並保存模型
                fit_combined = (each_net_parameter['train_fitness'] + each_net_parameter['test_fitness']) / 2
                if  fit_combined > self.best_fitness:
                    self.best_fitness = fit_combined
                    print("目前最好的平均獎勵:",self.best_fitness)                    
                    self.save_best_agent(i,each_net_parameter)  # 保存最好的代理人參數
            
            pop = self.next_generation(pop, mut_rate=self.mutation_rate,tournament_size=0.2) #E

            
    def save_best_agent(self, index:int ,best_agent):        
        os.makedirs(self.save_path, exist_ok=True)
        self.set_model_params(self.model, self.unpack_params(best_agent['params'])) # 設置模型參數
        checkpoint = {            
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.save_path ,f'best_agent_params_{index}_{self.best_fitness}.pt'))
    
    def get_MAX_drawdown(self,ClosedPostionprofit:list):
        DD_array = np.empty(shape=len(ClosedPostionprofit))
        max_profit = 0
        for i in range(len(ClosedPostionprofit)):
            if ClosedPostionprofit[i] > max_profit:
                max_profit = ClosedPostionprofit[i]
                DD_array[i] = 0
            else:
                DD_array[i] = (ClosedPostionprofit[i] - max_profit)
        return abs(min(DD_array))

    def test_model(self, agent: dict, env):
        """
        測試模型在給定環境上的表現。
        Args:
            agent (dict): 代理人的參數和適應度信息。
                {'params': tensor([ 7.1997e-01,  3.4702e-01, -4.1043e-02,  2.0148e-01,  3.9200e-01,
                    4.7997e-01, -7.8605e-02,  4.5075e-01,  8.3934e-02, -2.8988e-02,
                    9.1045e-01,  6.6314e-01,  1.6042e-01,  1.6423e-01,  7.6869e-02,
                    -1.2855e-01, -3.0153e-01,  1.7491e-01,  9.4341e-02, -2.8848e-01,
                    .....
                    -3.7803e-01, -4.5695e-01, -2.7511e-01, -6.5695e-02,  2.3287e-01,
                    4.0014e-01,  3.7505e-01,  4.2098e-01,  1.4463e-01, -2.5852e-01,
                    4.5289e-01, -3.0700e-03,  2.4202e-01,  3.1440e-01,  7.9634e-01,
                    -8.3145e-01,  4.8587e-01]), 'train_fitness':0.0, 'test_fitness':0.0            
                }
            env (Env): 要進行測試的環境，可以是訓練環境或驗證環境。
        """
        done = False
        state = torch.from_numpy(env.reset()).view(1, self.env_info_static['shapeOfFeature'][0]).to(self.device)
        params = self.unpack_params(agent['params'])            
        self.set_model_params(self.model, params)  # 設置模型參數
        while not done:
            probs = self.model(state)  # 使用 CustomNet 實例
            action = probs.max(dim=1)[1].item()
            state_, reward, done, info = env.step(action)
            state = torch.from_numpy(state_).view(1, self.env_info_static['shapeOfFeature'][0]).to(self.device)
        
        # 計算期望值
        if info['TotalTrades'] == 0:
            return 0

        Win_percent = info['WinTrades'] / info['TotalTrades']
        B_percent = info['WinMoney'] / (abs(info['LossMoney']) +info['Cost'])
        # 勝率 * (賠率) - 輸率 * 1)
        E = Win_percent * B_percent - ((1-Win_percent) * 1)     
        return E * info['TotalTrades']

    def evaluate_population(self, pop: list, env):
        """
        評估給定環境中代理人群體的適應度。
        Args:
            pop (list): 代理人群體。
            env (Env): 要進行評估的環境。

        """
        for agent in pop:
            score = self.test_model(agent, env)
            if env.env_data_type =='train_data':
                agent['train_fitness'] = score
            elif env.env_data_type =='test_data':
                agent['test_fitness'] = score
            
    
    def spawn_population(self, N=50, size=368131): #A 代表族群中的個體數量,size 則是參數向量的參數總數
        pop = []
        for i in range(N):
            vec = torch.randn(size,device=self.device) / 2.0 #B 隨機產生代理人的初始參數向量
            fit = 0
            p = {'params':vec, 'train_fitness':fit, 'test_fitness':fit} #C 將參數向量和適應度分數存入字典中,代表一個代理人的資訊
            pop.append(p)
        return pop
    

    
    def set_model_params(self, model, unpacked_params):
        # 將解包的參數設置到模型中
        with torch.no_grad():
            l1, b1, l2, b2, l3, b3 = unpacked_params
            model.fc1.weight = nn.Parameter(l1)
            model.fc1.bias = nn.Parameter(b1)
            model.fc2.weight = nn.Parameter(l2)
            model.fc2.bias = nn.Parameter(b2)
            model.fc3.weight = nn.Parameter(l3)
            model.fc3.bias = nn.Parameter(b3)

    # 102912 + 262144 +  1536 + 512 + 512 + 3 =
    # 103424 + 262144 +  1536 + 512 + 512 + 3 = 
    def unpack_params(self, params): #A
        """
            製作權重和偏差
            注意權重的順序要添倒
        """
        unpacked_params = [] #B
        e = 0
        layers=[(512,self.env_info_static['shapeOfFeature'][0]),(512,512),(3,512)]      
        for i,l in enumerate(layers): #C
            s,e = e,e+np.prod(l)# 兩者相乘
            weights = params[s:e].view(l) #D
            s,e = e,e+l[0]      
            bias = params[s:e]
            unpacked_params.extend([weights,bias]) #E
        
        
        return unpacked_params


    
    def recombine(self,x1:dict,x2:dict): #A
        x1 = x1['params'] #B
        x2 = x2['params']
        l = x1.shape[0]
        split_pt = np.random.randint(l) #C
        child1 = torch.zeros(l,device=self.device)
        child2 = torch.zeros(l,device=self.device)
        child1[0:split_pt] = x1[0:split_pt] #D
        child1[split_pt:] = x2[split_pt:]
        child2[0:split_pt] = x2[0:split_pt]
        child2[split_pt:] = x1[split_pt:]
        c1 = {'params':child1, 'train_fitness':0.0, 'test_fitness':0.0} #E
        c2 = {'params':child2, 'train_fitness':0.0, 'test_fitness':0.0}
        return c1, c2

    def mutate(self,x:dict, rate:float=0.01) -> dict: #A
        """ 

        Args:
            x (_type_): {'params': tensor([-0.5662,  0.3128,  0.4640,  ..., -0.5078, -0.2047,  0.2499], device='cuda:0'), 'train_fitness':0.0, 'test_fitness':0.0}
            rate (float, optional): 用來替換親代的原始比例. Defaults to 0.01.

        Returns:
            dict: new : {'params': tensor([-0.5662,  0.3128,  0.4640,  ..., -0.5078, -0.2047,  0.2499], device='cuda:0'), 'train_fitness':0.0, 'test_fitness':0.0}
        """
        x_ = x['params']
        num_to_change = int(rate * x_.shape[0]) #B
        idx = np.random.randint(low=0,high=x_.shape[0],size=(num_to_change,))
        x_[idx] = torch.randn(num_to_change,device=self.device) / 10.0 #C 將製作好的隨機數替代原本的權重
        x['params'] = x_
        return x

    def next_generation(self, pop,mut_rate=0.001,tournament_size=0.2):
        # tournament_size ==> 介於0和1之間,用來決定競賽人數
        new_pop = []
        lp = len(pop)
        while len(new_pop) < len(pop): #A 若後代族群商未被填滿,用來決定競賽人數
            rids = np.random.randint(low=0,high=lp,size=(int(tournament_size*lp))) #B 隨機選擇一定比例的族群個體組成子集(將它們的索引存到rids)
            #C 從族群中挑選代理人組成代理人批次,並記錄這些代理人在原始族中的索引值,以及他們的適應度
            batch = np.array([[i,x['train_fitness']] for (i,x) in enumerate(pop) if i in rids]) 
            scores = batch[batch[:, 1].argsort()] #D 將批次中的代理人依照適應度由低至高排序
            i0, i1 = int(scores[-1][0]),int(scores[-2][0]) #E 順序位於最下單的代理人原有最高的適應度
            parent0,parent1 = pop[i0],pop[i1]
            offspring_ = self.recombine(parent0,parent1) #F

            
            child1 = self.mutate(offspring_[0], rate=mut_rate) #G
            child2 = self.mutate(offspring_[1], rate=mut_rate)
            offspring = [child1, child2]
            new_pop.extend(offspring)
        return new_pop
    
if __name__ == '__main__':
    app = EVO_Train()

