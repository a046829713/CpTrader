from OOAS.ooas_brain import PolicyGradientAgent 
from OOAS.environment import Env 
import time
import torch
from torch import optim
import numpy as np
from OOAS.utils import Datatransformer
import torch


class OOAS_Train():
    def __init__(self) -> None:
        self.train_env = Env(reset_on_close = False,random_ofs_on_reset= False)
        self.A_SPACE = {1,0} 
        self.Agent = PolicyGradientAgent(lr=0.0001,num_classes=len(self.A_SPACE))        
        self.main()
        
    def main(self):        
        # 每條程序有獨立的運行環境和優化器,但共享參數模型
        self.worker_opt = optim.Adam(lr=self.Agent.lr,params=self.Agent.policy.parameters()) #A
        self.worker_opt.zero_grad()        
        step_idx =0    
        # 訓練階段
        while True:            
            states, next_states, infos = self.Calculate_probs()
            loss = self.update_params(states, next_states, infos)
            print(loss)
            self.worker_opt.zero_grad()
            loss.backward()
            self.worker_opt.step()            
            
            if step_idx % 10 == 0:
                self.val_test()
            
            step_idx +=1
            
    def Calculate_probs(self, N_steps=2000):
        """
            用來取得基本資料的
        """
        raw_state = self.train_env.reset()
        state = torch.from_numpy(raw_state).to(self.Agent.device)
        states, next_states, infos= [],[],[]
        done = False        
        j=0
        while (j < N_steps and done == False): #B
            j+=1                     
            states.append(state)
            state_, done, info = self.train_env.step()            
            infos.append(info)
            # 檢查是否為序列的最後一步
            if j == N_steps or done:
                state = torch.zeros_like(state)  # 設置 state2 為全零狀態
            else:
                state = torch.from_numpy(state_).float().to(self.Agent.device)
            
            next_states.append(state)            
        return states, next_states, infos
    
    def update_params(self, states, next_states, infos,gamma=1):
        """
            論文中的gamma 為1
        """
        Sa ={1,0}        
        # 状态转移概率
        def P(sa_i, sa_next, se_i, a_i):
            return 1 if a_i == sa_next else 0
        
        def D_value(sa_i, a_i, Q):            
            return sum(P(sa_i, sa_next, None, a_i) * Q[sa_i][sa_next] for sa_next in Sa)

        
        Q = np.zeros((len(Sa), len(Sa)))
        next_state_value = np.zeros(len(Sa))
        
        total_grad = 0
        nb = 20
        ne = 20
        for i in reversed(range(len(states))):
            for each_action_compare in Datatransformer.count_cartesian_product(Sa,self.A_SPACE):                                
                sa1,sa2 = each_action_compare                
                reward = Datatransformer.calculate_reward(sa_current=sa1,sa_next=sa2,price=infos[i]['price'],alpha=infos[i]['alpha'])
                if i==0:
                    print("SA1:",sa1,"SA2:",sa2,"獎勵:",reward,"下一個動作的狀態值:",next_state_value[sa2],"alpha 目前為多少:",str(infos[i]['alpha']))
                Q[sa1][sa2] = reward + gamma * next_state_value[sa2]
            
            
            # 计算 D 值和 E 值
            for sa_i in Sa:
                D = np.zeros((len(self.A_SPACE)))
                for ai in self.A_SPACE:
                    if i==0:
                        print("當前狀態:",sa_i,"下一步的狀態:",ai ,"行動價值為:",D_value(sa_i, ai, Q))
                    D[ai] = D_value(sa_i, ai, Q)
                
                # 创建一个新的列向量，大小与原始张量的行数相同
                new_column = torch.tensor(states[i].size(0) * [sa_i], device=self.Agent.device).unsqueeze(1)

                # 沿着列方向拼接原始张量和新列向量
                states_with_action = torch.cat((states[i], new_column), dim=1)
                
                probs= self.Agent.policy(states_with_action.unsqueeze(0))
                # 将 D 转换为张量并确保它在同一设备上
                D_tensor = torch.tensor(D, device=probs.device)
                
                b = D_tensor.mean()
                
                # 计算 G(se_i, sa_i | π, θ)
                next_state_value[sa_i] = torch.sum(probs * D_tensor)
                grad_Li = len(self.A_SPACE) * torch.sum((D_tensor - b) * probs)
                
                # 累积梯度
                if i > (nb-1) and  i < (len(states) -20):
                    total_grad += grad_Li
        
            
        avg_grad = total_grad / ((len(states) - nb - ne) * len(Sa))
        return avg_grad   
        
    
    def val_test(self):
        #  validation    
        test_env = Env(reset_on_close = False,random_ofs_on_reset= False)
        raw_state = test_env.reset()
        state = torch.from_numpy(raw_state).to(self.Agent.device)        
        # 起始為0
        new_column = torch.tensor(state.size(0) * [0], device=self.Agent.device).unsqueeze(1)

        # 沿着列方向拼接原始张量和新列向量
        state = torch.cat((state, new_column), dim=1)

        done = False
        model = self.Agent.policy
        model.eval()
        eval_count = 0
        with torch.no_grad():             
            actino_records =[]
            first =True
            while not done:                                   
                policy = model(state.unsqueeze(0))
                print("策略:",policy)             
                action = torch.argmax(policy).item()
                
                if first:
                    action = 0
                    first = False
                                    
                actino_records.append(action)
                
                state_, done, info = test_env.step()
                
                state = torch.from_numpy(state_).to(self.Agent.device)                
                new_column = torch.tensor(state.size(0) * [action], device=self.Agent.device).unsqueeze(1)
                # 沿着列方向拼接原始张量和新列向量
                state = torch.cat((state, new_column), dim=1)          
                
                eval_count +=1
                if eval_count >2000:
                    break
                
        print(actino_records)
        print(sum(actino_records))
        model.train()
    
        

if __name__ =='__main__':    

   OOAS_Train()
        