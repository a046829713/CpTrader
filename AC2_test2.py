import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp #A
import time
from AC22.AC2_brain import  ActorCritic
from AC22.environment import Env
import matplotlib.pyplot as plt
import os
from multiprocessing import Manager

def run_episode(worker_env, worker_model, N_steps=2000):
    raw_state = np.array(worker_env.reset())
    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    G=torch.Tensor([0]) #A 變數G代表回報值,他的初始值為0
    while (j < N_steps and done == False): #B
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        # 序列的第一步和最后一步强制动作为0
        if j == 1 or j == N_steps:
            action = 0
        else:
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            action = action.cpu().numpy()
        
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)        
        state_, reward, done, info = worker_env.step(action)
        state = torch.from_numpy(state_).float()
        if done:
            worker_env.reset()            
        else: #C
            G = value.detach()
            
        rewards.append(reward)
    return values, logprobs, rewards, G 
     
def update_params(worker_opt,values,logprobs,rewards,G,clc=0.1,gamma=0.95):    
    # 將rewards,logprobs,values的元素順序顛倒,以方便計算折扣回報陣列
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) #A    
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)    
    # 檢查 values 是否為空
    if len(values) > 0:
        values = torch.stack(values).flip(dims=(0,)).view(-1)        
    else:
        # 若 values 為空，則將其設定為一個與 rewards 同形狀的零張量
        values = torch.zeros_like(rewards)
        
    Returns = []
    ret_ = G
    for r in range(rewards.shape[0]): #B
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns,dim=0)
    actor_loss = -1*logprobs * (Returns - values.detach()) #C #最大化 1- probs ,等於0 ,log 解決經度問題
    critic_loss = torch.pow(values - Returns,2) #D # 最小化
    loss = actor_loss.sum() + clc*critic_loss.sum() #E
    
    print("損失函數:",loss)
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


def worker(t, worker_model, counter, shared_rewards_list):
    env = Env(reset_on_close =False,random_ofs_on_reset= True)
    env.reset()
    # 每條程序有獨立的運行環境和優化器,但共享參數模型
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters()) #A
    worker_opt.zero_grad()    
    while True:
        worker_opt.zero_grad()
        values, logprobs, rewards,G = run_episode(env,worker_model) #B
        print(f"這是哪個程序:{t}","獎勵平均:",sum(rewards))
        print('*'*120)
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards,G) #C
        counter.value = counter.value + 1 #D
        shared_rewards_list.append(sum(rewards))  # 将当前回合的奖励总和添加到列表中
        if counter.value % 10 == 0:  # 每100次迭代保存一次训练进度
            save_training_progress(shared_rewards_list, counter.value)    
        env.update_alpha(counter.value)

def save_training_progress(rewards, iteration, save_path='training_progress'):
    """保存训练进度的图像"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure()
    plt.plot(rewards)
    plt.title(f'Training Progress at Iteration {iteration}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'{save_path}/progress_{iteration}.png')
    plt.close()

if __name__ == '__main__':
    with Manager() as manager:
        shared_rewards_list = manager.list()  # 创建一个共享的列表

        MasterNode = ActorCritic()
        MasterNode.share_memory()
        processes = []
        counter = mp.Value('i', 0)
        params = {
        'n_workers':1,
        }
        for i in range(params['n_workers']):
            p = mp.Process(target=worker, args=(i, MasterNode, counter, shared_rewards_list))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for p in processes:
            p.terminate()
    
    
    # print(counter.value,processes[1].exitcode) #H
    
    # env = Env(reset_on_close =False,random_ofs_on_reset= False)    
    # done =False
    # state = torch.from_numpy(env.reset()).float()
    # while not done:
    #     logits,value = MasterNode(state)
    #     # action_dist = torch.distributions.Categorical(logits=logits)
    #     # action = action_dist.sample()
    #     print(logits)
    #     max_index = torch.argmax(logits).item()     
    #     print(max_index)
    #     time.sleep(0.1)
    #     state_, _,  done, info= env.step(max_index)        
    #     state = torch.from_numpy(state_).float()
        