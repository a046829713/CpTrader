import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp #A
import time
from AC2.AC2_brain import ActorCriticAgent
from AC2 import environment
from AC2.ExperienceBuffer import ExperienceBuffer
import os
from datetime import datetime    
from utils.AppSetting import AppSetting
from AC2.utils import save_training_progress,Create_all_model_params
from AC2.ICM_model import ICM_network

NSTEPS = 10000000


def run_episode(worker_env,Agent, N_steps=NSTEPS):    
    raw_state = worker_env.reset()
    state = torch.from_numpy(raw_state).float().to(Agent.device)
    state_for_critic = torch.from_numpy(raw_state).float().view(-1).to(Agent.device)  # 同样扁平化为 [250]
    
    values, logprobs, rewards, state1s, actions, state2s = [],[],[],[],[],[]
    done = False
    j=0
    G = torch.Tensor([0]).to(Agent.device) #A 變數G代表回報值,他的初始值為0
    while (j < N_steps and done == False): #B
        j+=1
        policy = Agent.policy(state.unsqueeze(0))# torch.Size([50, 5]) -> [1, 50, 5]
        state1s.append(state)    
        value = Agent.critic(state_for_critic)
        values.append(value)
        
        # 序列的第一步和最后一步强制动作为0
        if j == 1 or j == N_steps:
            action = torch.tensor(0).to(Agent.device)
        else:
            logits = policy.view(-1)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()            
        
        logprob_ = policy.view(-1)[action] # 取得實際的logprob,透過mask的方式
        logprobs.append(logprob_)
        state_, reward, done, info = worker_env.step(action.item())
        actions.append(action)
        
        # 檢查是否為序列的最後一步
        if j == N_steps or done:
            state = torch.zeros_like(state)  # 設置 state2 為全零狀態
        else:
            state = torch.from_numpy(state_).float().to(Agent.device)

        state2s.append(state)        
        state_for_critic = torch.from_numpy(state_).float().view(-1).to(Agent.device)   # 同样扁平化为 [250]
    
        if done:            
            worker_env.reset()            
        else: 
            G = value.detach()
            
        rewards.append(reward)
    return values, logprobs, state1s, actions, rewards, state2s, G 
     
def update_params(Agent,values,logprobs,rewards,G):    
    # 將rewards,logprobs,values的元素順序顛倒,以方便計算折扣回報陣列
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1) #A    
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)    
    # 檢查 values 是否為空
    if len(values) > 0:
        values = torch.stack(values).flip(dims=(0,)).view(-1)        
    else:
        # 若 values 為空，則將其設定為一個與 rewards 同形狀的零張量
        values = torch.zeros_like(rewards).to(Agent.device)
    
    Returns = []
    ret_ = G
    for r in range(rewards.shape[0]): #B
        ret_ = rewards[r] + Agent.gamma * ret_
        Returns.append(ret_)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns,dim=0)
    
    actor_loss = -1*logprobs * (Returns - values.detach()) #C #最大化 1- probs ,等於0 ,log 解決經度問題
    critic_loss = torch.pow(values - Returns,2) #D # 最小化
    return actor_loss, critic_loss

def ICM(icm_module, state1, action, state2, forward_scale=1, inverse_scale=1):
    encoder = icm_module.encoder
    forward_model = icm_module.forward_model
    inverse_model = icm_module.inverse_model
    
    state1_hat = encoder(state1) #A
    state2_hat = encoder(state2)
    
       
    state2_hat_pred = forward_model(state1_hat.detach(), action) #B
    
    forward_pred_err = forward_scale * icm_module.forward_loss(state2_hat_pred, state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
    pred_action = inverse_model(state1_hat, state2_hat)
    inverse_pred_err = inverse_scale * icm_module.inverse_loss(pred_action, action.flatten()).unsqueeze(dim=1)
    return forward_pred_err, inverse_pred_err

def ICM_train(icm_module,state1_batch, action_batch, reward_batch, state2_batch):
    state1 = torch.stack(state1_batch).to(icm_module.device)
    state2 = torch.stack(state2_batch).to(icm_module.device)
    action = torch.stack(action_batch).to(icm_module.device)

    forward_pred_err, inverse_pred_err = ICM(icm_module, state1, action, state2) #B
    return forward_pred_err, inverse_pred_err

def loss_fn(actor_loss,critic_loss,forward_pred_err,inverse_pred_err, alpha=1.0, beta=0.1, lambda_=0.8, gamma=0.1):
    # 有沒有flatten,在mean答案都相同    
    # 对actor和critic损失加权
    weighted_actor_loss = alpha * actor_loss.sum()
    weighted_critic_loss = beta * critic_loss.sum()
    # 对ICM的inverse和forward损失加权
    weighted_forward_loss = gamma * forward_pred_err.sum()
    weighted_inverse_loss = lambda_ * inverse_pred_err.sum()
    
    # 计算总损失
    total_loss = weighted_actor_loss + weighted_critic_loss + weighted_inverse_loss + weighted_forward_loss  
    
    return total_loss

def test_loss_fn(actor_loss,critic_loss):
    return actor_loss.sum() + 0.1*critic_loss.sum()

def worker(Agent,icm_module,buffer):    
    # 保存檢查點的函數
    def save_checkpoint(state, filename):
        torch.save(state, filename)
        
    CHECKPOINT_EVERY_STEP = 100000
    
    setting = AppSetting.get_ooas_setting()
    
    saves_path = os.path.join(setting['SAVES_PATH'], datetime.strftime(
            datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(setting['BARS_COUNT']) + 'k-' + str(setting['REWARD_ON_CLOSE']) +'-'+ str(setting['RESET_ON_CLOSE']))
    os.makedirs(saves_path, exist_ok=True)    
    worker_env = environment.Env(reset_on_close=False, random_ofs_on_reset= False)   
    
    # 每條程序有獨立的運行環境和優化器,但共享參數模型
    opt = Create_all_model_params(Agent,icm_module)
    opt.zero_grad()
    step_idx = 0
    while True:
        opt.zero_grad()
        values, logprobs, state1s, actions, rewards, state2s, G = run_episode(worker_env,Agent) #B
        actor_loss,critic_loss = update_params(Agent,values,logprobs,rewards,G) #C       
        # forward_pred_err, inverse_pred_err = ICM_train(icm_module,state1s, actions, rewards, state2s)
        # actor_loss torch.Size([2000])
        # critic_loss torch.Size([2000])
        # forward_pred_err torch.Size([2000, 1])
        # inverse_pred_err torch.Size([2000, 1])
        # print("演員損失:",actor_loss,"評論家損失:",critic_loss,"狀態預測損失:",forward_pred_err,"動作預測損失:",inverse_pred_err)                  
        # loss = loss_fn(actor_loss,critic_loss,forward_pred_err,inverse_pred_err)        
        loss = test_loss_fn(actor_loss,critic_loss)        
        loss.backward()
        
        
        # for name, param in Agent.policy.named_parameters():
        #     if 'weight' in name:
        #         print(param)
        
        #     if param.grad is not None:
        #         print(f"{name} gradient norm: {param.grad.norm()}")
        #     else:
        #         break        
        opt.step()

        # 在主訓練循環中的合適位置插入保存檢查點的代碼
        # if step_idx % CHECKPOINT_EVERY_STEP == 0:
        #     idx = step_idx // CHECKPOINT_EVERY_STEP
        #     checkpoint = {
        #         'step_idx': step_idx,
        #         'model_state_dict': Agent.policy.state_dict(),
        #         'optimizer_state_dict': opt.state_dict(),
        #     }
        #     save_checkpoint(checkpoint, os.path.join(saves_path, f"checkpoint-{idx}.pt"))

        step_idx +=1
        print(f"目前次數:{step_idx}","總獎勵:",sum(rewards),"全部加起來的損失函數:",loss)  # 将当前回合的奖励总和添加到列表中
        
        # if step_idx % 10 == 0:  # 每100次迭代保存一次训练进度
            # save_training_progress(shared_rewards_list, step_idx)
        
        if step_idx % 10 == 0 or step_idx == 1:
            #  validation    
            test_env = environment.Env(reset_on_close=False, random_ofs_on_reset= False)
            raw_state = test_env.reset()
            state = torch.from_numpy(raw_state).float().unsqueeze(0).to(Agent.device)# [1, 50, 5]
            done = False
            model = Agent.policy
            model.eval()
            eval_count = 0
            with torch.no_grad(): 
                first = True
                actinos_space =[]
                while not done:                                   
                    policy = model(state)                
                    action = torch.argmax(policy).item()
                    if first:
                        action = 0
                        first = False
                                
                    actinos_space.append(action)
                    state_, reward, done, info = test_env.step(action)
                    state = torch.from_numpy(state_).float().unsqueeze(0).to(Agent.device)
                    eval_count +=1
                    if eval_count >2000:
                        break
                    
            print(actinos_space)
            print(sum(actinos_space))
            model.train()


    
if __name__ == '__main__':
    # 初始化環境
    worker_env = environment.Env(reset_on_close=False, random_ofs_on_reset= True)
    raw_state = worker_env.reset()
    Agent = ActorCriticAgent(0.01, raw_state) #A
    icm_module = ICM_network(raw_state)
    
    
    
    
    # encoder = Phi()
    # forward_model = Fnet()
    # inverse_model = Gnet()
    # forward_loss = nn.MSELoss(reduction='none')
    # inverse_loss = nn.CrossEntropyLoss(reduction='none')
    # qloss = nn.MSELoss()


    buffer = ExperienceBuffer(max_size=100000)  
    worker(Agent,icm_module,buffer)