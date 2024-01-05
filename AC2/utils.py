import matplotlib.pyplot as plt
import os
from torch import optim
import time

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
    
    
    
def Create_all_model_params(Agent,icm_module):
    all_model_params = list(Agent.policy.parameters()) + list(Agent.critic.parameters())
    all_model_params += list(icm_module.encoder.parameters())+ list(icm_module.forward_model.parameters()) + list(icm_module.inverse_model.parameters())
    return optim.Adam(lr=Agent.lr, params=all_model_params)
    