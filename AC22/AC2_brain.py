import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import time

class ActorCritic(nn.Module): #B
    def __init__(self,input_dims = 202):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(input_dims,512)
        self.l2 = nn.Linear(512,512)
        self.actor_lin1 = nn.Linear(512,3)
        self.l3 = nn.Linear(512,25)
        self.critic_lin1 = nn.Linear(25,1)
    
    def forward(self,x):
        x = F.normalize(x,dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y),dim=0) #C
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c)) #D
        return actor, critic #E

    