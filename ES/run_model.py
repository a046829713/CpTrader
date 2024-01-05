#!/usr/bin/env python3
import numpy as np
import torch
import re
from utils.AppSetting import AppSetting
from .common import Strategy_base_DQN
import time
import os
from .DataFeature import DataFeature
from .environment import Env
from .models import CustomNet
from . import Backtest



class Record_Orders():
    def __init__(self, strategy: Strategy_base_DQN, formal: bool = False) -> None:
        self.strategy = strategy
        self.model_count_path = strategy.model_count_path
        self.setting = AppSetting.get_evo_setting()
        self.formal = formal
        self.BARS = re.search(
            '\d+', self.model_count_path.split('\\')[1].split('-')[2])
        self.BARS = int(self.BARS.group())
        self.EPSILON = 0.00
        self.main_count()

    def main_count(self):
        app = DataFeature(self.formal)

        # 實際上在使用的時候 他並沒有reset_on_close
        env = Env(data_type='all_data', random_ofs_on_reset=False)

        net = CustomNet(input_size=202,hidden_size1=512,hidden_size2=512,output_size=3)
        
        if self.model_count_path and os.path.isfile(self.model_count_path) and '.pt' in self.model_count_path :
            print("pt,model指定運算模式")
            checkpoint = torch.load(self.model_count_path, map_location=lambda storage, loc: storage)
            # print(checkpoint)
            # time.sleep(100)
            net.load_state_dict(checkpoint['model_state_dict'])
        else:            
            net.load_state_dict(torch.load(
                self.model_count_path, map_location=lambda storage, loc: storage))

        # 開啟評估模式
        net.eval()
        
        obs = env.reset()  # 從1開始,並不是從0開始
        start_price = env._state._cur_close()
        step_idx = 0
        self.record_orders = []
        total_reward = 0.0
        
        while True:
            step_idx += 1
            obs_v = torch.tensor(np.array([obs]))
            out_v = net(obs_v)            
            action_idx = out_v.max(dim=1)[1].item()
            self.record_orders.append(self.parser_order(action_idx))
            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward            
            if step_idx % 100 == 0:
                print("%d: reward=%.3f" % (step_idx, total_reward))
                # print("動作為:",action_idx,"獎勵為:",reward,"總獎勵:",total_reward)
            if done:
                break
        
        
        print(self.record_orders)
        print('已經轉換成訂單')
        

    def get_marketpostion(self):
        self.shiftorder = np.array(self.record_orders)
        self.shiftorder = np.roll(self.shiftorder, 1)
        self.shiftorder[0] = 0  # 一率將其歸零即可

        marketpostion = 0  # 部位方向
        for i in range(len(self.shiftorder)):
            current_order = self.shiftorder[i]  # 實際送出訂單位置(訊號產生)
            # 部位方向區段
            if current_order == 1:
                marketpostion = 1
            if current_order == -1:
                marketpostion = 0

        return marketpostion

    def getpf(self):
        
        return Backtest.Backtest(
            self.strategy.df, self.BARS, self.strategy).order_becktest(self.record_orders)
        
    def parser_order(self,action_idx):
        """
        
        """
        if action_idx == 2:
            return -1
        
        return action_idx

        
        