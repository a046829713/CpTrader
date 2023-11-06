#!/usr/bin/env python3
import numpy as np
from DQN.lib.DataFeature import DataFeature
import torch
from DQN.lib import environ, models, Backtest
import re
from utils.AppSetting import AppSetting
from .common import Strategy_base_DQN
import time
# 尚未驗算實際下單部位


class Record_Orders():
    def __init__(self, strategy: Strategy_base_DQN, formal: bool = False) -> None:
        self.strategy = strategy
        self.model_count_path = strategy.model_count_path
        self.setting = AppSetting.get_DQN_setting()
        self.formal = formal

        self.BARS = re.search(
            '\d+', self.model_count_path.split('\\')[1].split('-')[2])
        self.BARS = int(self.BARS.group())
        self.EPSILON = 0.00

        self.main_count()

    def main_count(self):
        app = DataFeature(self.formal)
        prices = app.get_test_net_work_data(
            symbol=self.strategy.symbol_name, symbol_data=self.strategy.df)  # len(prices.open) 2562

        # 實際上在使用的時候 他並沒有reset_on_close
        env = environ.StocksEnv(prices, bars_count=self.BARS, reset_on_close=False, commission=self.setting['MODEL_DEFAULT_COMMISSION_PERC'],
                                state_1d=self.setting['STATE_1D'], random_ofs_on_reset=False, reward_on_close=self.setting['REWARD_ON_CLOSE'],  volumes=self.setting['VOLUMES_TURNON'])

        if self.setting['STATE_1D']:
            net = models.DQNConv1D(
                env.observation_space.shape, env.action_space.n)
        else:
            net = models.SimpleFFDQN(
                env.observation_space.shape[0], env.action_space.n)

        net.load_state_dict(torch.load(
            self.model_count_path, map_location=lambda storage, loc: storage))

        # 对模型进行脚本化，并用示例输入
        # example_input = torch.tensor(np.array([env.reset()]))  # 使用环境重置作为示例输入
        # scripted_net = net.script_model(example_input)

        obs = env.reset()  # 從1開始,並不是從0開始
        start_price = env._state._cur_close()
        step_idx = 0
        self.record_orders = []
        while True:
            step_idx += 1
            obs_v = torch.tensor(np.array([obs]))
            out_v = net(obs_v)
            action_idx = out_v.max(dim=1)[1].item()
            self.record_orders.append(self._parser_order(action_idx))
            obs, reward, done, _ = env.step(action_idx)
            if done:
                break

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


    def _parser_order(self, action_value: int):
        if action_value == 2:
            return -1
        return action_value
