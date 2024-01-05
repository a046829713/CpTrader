import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import time
from utils.AppSetting import AppSetting


class Actions(enum.Enum):
    No_hold = 0
    Buy = 1
    SellShort = 2


class State:
    def __init__(self, bars_count, reset_on_close):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(reset_on_close, bool)
        self.bars_count = bars_count        
        self.reset_on_close = reset_on_close
        self.alpha = 0
        self.setting = AppSetting.get_LSTM_DQN_setting()
        self.step_index = 0 # 用來測試呼叫了step多少次
        
    def reset(self, prices, offset):
        assert offset >= self.bars_count-1        
        self._prices = prices
        self._offset = offset        

        self.count_env_step = 0
        self.current_action = None
        self.update_scale()
        
    @property
    def shape(self):
        return (self.bars_count,self.get_dataFeature_shape(model='RNN'))
    
    def get_dataFeature_shape(self,model:str) ->int:        
        if model == 'RNN':
            # [high,low,close,volume,position]
            return 5
    
    def encode(self):
        """
            [batch_size,seqlength,input_size] (RNN的輸入模型為此)
            
            在這個function 裡面我認為,
                只需要準備 [seqlength,input_size]
        """        
        res = np.ndarray(shape= [self.bars_count,self.get_dataFeature_shape(model='RNN')], dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count, 0):            
            res[shift,0] = self._prices.high[self._offset + bar_idx]            
            res[shift,1] = self._prices.low[self._offset + bar_idx]            
            res[shift,2] = self._prices.close[self._offset + bar_idx]            
            res[shift,3] = self._prices.volume[self._offset + bar_idx]
            res[shift,4] = 0.0 if self.current_action is None else self.current_action
            shift += 1
        
        # 確定論文裡面是有將部位狀態放入的        
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar

        # 為甚麼會這樣寫的原因是因為 透過rel_close 紀錄的和open price 的差距(百分比)來取得真實的收盤價
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)
    
    def update_scale(self):
        """
            為了要將收盤價格正則化
            所以要知道最高最低價
        """
        
        close_array = self._prices.open* (1+self._prices.close)
        self.maxprice = max(close_array)
        self.minprice = min(close_array)
    
    def normalize_price(self, price):
        """
        对给定的价格进行规范化。
        
        参数:
        price (float): 要规范化的价格。

        返回:
        float: 规范化后的价格。
        """
        return (price - self.minprice) / (self.maxprice - self.minprice)
    
    def calculate_reward(self,sa_current, sa_next, price, alpha):
        """
            1.定义奖励函数:
                当代理决定买入时，代理花费的钱将增加，因此奖励为负值。
                当代理决定卖出时，代理获得的钱将增加，因此奖励为正值。                
                
            2.交易成本:
                交易成本通过交易金额和成本比率的乘积计算。
                
        
        计算奖励函数。

        参数:
        sa_current: 当前时间步的代理位置状态。
        sa_next: 下一时间步的代理位置状态。
        p_buy: 当前时间步的买入价格。
        p_sell: 当前时间步的卖出价格。
        alpha_buy: 買入時的交易成本比率。 (手續費,交易稅,滑價)
        alpha_sell: 卖出时的交易成本比率。

        返回:
        奖励值。
        """
        if sa_next > sa_current:
            reward = -abs(sa_next - sa_current) * (1 + alpha) * price
        else:
            reward = abs(sa_next - sa_current) * (1 - alpha) * price

        return reward
    
    def step(self, action):
        """
            重新建構獎勵:
                採用新的公式嘗試看看
                                
            sa_current: 当前时间步的代理位置状态。
            sa_next: 下一时间步的代理位置状态。
            p_buy: 当前时间步的买入价格。
            p_sell: 当前时间步的卖出价格。
            alpha_buy: 買入時的交易成本比率。 (手續費,交易稅,滑價)
            alpha_sell: 卖出时的交易成本比率。
        """
        assert isinstance(action, Actions)
        
        reward = 0.0
        done = False
        close = self._cur_close()        
        
        
        sa_current = 0.0 if self.current_action is None else self.current_action
        
        if action == Actions.No_hold:       
            sa_next = 0.0 #  if 𝑎𝑖 = 1,then 𝑠𝑎𝑖+1 = 1 at the next time step.
        elif action == Actions.Buy:
            sa_next = 1.0
        elif action == Actions.SellShort:
            sa_next = -1.0
        
        self.count_env_step +=1                        
        self._offset += 1
        
        if self.reset_on_close and self.count_env_step >2000:
            done = True
                
        prev_close = close # 上一根的收盤價
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1        
        self.current_action = sa_next
        
        reward = self.calculate_reward(sa_current=sa_current,
                              sa_next=sa_next,
                              price = self.normalize_price(close),
                              alpha=self.alpha,
        )
        
        
        self.update_alpha()
        return reward, done
    
    
    def update_alpha(self):
        # 每次调用时逐步增加alpha值，但不超过最大值
        self.step_index +=1        
        self.alpha = min(self.step_index / 1000000, self.setting['MODEL_DEFAULT_COMMISSION_PERC'] +self.setting['DEFAULT_SLIPPAGE'] )
    



class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count,reset_on_close=True,
                 random_ofs_on_reset=True):

        
        assert isinstance(prices, dict)
        # 用來記憶全部的價格序列
        self._prices = prices
        self._state = State(bars_count, reset_on_close)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars

        self._state.reset(prices, offset)
        return self._state.encode()
    
    def step(self, action_idx):
        """
            呼叫子類_state 來獲得獎勵
        Args:
            action_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        action = Actions(action_idx)
        reward, done = self._state.step(action) # 這邊會更新步數
        obs = self._state.encode() # 呼叫這裡的時候就會取得新的狀態
        info = {"instrument": self._instrument, "offset": self._state._offset}        
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
