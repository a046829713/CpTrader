import numpy as np
import enum
from DQN.lib.DataFeature import DataFeature
from utils.AppSetting import AppSetting
import time

class Actions(enum.Enum):
    No_hold = 0 
    Buy = 1

    
class State:
    def __init__(self, bars_count:int, reset_on_close:bool):
        """
            bars_count(int):
            用來組成神經網絡所需要的環境

        """
        self.bars_count = bars_count # 所需要使用的K棒數量
        self.setting = AppSetting.get_ooas_setting()
        self.reset_on_close = reset_on_close
        self.alpha =0.0
        
    @property
    def shape(self):
        """        
            根據你的特徵來決定shape
        """
        return (4 * self.bars_count + 1, )

    
    def reset(self,prices,offset):
        """
            透過reset 來指定需要的標的物
        """
        self._prices = prices
        self._offset = offset
        self.current_action = None
        self.update_scale()
    
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
            reward = 2 * abs(sa_next - sa_current) * (1 - alpha) * price

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

        self._offset += 1        
        prev_close = close # 上一根的收盤價
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1        
        self.current_action = sa_next
        reward = self.calculate_reward(sa_current=sa_current,
                              sa_next=sa_next,
                              price = self.normalize_price(close),
                              alpha=self.alpha,
        )
        
        # print("當前動作:",action,"目前部位:",sa_current,"下一個動作:",sa_next,"正則化價格",self.normalize_price(close),"目前獎勵:",reward)
        return reward, done
    
    
    def update_alpha(self,step_idx):
        # 每次调用时逐步增加alpha值，但不超过最大值
        self.alpha = min(step_idx / 1000000, self.setting['MODEL_DEFAULT_COMMISSION_PERC'] +self.setting['DEFAULT_SLIPPAGE'] )
    
class Simulate_env():
    def __init__(self,all_prices:dict) -> None:
        """
        用來模擬
            replay中的reward計算        
        """
        self.all_prices  = all_prices
        self.setting = AppSetting.get_ooas_setting()
         
    def asign(self,state,info:dict):
        """
        Returns:
            info: {'instrument': 'BTCUSDT', 'offset': 68638}
        
        """
        assert isinstance(info,dict) == True,"info must be dict"
        
        self.instrument = info['instrument']
        self._offset = info['offset']

        # 更新至指定商品
        self._prices = self.all_prices[self.instrument]

        
        if state[-1] == 0:
            self.have_position = False
        elif state[-1] == 1:
            self.have_position = True
        
    def _cur_close(self):
        """
        Calculate real close price for the current bar

        # 為甚麼會這樣寫的原因是因為 透過rel_close 紀錄的和open price 的差距(百分比)來取得真實的收盤價
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)    
    
    def step(self, action):
        """
            重新建構獎勵:
                採用新的公式嘗試看看
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        
        sa_current = float(self.have_position)
        
        if action == Actions.Buy and not self.have_position:
            self.have_position = True           
            self.open_price = close * (1 + self.setting['DEFAULT_SLIPPAGE'])
            
        elif action == Actions.Close and self.have_position:
            self.have_position = False
            self.open_price = 0.0
                
        self._offset += 1        
        prev_close = close # 上一根的收盤價
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1        
    
        reward = self.calculate_reward(sa_current=sa_current,
                              sa_next=float(self.have_position),
                              p_buy=prev_close,
                              p_sell=close,
                              alpha_buy=self.setting['MODEL_DEFAULT_COMMISSION_PERC'],
                              alpha_sell=self.setting['MODEL_DEFAULT_COMMISSION_PERC'] +self.setting['DEFAULT_SLIPPAGE'] )
        
        
        return reward, done        
    
    def calculate_reward(self,sa_current, sa_next, p_buy, p_sell, alpha_buy, alpha_sell):
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

        # 计算位置状态的变化
        position_change = sa_next - sa_current

            # 如果下一状态大于当前状态（代表买入）
        if position_change > 0:
            reward = -abs(position_change) * (1 + alpha_buy) * p_buy
            # 否则（代表卖出或保持不变）
        else:
            reward = abs(position_change) * (1 - alpha_sell) * p_sell

        return reward        
        
class Env():
    def __init__(self,reset_on_close = False, random_ofs_on_reset=True) -> None:
        """
            用來建構完整環境
            
        """        
        self._state = State(bars_count=50,reset_on_close=reset_on_close)
        self._prices = DataFeature().get_train_net_work_data()
        self.random_ofs_on_reset = random_ofs_on_reset
        
    def reset(self):
        """
            商品隨機性
            起步隨機性
        """        
        self._instrument = np.random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        bars = self._state.bars_count
        
        if self.random_ofs_on_reset:
            offset = np.random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars

        print("此次步數為:",offset)
        self._state.reset(prices, offset)
        return self._state.encode()
    
    
    def step(self,action_idx):
        """
            retunr : 
                oberservation_ (下一個觀察狀態),
                reward(獎勵),
                done(是否完成),
                info(其他資料),
        """
        
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
    
    
    def update_alpha(self,step_idx):
        self._state.update_alpha(step_idx)