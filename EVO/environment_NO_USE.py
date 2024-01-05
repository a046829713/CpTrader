import numpy as np
import enum
from utils.AppSetting import AppSetting
from .DataFeature import DataFeature
import time

class Actions(enum.Enum):
    No_hold = 0 
    Buy = 1
    Sellshort = 2
    
class State:
    def __init__(self, bars_count:int, reset_on_close:bool):
        """
            bars_count(int):
            用來組成神經網絡所需要的環境

        """
        self.bars_count = bars_count # 所需要使用的K棒數量
        self.setting = AppSetting.get_evo_setting()
        self.reset_on_close = reset_on_close
        self.alpha = self.setting['MODEL_DEFAULT_COMMISSION_PERC'] +self.setting['DEFAULT_SLIPPAGE']
        
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
        self.game_count = 0
        
    
    def encode(self):
        """
        Convert current state into numpy array.

        用來製作state 一維狀態的函數

        return res:
            [ 0.01220753 -0.00508647 -0.00508647  0.00204918 -0.0204918  -0.0204918
            0.01781971 -0.00419287 -0.00419287  0.         -0.0168421  -0.00736842
            0.01359833 -0.0041841   0.00732218  0.00314795 -0.00629591 -0.00314795
            0.00634249 -0.00422833 -0.00317125  0.01800847  0.          0.01800847
            0.01155462 -0.00315126  0.00945378  0.0096463  -0.00214362  0.0096463
            0.          0.        ]

            # 倒數第二個0 為部位
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)

        shift = 0

        # 我認為這邊有一些問題,為甚麼要從1開始,而不從0開始呢?
        # 1-10
        for bar_idx in range(-self.bars_count+1, 1):

            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.volume[self._offset + bar_idx]
            shift += 1

        res[shift] = 0.0 if self.current_action is None else self.current_action        
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
        self.game_count +=1
        
        reward = 0.0
        done = False
        close = self._cur_close()
        sa_current = 0.0 if self.current_action is None else self.current_action
        
        if action == Actions.No_hold:       
            sa_next = 0.0 #  if 𝑎𝑖 = 1,then 𝑠𝑎𝑖+1 = 1 at the next time step.
        elif action == Actions.Buy:
            sa_next = 1.0
        elif action == Actions.Sellshort:
            sa_next = -1.0

        self._offset += 1        
        prev_close = close # 上一根的收盤價
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1        
        self.current_action = sa_next
        reward = self.calculate_reward(sa_current=sa_current,
                              sa_next=sa_next,
                              price = close,
                              alpha=self.alpha,
        )
        
        # 暫時棄用        
        # if self.reset_on_close:
        #     if self.game_count > self.setting['GAME_MAX_COUNT']:
        #         done = True
        
        # print("當前動作:",action,"目前部位:",sa_current,"下一個動作:",sa_next,"正則化價格",self.normalize_price(close),"目前獎勵:",reward)
        return reward, done
    
    
class Env():
    def __init__(self,data_type:str,reset_on_close = False, random_ofs_on_reset=True) -> None:
        """
            用來建構完整環境
            
        """       
        self._state = State(bars_count=50,reset_on_close=reset_on_close)
        self._prices = DataFeature(data_type=data_type).get_train_net_work_data()
        self.env_data_type = data_type
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
    
    
    def step(self, action_idx):
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
    
    def info_provider(self) -> dict:
        """
            用來記錄環境的資訊,並且和一般的info,不相同

        Returns:
            dict: 
        """
        info_ = {
            "shapeOfFeature":self._state.shape
        }        
        return info_