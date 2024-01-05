import numpy as np
import enum
from DQN.lib.DataFeature import DataFeature
from utils.AppSetting import AppSetting
import time
from .utils import Datatransformer
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
        self.setting = AppSetting.get_ooas_setting()
        self.reset_on_close = reset_on_close
        self.alpha = 0.0 
        self.all_env_count = 0 # 計算這個程序計算了幾次
    
    def reset(self,prices,offset):
        """
            透過reset 來指定需要的標的物
        """
        self._prices = prices
        self._offset = offset
        self.game_count = 0
        self.update_scale()
        
    def get_dataFeature_shape(self,model:str) ->int:        
        if model == 'RNN':
            # [high,low,close,volume,position]
            return 4
        
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
            shift += 1
                
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
    

    def update_alpha(self,step_idx):
        # 每次调用时逐步增加alpha值，但不超过最大值
        # self.alpha = min(step_idx / 1000000000, self.setting['MODEL_DEFAULT_COMMISSION_PERC'] +self.setting['DEFAULT_SLIPPAGE'] )
        self.alpha = 0
    
    def step(self, action):
        """
            取得各項資料
        """
        assert isinstance(action, Actions)
        self.game_count +=1
        self.all_env_count +=1
        
        done = False
        self._offset += 1        

        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1        

        _record_alpha = self.alpha
        _record_price = self.normalize_price(close)        
        # _record_price = close        
             
        # print("當前動作:",action,"目前部位:",sa_current,"下一個動作:",sa_next,"正則化價格",self.normalize_price(close),"目前獎勵:",reward)
        self.update_alpha(self.all_env_count)
        return done, _record_alpha, _record_price
    
       
        
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

        self._state.reset(prices, offset)
        return self._state.encode()
    
    
    def get_leanring_shape(self):
        """
            用來給Critic 使用
        """  
        return self.reset().flatten().shape[0]
    
    def step(self):
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
        action = Actions(0)
        
        done, _record_alpha, _record_price = self._state.step(action) # 這邊會更新步數
        obs = self._state.encode() # 呼叫這裡的時候就會取得新的狀態
        
        info = {
                "instrument": self._instrument,
                "offset": self._state._offset,
                'price':_record_price,
                'alpha':_record_alpha
                }
        
        return obs, done, info
    
