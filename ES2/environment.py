import numpy as np
import enum
from utils.AppSetting import AppSetting
from .DataFeature import DataFeature
import time

class Actions(enum.Enum):
    skip = 0 
    Buy = 1
    Close = 2
    
class State:
    def __init__(self, bars_count:int):
        """
            bars_count(int):
            用來組成神經網絡所需要的環境

        """
        self.bars_count = bars_count # 所需要使用的K棒數量
        self.setting = AppSetting.get_es_setting()                
        self.initial_investment = self.setting['INITIAL_INVESTMENT']
    
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
        self.have_position = False
        self.get_now_share = 0.0        
        
        
        
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

        res[shift] = self.get_now_share
        return res
    
    def _cur_close(self):
        """
        Calculate real close price for the current bar

        # 為甚麼會這樣寫的原因是因為 透過rel_close 紀錄的和open price 的差距(百分比)來取得真實的收盤價
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)    
    
    def update_alpha(self,step_idx):
        # 每次调用时逐步增加alpha值，但不超过最大值
        # self.alpha = min((step_idx / 10000000) *( self.setting['MODEL_DEFAULT_COMMISSION_PERC'] + self.setting['DEFAULT_SLIPPAGE'] ),self.setting['MODEL_DEFAULT_COMMISSION_PERC'] + self.setting['DEFAULT_SLIPPAGE'])
        self.alpha = 0
        

    # def step(self, action,count_play_steps):
    #     """ 
    #         買進資產減少
    #         賣出資產增加
    #         # 歸一化處理:
    #         #     通過將每次交易的獎勵除以該股票當時的價格，對獎勵進行歸一化。
    #         #     這種方法將每次交易的獎勵調整到一個相對一致的範圍，使其對總資金量的影響更加均衡。
            
            
    #     Args:
    #         action (_type_): _description_
    #     """
    #     assert isinstance(action, Actions)       
        
    #     reward = 0.0
    #     change_moeny = 00
    #     done = False
    #     close = self._cur_close()        
        
    #     # 慢慢更新難度
    #     self.update_alpha(count_play_steps)
    #     price_ratio_factor = close / self.initial_investment
        
    #     if action == Actions.Buy and self.get_now_share < 3 :
    #         # 記錄開盤價
    #         self.get_now_share +=1
    #         change_moeny = -(close * (1 + self.alpha)) / price_ratio_factor # 資產減少
            

    #     elif action == Actions.Close and self.get_now_share > 0 :
    #         self.get_now_share -=1                                                        
    #         change_moeny = (close * (1 - self.alpha)) / price_ratio_factor
            
    #     self._offset += 1
    #     # 判斷遊戲是否結束
    #     done |= self._offset >= self._prices.close.shape[0] - 1 
    #     self.current_money = self.current_money + change_moeny
        
    #     reward = (self.current_money - self.initial_investment ) / self.initial_investment *100
    #     return reward, done
    
    def step(self, action,count_play_steps):
        """ 
            買進資產減少
            賣出資產增加
            # 歸一化處理:
            #     通過將每次交易的獎勵除以該股票當時的價格，對獎勵進行歸一化。
            #     這種方法將每次交易的獎勵調整到一個相對一致的範圍，使其對總資金量的影響更加均衡。
            
            # 在第二次測試的時候我認為如果一直傳遞輸錢的資訊出去 會導致神經網絡歸零
        Args:
            action (_type_): _description_
        """
        assert isinstance(action, Actions)       
        
        reward = 0.0
        done = False
        close = self._cur_close()        
        
        # 慢慢更新難度
        self.update_alpha(count_play_steps)
        
        price_ratio_factor = close / self.initial_investment
        
        if action == Actions.Buy and self.get_now_share < 3 :
            # 記錄開盤價
            self.get_now_share +=1
            reward = -(close * (1 + self.alpha)) / price_ratio_factor # 資產減少
            

        elif action == Actions.Close and self.get_now_share > 0 :
            self.get_now_share -=1                                                        
            reward = (close * (1 - self.alpha)) / price_ratio_factor
            
        self._offset += 1
        # 判斷遊戲是否結束
        done |= self._offset >= self._prices.close.shape[0] - 1 
        
        
        
        return reward, done

class Env():
    def __init__(self,data_type:str, random_ofs_on_reset=True) -> None:
        """
            用來建構完整環境
            
        """       
        self._state = State(bars_count=50)
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

        # print("此次步數為:",offset)
        self._state.reset(prices, offset)
        return self._state.encode()
    
    
    def step(self, action_idx,count_play_steps):
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
        reward, done = self._state.step(action,count_play_steps) # 這邊會更新步數
        obs = self._state.encode() # 呼叫這裡的時候就會取得新的狀態
        
        info = {
                "instrument": self._instrument,
                "offset": self._state._offset,
                }
        
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