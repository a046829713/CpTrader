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
        self.setting = AppSetting.get_evo_setting()                
        self.commission_perc = self.setting['MODEL_DEFAULT_COMMISSION_PERC']
    
    @property
    def shape(self):
        """        
            根據你的特徵來決定shape
        """
        return (4 * self.bars_count + 1 + 1, )
    
    def reset(self,prices,offset):
        """
            透過reset 來指定需要的標的物
        """
        self._prices = prices
        self._offset = offset
        self.have_position = False        
        self.cost_sum = 0.0
        self.closecash= 0.0
        self.Netprofit  = 10000.0
        self.ClosedPostionprofit = 10000.0
        self.ClosedPostionprofit_list =[]
        
        self.TotalTrades =0 
        self.WinTrades = 0
        self.LossTrades = 0
        self.winmoneys = 0
        self.lossmoneys = 0
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

        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        
        return res
    
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
            重新設計
            最佳動作空間探索的獎勵函數
            
            "找尋和之前所累積的獎勵之差距"

        Args:
            action (_type_): _description_
        """
        assert isinstance(action, Actions)       
        
        reward = 0.0
        done = False
        close = self._cur_close()
        
        # 以平倉損益每局從新歸零
        closecash_diff = 0.0
        # 未平倉損益
        opencash_diff = 0.0
        # 手續費
        cost = 0.0        

        
        # 第一根買的時候不計算未平倉損益
        if self.have_position:
            opencash_diff = (close - self.open_price) / self.open_price        
        
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            # 記錄開盤價
            self.open_price = close * (1 + self.setting['DEFAULT_SLIPPAGE'])            
            cost = self.commission_perc  * self.open_price
                      

        elif action == Actions.Close and self.have_position:
            sell_price = close * (1 - self.setting['DEFAULT_SLIPPAGE'])
            cost = self.commission_perc * sell_price                                 
            self.have_position = False            
            # 計算出賣掉的資產變化率,並且累加起來
            closecash_diff = sell_price - self.open_price
            self.open_price = 0.0
            opencash_diff = 0.0        
            self.TotalTrades +=1
            if closecash_diff >=0:
                self.WinTrades +=1
            else:
                self.LossTrades +=1

            
        # 原始獎勵設計        
        self.cost_sum += cost
        self.closecash += closecash_diff
        
        
        if closecash_diff >=0 :
            self.winmoneys += closecash_diff            
        else:
            self.lossmoneys += closecash_diff        
        
        # 累積的概念為? 淨值 = 起始資金 - 手續費 +  已平倉損益 + 未平倉損益 
        self.Netprofit  =  10000.0 - self.cost_sum + self.closecash +  opencash_diff         
        self.ClosedPostionprofit = self.ClosedPostionprofit - cost + closecash_diff
        self.ClosedPostionprofit_list.append(self.ClosedPostionprofit)       
        
        self._offset += 1
        # 判斷遊戲是否結束
        done |= self._offset >= self._prices.close.shape[0] - 1        
        return 0, done
    
    
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
        info = {
                "instrument": self._instrument,
                "offset": self._state._offset,
                "Netprofit":self._state.Netprofit,
                "ClosedPostionprofit_list":self._state.ClosedPostionprofit_list,
                "TotalTrades":self._state.TotalTrades,
                "WinTrades":self._state.WinTrades,
                "LossTrades":self._state.LossTrades,
                "WinMoney":self._state.winmoneys, # 我的定義為總平倉獲利
                "LossMoney":self._state.lossmoneys, # 我的定義為總平倉損失(有算滑價)
                "Cost":self._state.cost_sum # 我的定義為總費用
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