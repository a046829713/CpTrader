import pandas as pd
import collections
import numpy as np
import time

PricesObject = collections.namedtuple(
    'Prices', field_names=['open', 'high', 'low', 'close', 'volume'])


class DataFeature():
    """
        用來轉換成 類神經網絡可以使用的資料特徵
        用來產生資料特徵,
        目前設計是訓練模式才會使用到
    """

    def __init__(self, data_type :str = None) -> None:
        self.targetsymbols = [ 'BTCUSDT']
        self.data_type = data_type

    def load_relative(self):
        array_data = self.df.values
        volume_change = self.calculate_volume_change(array_data[:, 4])
        return self.prices_to_relative(PricesObject(open=array_data[:, 0],
                                                    high=array_data[:, 1],
                                                    low=array_data[:, 2],
                                                    close=array_data[:, 3],
                                                    volume=volume_change,
                                                    ))

    def calculate_volume_change(self, volumes):
        """
        Calculate relative volume change
        """
        shift_data = np.roll(volumes, 1)
        shift_data[0] = 0
        diff_data = volumes - shift_data
        # 如果除數為0會返回0
        volume_change = np.divide(
            diff_data, volumes, out=np.zeros_like(diff_data), where=volumes != 0)
        return volume_change

    def prices_to_relative(self, prices):
        """
        # 原始作者不知道為甚麼,使用原始的volume, 我打算使用前一根量的變化來餵給神經網絡
        Convert prices to relative in respect to open price
        :param ochl: tuple with open, close, high, low
        :return: tuple with open, rel_close, rel_high, rel_low
        """
        assert isinstance(prices, PricesObject)
        rh = (prices.high - prices.open) / prices.open
        rl = (prices.low - prices.open) / prices.open
        rc = (prices.close - prices.open) / prices.open
        
        
        if self.data_type == 'train_data':            
            split_num = int(len(prices.open) * 0.7) 
            return PricesObject(open=prices.open[:split_num], high=rh[:split_num], low=rl[:split_num], close=rc[:split_num], volume=prices.volume[:split_num])
        elif self.data_type == 'test_data':
            split_num = int(len(prices.open) * 0.7)            
            return PricesObject(open=prices.open[split_num:], high=rh[split_num:], low=rl[split_num:], close=rc[split_num:], volume=prices.volume[split_num:])
        elif self.data_type=='all_data':
            return PricesObject(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

    def get_train_net_work_data(self) -> dict:
        """
            用來取得類神經網絡所需要的資料
        """
        out_dict = {}
        for symbol in self.targetsymbols:
            df = pd.read_csv(f'DQN\{symbol}-F-15-Min.csv')
            df.set_index('Datetime',inplace=True)
            self.df = df
            out_dict.update({symbol: self.load_relative()})
        return out_dict