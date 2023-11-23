import pandas as pd
import numpy as np
import pandas as pd
from Count import nb


class Np_Order_Info(object):
    """ 用來處理訂單的相關資訊
    Args:
        object (_type_): _description_
    """

    def __init__(self, datetime_list,
                 orders: np.ndarray,
                 marketpostion: np.ndarray,
                 entryprice: np.ndarray,
                 buy_Fees: np.ndarray,
                 sell_Fees: np.ndarray,
                 OpenPostionprofit: np.ndarray,
                 ClosedPostionprofit: np.ndarray,
                 profit: np.ndarray,
                 Gross_profit: np.ndarray,
                 Gross_loss: np.ndarray,
                 all_Fees: np.ndarray,
                 netprofit: np.ndarray,
                 ) -> None:
        # 取得order儲存列
        self.order = pd.DataFrame(datetime_list, columns=['Datetime'])
        self.order['Order'] = orders
        self.order['Marketpostion'] = marketpostion
        self.order['Entryprice'] = entryprice
        self.order['Buy_Fees'] = buy_Fees
        self.order['Sell_Fees'] = sell_Fees
        self.order['OpenPostionprofit'] = OpenPostionprofit
        self.order['ClosedPostionprofit'] = ClosedPostionprofit
        self.order['Profit'] = profit
        self.order['Gross_profit'] = Gross_profit
        self.order['Gross_loss'] = Gross_loss
        self.order['all_Fees'] = all_Fees
        self.order['netprofit'] = netprofit

        # 壓縮資訊減少運算
        self.order = self.order[self.order['Order'] != 0]
        self.order.set_index("Datetime", inplace=True)

        # 取得需要二次運算的資料(計算勝率，賠率....繪圖)
        self.ClosedPostionprofit_array = self.order['ClosedPostionprofit'].to_numpy(
        )
        
        
        print(self.order)

    def register(self, strategy_info):
        """將原始策略的資訊記錄起來

        Args:
            strategy_info (_type_): _description_
        """
        self.strategy_info = strategy_info

    @property
    def holdtime(self) -> float:
        """
            計算持有倉位時間 和 未持有倉位時間

        Returns:
            _type_: _description_
        """
        pass


    @property
    def avgloss(self) -> float:
        """
            透過毛損來計算平均策略虧損

        Returns:
            _type_: _description_
        """
        if self.LossTrades == 0:
            return -100.0  # 由於都沒有交易輸的紀錄
        else:
            return self.order['Gross_loss'].iloc[-1] / self.LossTrades

    @property
    def TotalTrades(self) -> int:
        """ 透過訂單的長度即可判斷交易的次數

        Returns:
            _type_: _description_
        """
        return divmod(len(self.order), 2)[0]

    @property
    def WinTrades(self) -> int:
        """ 取得獲勝的次數
        Returns:
            _type_: _description_
        """
        return len(self.order['Profit'][self.order['Profit'] > 0])

    @property
    def LossTrades(self) -> int:
        """取得失敗的次數

        Returns:
            int: to get from profit
        """
        return len(self.order['Profit'][self.order['Profit'] < 0])

    @property
    def Percent_profitable(self) -> float:
        """取得盈利(勝率)(非百分比) 不全部輸出小數點位數

        Returns:
            float: _description_
        """
        return round(self.WinTrades / self.TotalTrades, 3)

    @property
    def drawdown(self):
        """
            取得回撤金額
        """
        return nb.get_drawdown(self.ClosedPostionprofit_array)

    @property
    def drawdown_per(self):
        """
            取得回撤百分比

        """

        if self.__class__.__name__ == "Portfolio_Order_Info":
            return nb.get_drawdown_per(self.ClosedPostionprofit_array, self.Portfolio_initcash)
        else:
            return nb.get_drawdown_per(self.ClosedPostionprofit_array, self.strategy_info.init_cash)

    @property
    def UI_indicators(self):
        """
            可根据价格下跌的深度和持续时间来衡量下行风险
        Returns:
            _type_: _description_
        """
        sumallDD = np.sum(self.drawdown_per**2)
        ROI = (
            self.ClosedPostionprofit_array[-1] / self.strategy_info.init_cash)-1
        ui_ = (ROI*100) / ((sumallDD / self.TotalTrades)**0.5)
        return ui_


class Portfolio_Order_Info(Np_Order_Info):
    def __init__(self, datetime_list, orders, stragtegy_names, Portfolio_profit, Portfolio_ClosedPostionprofit, Portfolio_initcash, sizes):
        self.order = pd.DataFrame(datetime_list, columns=['Datetime'])
        self.order['Order'] = orders
        self.order['StragtegyNames'] = stragtegy_names
        self.order['Profit'] = Portfolio_profit
        self.order['ClosedPostionprofit'] = Portfolio_ClosedPostionprofit
        self.order['sizes'] = sizes
        self.order.set_index("Datetime", inplace=True)

        self.ClosedPostionprofit_array = self.order['ClosedPostionprofit'].to_numpy(
        )
        self.Portfolio_initcash = Portfolio_initcash
        self.norepeat_stragtegy_names = set(stragtegy_names)

    def get_last_status(self):
        last_status = {}
        for each_stragtegy_name in self.norepeat_stragtegy_names:
            each_df = self.order[self.order['StragtegyNames']
                                 == each_stragtegy_name]
            current_marketpostion = each_df['Order'].sum()
            current_size = each_df['sizes'].iloc[-1]
            last_status.update(
                {each_stragtegy_name: [current_marketpostion, current_size]})

        return last_status