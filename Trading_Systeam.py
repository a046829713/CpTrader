from DQN.lib import Backtest
from DQN.lib.Backtest import FastCreateOrder
from utils.AppSetting import AppSetting
from utils.Datatransformer import Datatransformer
from Major.DataProvider import DataProvider, AsyncDataProvider
from binance.exceptions import BinanceAPIException
import time
from datetime import timedelta
import datetime
import sys
import pandas as pd
from utils import BackUp, Debug_tool
from Database.SQL_operate import SqlSentense
import copy
import threading
import asyncio
from Infrastructure.AlertMsg import LINE_Alert


# LINE通知優先拿掉,因為變成每個人不一樣
# 每天關閉的功能尚未加入(原本在UI介面的) 儲存資料的(每天早上8點重新啟動)
# Backtest.Quantify_systeam_DQN 這個可以換成其他的object 不一定需要這個
# 是否要朝web版本前進呢?


# 1.資料保存是否正常
# 2.資料是否可以正確傳入回測系統
# 3.order單是否可以正確依據各個帳戶下單
# 4.檢查資產是否會更新(資金水位功能)

exit_event = threading.Event()  # 定义一个事件


class Trading_system():
    def __init__(self) -> None:
        self.symbol_map = {}
        # 這次新產生的資料
        self.new_symbol_map = {}
        self.dataprovider = DataProvider()
        self.engine = self.buildEngine()
        self.datatransformer = Datatransformer()

        # 用來製作資產平衡
        self.balance_map = {}
        self.GuiStartDay = str(datetime.date.today())

    def DailyChange(self):
        """
            每天都要重新關閉,怕資料量過大,並且會重新讀取每天的強勢標的
        """
        while not exit_event.is_set():            
            if str(datetime.date.today()) != self.GuiStartDay:
                self.click_save_data()
            # 每5分鐘判斷一次就好
            time.sleep(300)
            
            
    def click_save_data(self):
        """ 
        保存資料並關閉程序 注意不能使用replace 資料長短問題

        """
        for name, each_df in self.new_symbol_map.items():
            # 不保存頭尾 # 異步模式
            each_df.drop(
                [each_df.index[0], each_df.index[-1]], inplace=True)

            each_df = each_df.astype(float)
            if len(each_df) != 0:
                # 準備寫入資料庫裡面
                self.dataprovider.save_data(
                    symbol_name=name, original_df=each_df, symbol_type='FUTURES', time_type='1m', exists="append")

        print("保存資料完成-退出程序")
        exit_event.set()  # 设置事件，导致所有线程退出

    def buildEngine(self):
        """ 用來創建回測系統

        Returns:
            _type_: _description_
        """
        return FastCreateOrder(formal=AppSetting.get_trading_permission()['Data_transmission'])

    def checkDailydata(self):
        """
            檢查資料庫中的日資料是否已經回補
            if already update then contiune
        """
        data = self.dataprovider.SQL.get_db_data(
            """select *  from `btcusdt-f-d` order by Datetime desc limit 1""")
        sql_date = str(data[0][0]).split(' ')[0]
        _todaydate = str((datetime.datetime.today() -
                         timedelta(days=1))).split(' ')[0]

        if sql_date != _todaydate:
            # 更新日資料 並且在回補完成後才繼續進行 即時行情的回補
            self.dataprovider.reload_all_data(
                time_type='1d', symbol_type='FUTURES')
        else:
            # 判斷幣安裡面所有可交易的標的
            allsymobl = self.dataprovider.Binanceapp.get_targetsymobls()
            all_tables = self.dataprovider.SQL.read_Dateframe(
                """show tables""")
            all_tables = [
                i for i in all_tables['Tables_in_crypto_data'].to_list() if '-f-d' in i]

            # 當有新的商品出現之後,會導致有錯誤,錯誤修正
            if list(filter(lambda x: False if x.lower() + "-f-d" in all_tables else True, allsymobl)):
                self.dataprovider.reload_all_data(
                    time_type='1d', symbol_type='FUTURES')
    
    def reload_all_futures_data(self):
        """
            用來回補所有日線期貨資料
        """
        self.dataprovider.reload_all_data(time_type='1m', symbol_type='FUTURES')
    
    def exportavgloss(self):
        """
            將avgloss資料匯出
        """
        df = self.dataprovider.SQL.read_Dateframe("avgloss")
        df.set_index("strategyName", inplace=True)
        df.to_csv("avgloss.csv")

    def importavgloss(self):
        """
            將avgloss資料導入
        """
        try:
            df = pd.read_csv("avgloss.csv")
            df.set_index("strategyName", inplace=True)

            self.dataprovider.SQL.change_db_data(
                "DELETE FROM `avgloss`;"
            )
            # 為了避免修改到原始sql 的create 使用append
            self.dataprovider.SQL.write_Dateframe(
                df, "avgloss", exists='append')
        except Exception as e:
            print(f"導入資料錯誤:{e}")

    def count_all_avgloss(self):
        all_symbols = self.dataprovider.Binanceapp.get_targetsymobls()

        # 將資料讀取出來
        avglossdf = self.dataprovider.SQL.read_Dateframe(
            "select strategyName, symbol from avgloss")

        strategylist = avglossdf['strategyName'].to_list()

        print("開始進行更新")
        # 使用 Optimizer # 建立DB
        for eachsymbol in all_symbols:
            target_strategy_name = eachsymbol + '-15K-OB-DQN'
            print(eachsymbol)

            result = Backtest.Optimizer_DQN(
            ).Create_strategy_to_get_avgloss([eachsymbol])
            
            if result['strategyName'] in strategylist:
                self.dataprovider.SQL.change_db_data(
                    SqlSentense.update_avgloss(result))
            else:
                self.dataprovider.SQL.change_db_data(
                    SqlSentense.insert_avgloss(result))

    def export_all_tables(self):
        """
            將資料全部從Mysql寫出

        """
        BackUp.DatabaseBackupRestore().export_all_tables()

    def get_target_symbol(self):
        """ 
        # 取得交易標的

        """
        all_symbols = self.dataprovider.get_symbols_history_data(
            symbol_type='FUTURES', time_type='1d')
        example = self.datatransformer.get_symobl_filter_useful(all_symbols)

        return example

    def check_money_level(self):
        """
            取得實時運作的資金水位 
            並且發出賴通知
            採用USDT 
            {'09XXXXXXXX': 86619.85787802}

            # 每天平衡一次所以 要製作一個5%的平衡機制
        """

        # 這邊取得的是幣安的保證金(以平倉損益) # 取得所有戶頭的資金
        balance_map = self.dataprovider.Binanceapp.get_alluser_futuresaccountbalance()

        for key, value in balance_map.items():
            if key in self.balance_map:
                if abs((self.balance_map[key] - value) / self.balance_map[key]) * 100 > 5:
                    self.balance_map[key] = value
            else:
                self.balance_map[key] = value

    def timewritersql(self):
        """
            寫入保存時間
        """
        self.dataprovider.SQL.change_db_data(
            f""" UPDATE `sysstatus` SET `systeam_datetime`='{str(datetime.datetime.now())}' WHERE `ID`='1';""")

    def printfunc(self, *args):
        out_str = ''
        for i in args:
            out_str += str(i)+" "

        print(out_str)

    def get_catch(self, name, eachCatchDf):
        """
            取得當次回補的總資料並且不存在於DB之中

        """
        if name in self.new_symbol_map:
            new_df = pd.concat(
                [self.new_symbol_map[name].reset_index(), eachCatchDf])
            new_df.set_index('Datetime', inplace=True)
            # duplicated >> 重複 True 代表重複了
            new_df = new_df[~new_df.index.duplicated(keep='last')]
            self.new_symbol_map.update({name: new_df})
        else:
            eachCatchDf.set_index('Datetime', inplace=True)
            self.new_symbol_map.update({name: eachCatchDf})


class AsyncTrading_system(Trading_system):
    def __init__(self) -> None:
        super().__init__()
        self.check_and_reload_dailydata()
        self.process_target_symbol()
        self.init_async_data_provider()
        self.register_portfolio()

    def check_and_reload_dailydata(self):
        """ check if already update, and reload data"""
        print("開始回補日內資料")
        self.checkDailydata()

    def process_target_symbol(self):
        # 取得要交易的標的
        market_symobl = list(map(lambda x: x[0], self.get_target_symbol()))

        # 取得binance實際擁有標的,合併 (因為原本有部位的也要持續追蹤)
        self.targetsymbols = self.datatransformer.target_symobl(
            market_symobl, self.dataprovider.Binanceapp.getfutures_account_name())

    def init_async_data_provider(self):
        # 將標得注入引擎
        self.asyncDataProvider = AsyncDataProvider()

    def register_portfolio(self):
        # 初始化投資組合 (傳入要買的標的物, 並且讀取神經網絡的參數)
        self.engine.Portfolio_register(
            self.targetsymbols, self._get_avgloss())  # 傳入平均虧損的資料
        self.symbol_name: set = self.engine.get_symbol_name()

    def _get_avgloss(self) -> dict:
        """
                    strategyName  freq_time    symbol strategytype updatetime  avgLoss
            0    AAVEUSDT-15K-OB-DQN         15  AAVEUSDT  DQNStrategy 2023-10-20    -2.03
            1     ACHUSDT-15K-OB-DQN         15   ACHUSDT  DQNStrategy 2023-10-21  -100.00
            2     ADAUSDT-15K-OB-DQN         15   ADAUSDT  DQNStrategy 2023-10-20    -0.01
            3    AGIXUSDT-15K-OB-DQN         15  AGIXUSDT  DQNStrategy 2023-10-21    -0.01
            4    AGLDUSDT-15K-OB-DQN         15  AGLDUSDT  DQNStrategy 2023-10-21    -0.01
            ..                   ...        ...       ...          ...        ...      ...
            198   YGGUSDT-15K-OB-DQN         15   YGGUSDT  DQNStrategy 2023-10-21    -0.02
            199   ZECUSDT-15K-OB-DQN         15   ZECUSDT  DQNStrategy 2023-10-20    -1.14
            200   ZENUSDT-15K-OB-DQN         15   ZENUSDT  DQNStrategy 2023-10-21    -1.31
            201   ZILUSDT-15K-OB-DQN         15   ZILUSDT  DQNStrategy 2023-10-20     0.00
            202   ZRXUSDT-15K-OB-DQN         15   ZRXUSDT  DQNStrategy 2023-10-20    -0.04
        """
        avgloss_df = self.dataprovider.SQL.read_Dateframe('avgloss')
        avgloss_df = avgloss_df[['strategyName', 'avgLoss']]
        avgloss_df.set_index('strategyName', inplace=True)
        avgloss_data = avgloss_df.to_dict('index')
        return {key: value['avgLoss'] for key, value in avgloss_data.items()}

    async def main(self):
        self.printfunc("Crypto_trading 正式交易啟動")
        LINE_Alert().send_author("Crypto_trading 正式交易啟動")
        # 先將資料從DB撈取出來
        for name in self.symbol_name:
            original_df, eachCatchDf = self.dataprovider.get_symboldata(
                name, QMType='Online')
            self.symbol_map.update({name: original_df})
            self.get_catch(name, eachCatchDf)

        last_min = None
        self.printfunc("資料讀取結束")
        # 透過迴圈回補資料
        while not exit_event.is_set():
            try:
                if datetime.datetime.now().minute != last_min or last_min is None:
                    begin_time = time.time()
                    # 取得原始資料
                    all_data_copy = await self.asyncDataProvider.get_all_data()

                    # 避免在self.symbol_map
                    symbol_map_copy = copy.deepcopy(self.symbol_map)
                    for name, each_df in symbol_map_copy.items():
                        # 這裡的name 就是商品名稱,就是symbol_name EX:COMPUSDT,AAVEUSDT
                        original_df, eachCatchDf = self.datatransformer.mergeData(
                            name, each_df, all_data_copy)
                        self.symbol_map[name] = original_df
                        self.get_catch(name, eachCatchDf)

                    info = self.engine.get_symbol_info()
                    for strategy_name, symbol_name, freq_time in info:
                        # 取得可交易之資料
                        trade_data = self.dataprovider.get_trade_data(
                            self.symbol_map[symbol_name], freq_time)

                        # 類神經網絡會忽略掉最後一根K棒,所以傳遞完整的進去就可以了
                        self.engine.register_data(strategy_name, trade_data)

                    self.printfunc("開始進入回測")
                    # 取得所有戶頭的平衡資金才有辦法去運算口數
                    self.check_money_level()
                    self.engine.Portfolio_start()
                    last_status = self.engine.get_last_status(self.balance_map)
                    self.printfunc('目前交易狀態,校正之後', last_status)

                    current_size = self.dataprovider.Binanceapp.getfutures_account_positions()
                    self.printfunc("目前binance交易所內的部位狀態:", current_size)
                    self.printfunc("*" * 120)

                    # >>比對目前binance 內的部位狀態 進行交易
                    all_order_finally = self.dataprovider.transformer.calculation_size(
                        last_status, current_size, self.symbol_map,exchange_info = self.dataprovider.Binanceapp.getfuturesinfo())

                    print("測試all_order_finally", all_order_finally)

                    # 將order_finally 跟下單最小單位相比
                    all_order_finally = self.dataprovider.transformer.change_min_postion(
                        all_order_finally, self.dataprovider.Binanceapp.getMinimumOrderQuantity())

                    self.printfunc("差異單", all_order_finally)

                    if all_order_finally:
                        self.dataprovider.Binanceapp.execute_orders(
                            all_order_finally, current_size=current_size, symbol_map=self.symbol_map, formal=AppSetting.get_trading_permission()['execute_orders'])

                    self.printfunc("時間差", time.time() - begin_time)
                    last_min = datetime.datetime.now().minute
                    self.timewritersql()
                else:
                    time.sleep(1)

                if self.dataprovider.Binanceapp.trade_count > AppSetting.get_emergency_times():
                    self.printfunc("緊急狀況處理-交易次數過多")
                    sys.exit()

            except Exception as e:
                if isinstance(e, BinanceAPIException) and e.code == -1001:
                    Debug_tool.debug.print_info()
                else:
                    # re-raise the exception if it's not the expected error code
                    Debug_tool.debug.print_info()
                    raise e


def run_asyncio_loop(func, *args, exit_event=None):  # 使用默认参数，使得 exit_event 是可选的
    if func.__name__ == 'subscriptionData':
        asyncio.run(func(*args, exit_event=exit_event))
    elif func.__name__ == 'main':
        asyncio.run(func(*args))

if __name__ == '__main__':
    # 即時交易模式
    # trading_system = AsyncTrading_system()  # 创建类的实例

    # thread1 = threading.Thread(target=run_asyncio_loop,
    #                            args=(
    #                                trading_system.asyncDataProvider.subscriptionData, trading_system.symbol_name),
    #                            kwargs={'exit_event': exit_event})

    # thread2 = threading.Thread(
    #     target=run_asyncio_loop, args=(trading_system.main,))

    # thread3 = threading.Thread(target=trading_system.DailyChange)

    # thread1.start()
    # thread2.start()
    # thread3.start()

    # thread1.join()
    # thread2.join()
    # thread3.join()
    app = Trading_system()
    app.count_all_avgloss()