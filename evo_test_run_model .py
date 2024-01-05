
from Major.DataProvider import DataProvider
from EVO import Backtest

if __name__ == "__main__":    
    def _get_avgloss() -> dict:
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
        avgloss_df = DataProvider().SQL.read_Dateframe('avgloss')
        avgloss_df = avgloss_df[['strategyName', 'avgLoss']]
        avgloss_df.set_index('strategyName', inplace=True)
        avgloss_data = avgloss_df.to_dict('index')
        return {key: value['avgLoss'] for key, value in avgloss_data.items()}

    app = Backtest.Quantify_systeam_DQN(init_cash=20000, formal=False)
    
    # ['BCHUSDT', 'COMPUSDT','OGNUSDT', 'RNDRUSDT']
    # ['SOLUSDT', 'BTCUSDT', 'BTCDOMUSDT', 'DEFIUSDT', 'XMRUSDT', 'AAVEUSDT', 'TRBUSDT', 'MKRUSDT']
    # ['MATICUSDT','DOGEUSDT','XRPUSDT','LTCUSDT','AVAXUSDT']
    # 'LQTYUSDT', 'BANDUSDT', 'TOMOUSDT', 'INJUSDT', 'LINKUSDT', 'ANTUSDT', 'XVSUSDT'
    app.Portfolio_register(['BTCUSDT'], _get_avgloss())
    app.Portfolio_start()
