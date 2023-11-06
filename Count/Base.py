import pandas as pd

class Pandas_count():
    @staticmethod
    def momentum(price: pd.Series, periond: int) -> pd.Series:
        """
            取得動量
        """
        lagPrice = price.shift(periond)
        momen = price / lagPrice - 1
        return momen


class Event_count():
    @staticmethod
    def get_index(data: dict) -> list:
        return list(data.keys())