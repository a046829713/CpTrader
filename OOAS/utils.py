import itertools



class Datatransformer():   
    @staticmethod
    def count_cartesian_product(SA_SPACE,ACTION_SPACE):
        # 計算狄卡爾積
        return list(itertools.product(SA_SPACE, ACTION_SPACE))


    @staticmethod
    def calculate_reward(sa_current, sa_next, price, alpha):
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
        price: 当前时间步的標記價格。
        alpha: 買入時的交易成本比率。卖出时的交易成本比率。 (手續費,交易稅,滑價)

        返回:
            奖励值。
        """
        if sa_next > sa_current:
            reward = -abs(sa_next - sa_current) * (1 + alpha) * price
        else:
            reward = abs(sa_next - sa_current) * (1 - alpha) * price

        return reward