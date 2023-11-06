import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import time
from utils.AppSetting import AppSetting

setting = AppSetting.get_DQN_setting()

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)

        self.bars_count = bars_count
        self.commission_perc = commission_perc
        

        # self.slippagem = slippagem
        # 後來將滑價拿掉,感覺在計算上,因為reward每跟K棒都會計算難以使用滑價來衡量,但是在實際交易的時候 我認為在考慮就好

        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # 這邊準備特徵的時候 確實有準備了部位的方向
        # 和自從開啟之後到現在的損益
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3*self.bars_count + 1 + 1, )

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
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1

        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            # 其實我覺得丟這個進去,好像沒什麼用
            res[shift] = (self._cur_close() - self.open_price) / \
                self.open_price
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
            修改了原作者的setp程序            
            保留了在買進和平倉時​​扣除佣金的邏輯
            當持倉時每支K棒的收盤價與前一支K棒的收盤價進行比較根據漲跌給予即時獎勵。
            在平倉時根據交易的整體盈利或虧損給予獎勵或懲罰。
        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()

        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close * (1 + setting['DEFAULT_SLIPPAGE'])            
            reward -= self.commission_perc  # 扣除佣金
            
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            reward += 100.0 * (close* (1 - setting['DEFAULT_SLIPPAGE']) - self.open_price) / self.open_price            
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close # 上一根的收盤價
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        # 訓練時 每一根K棒都給獎勵 (及時獎勵機制)
        if self.have_position and not self.reward_on_close:
            
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done
        
class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
            
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / \
                self.open_price
               
        return res


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count,
                 commission=setting['MODEL_DEFAULT_COMMISSION_PERC'], reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False):

        
        assert isinstance(prices, dict)
        self._prices = prices

        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:

            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)

        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state

        # self._prices.keys() = dict_keys(['YNDX'])

        # 我認為要訓練多個目標要檢查這裡
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        # Prices(open=array([97.6, 100.5, 100.5, ..., 567.0, 574.0, 574.0], dtype=object), high=array([0.02356557377049192, 0.014925373134328358, 0.009950248756218905,
        # dtype=object))

        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            # prices.high.shape = (2562,)
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars

            # offset 1617
        else:
            offset = bars

        self._state.reset(prices, offset)
        return self._state.encode()
    
    def step(self, action_idx):
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

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
