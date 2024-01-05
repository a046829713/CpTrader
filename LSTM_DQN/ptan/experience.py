import gym
import torch
import random
import collections
from torch.autograd import Variable
import time
import numpy as np

from collections import namedtuple, deque
import time
from .agent import BaseAgent
from .common import utils

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        
        self.agent = agent # <DQN.ptan.agent.DQNAgent object at 0x0000016C63B05D60>
        self.steps_count = steps_count # 3
        self.steps_delta = steps_delta # 1
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized # False

    
    def __iter__(self):
        print("ExperienceSource iter 測試進入次數:",)
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count)) 
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state()) 

            
            # states # 隨機狀態
            # agent_states # [None]
            # histories # [deque([], maxlen=3)]
            # cur_rewards # [0.0]
            # cur_steps # [0]
            
            
        iter_idx = 0
        while True:
            actions = [None] * len(states) # [None]            
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state) # 狀態
                    states_indices.append(idx) # 索引
            
            if states_input:
                # 會吐出動作和新狀態[2] [None] # 不過原作者這邊好似沒有使用到agent_states
                states_actions, new_agent_states = self.agent(states_input, agent_states)

                for idx, action in enumerate(states_actions):                    
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            
            # [[2]]
            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0           
            
            # 參數檢查: [<TimeLimit<StocksEnv instance>>] 參數檢查: [[2]]
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                # 0 (<TimeLimit<StocksEnv instance>>, [2])                              
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(action_n)
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    # ofs = 0,action = 2,next_state = array[],r,is_done = False
                                
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]
                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    
                    states[idx] = next_state
                    # 遊戲完成之後計算總獎勵
                    if is_done:
                        # generate tail of history
                        while len(history) >= 1:
                            yield tuple(history)
                            history.popleft()
                        self.total_rewards.append(cur_rewards[idx])                        
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                    
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    這是一個圍繞ExperienceSource的包裝器（wrapper），
    用於在我們只需要初始和最終狀態時，防止在重播緩衝區（replay buffer）
    中儲存完整的軌跡。對於每一個軌跡片段，它會計算折扣獎勵，並且只輸出第一個和最後一個狀態，以及在初始狀態中採取的行動。

    如果在劇集結束時我們有部分軌跡，那麼last_state將為None。
    
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield (exp[-1].done,ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state))


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]






class ExperienceReplayBuffer:
    def __init__(self, experience_source,buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        self.buffer_size  = buffer_size
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = deque(maxlen=self.buffer_size)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """

        """
        return [self.buffer[i] for i in range(batch_size)]
    
    def reset_memory(self):
        self.buffer = deque(maxlen=self.buffer_size)
        
    def _add(self, sample):
        """
        
            將跌代的資料帶入
            萬一超過就覆寫
        Args:
            sample (_type_): _description_
        """
        self.buffer.append(sample)

    def populate(self, samples):
        """
        將樣本填入緩衝區中
        Populates samples into the buffer
        :param samples: how many samples to populate
        
        <class 'ptan.experience.ExperienceFirstLast'>
        entry: ExperienceFirstLast(state=array([ 0.00773994, -0.01083591,  0.00773994,  0.00456621, -0.01065449,
        0.00456621,  0.00607903, -0.00455927,  0.00455927,  0.        ,
       -0.01783061, -0.00148588,  0.00437956, -0.01021898, -0.00291971,
        0.00442478, -0.02359882, -0.02359882,  0.01226994, -0.00153374,
        0.00306748,  0.01076923, -0.00615385,  0.00153846,  0.00310559,
       -0.01086957, -0.00465839,  0.02503912, -0.00312989,  0.02190923,
        0.        ,  0.        ], dtype=float32), action=1, reward=-2.7099031710120034, last_state=array([ 0.00607903, -0.00455927,  0.00455927,  0.        , -0.01783061,
       -0.00148588,  0.00437956, -0.01021898, -0.00291971,  0.00442478,
       -0.02359882, -0.02359882,  0.01226994, -0.00153374,  0.00306748,
        0.01076923, -0.00615385,  0.00153846,  0.00310559, -0.01086957,
       -0.00465839,  0.02503912, -0.00312989,  0.02190923,  0.00311042,
       -0.00777605, -0.00311042,  0.00944882,  0.        ,  0.0015748 ,
        1.        , -0.02603369], dtype=float32))
        """
        for _ in range(samples):
            done, entry = next(self.experience_source_iter)
            self._add(entry)
            
        return done