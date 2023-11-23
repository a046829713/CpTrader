"""
Agent is something which converts states into actions and has state
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from . import actions
import time

class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        """
        初始化方法接受以下參數：
        dqn_model: 一個模型，用於預測給定狀態下每個可能行動的價值。
        action_selector: 根據dqn_model的輸出選擇行動的策略。
        device: 用於運算的設備（例如，"cpu"或"cuda"）。
        preprocessor: 用於預處理狀態的函數。默認使用上面的default_states_preprocessor。

        Args:
            dqn_model (_type_): _description_
            action_selector (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".
            preprocessor (_type_, optional): _description_. Defaults to default_states_preprocessor.
        """
        
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        

        # cnn會特別用到這個
        # marketpositions = states[:, -2, 0]  # 從 states 提取部位信息
        
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()        
        # actions = self.action_selector(q,marketpositions)
        actions = self.action_selector(q)
        
        return actions, agent_states
    
    



class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    這個TargetNet類是用來管理DQN中的目標網絡（Target Network）。為了確保DQN的穩定學習，目標網絡的作用是提供固定的目標Q值，而不是每一步都變動。下面我會詳細解釋每個方法：

        __init__(self, model):

        功能：初始化方法。當你建立這個類的一個實例時，它將被調用。
        參數：
        model: 主要的Q網絡模型。
        運作方式：
        它保存了一個主Q網絡的參考(self.model)。
        並創建了該模型的一個深拷貝(self.target_model)。這個拷貝代表目標網絡，其權重初始化時是和主網絡相同的，但在後續的訓練中會被不同步的方式更新。
        sync(self):

        功能：將主網絡的權重複製到目標網絡。
        運作方式：
        使用load_state_dict方法將self.model的權重複製到self.target_model。
        alpha_sync(self, alpha):

        功能：以某種比例混合主網絡和目標網絡的權重，並將結果存儲在目標網絡中。
        參數：
        alpha: 混合因子，值在0到1之間。它代表保留目標網絡權重的比例，而1-alpha則代表從主網絡中取得的比例。
        運作方式：
        首先，進行兩個確認：
        確保alpha是浮點數。
        確保alpha的值在0到1之間。
        使用state_dict取得主網絡和目標網絡的權重。
        用迴圈遍歷每一個權重，根據alpha值進行混合，並將結果保存到目標網絡中。
        總之，TargetNet類提供了兩種同步主網絡和目標網絡權重的策略：一是直接複製，二是加權混合。這些策略確保了DQN學習的穩定性，避免了Q值估計中的大幅波動。
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        """self.model.state_dict() 是PyTorch中的一個功能，用於取得模型中所有的參數（包括權重和偏差等）的有序字典(OrderedDict)。

        當你訓練一個神經網絡模型時，該模型中的每一層都有相對應的參數（例如全連接層的權重和偏差、卷積層的濾波器權重等）。這些參數在訓練過程中會不斷更新。

        state_dict() 方法能夠幫助我們以一個統一和方便的方式取得模型的所有參數。這對於模型的保存和載入，或者模型權重的複製和移動都非常有用。

        具體來說：

        返回值：返回一個有序字典，其中的鍵是每個參數的名稱，值是相對應的參數值（通常是一個tensor）。

        用途：

        模型的保存與載入：你可以使用torch.save(model.state_dict(), PATH)來保存模型的參數，然後在需要的時候使用model.load_state_dict(torch.load(PATH))來載入。
        權重的複製和移動：例如在DQN中，我們可能需要將一個模型的權重複製到另一個模型中，這可以通過target_model.load_state_dict(source_model.state_dict())來完成。
        注意：state_dict()只保存模型的參數，而不保存模型的結構。因此，當你載入參數時，必須先有一個相同結構的模型實例。
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    # TODO: unify code with DQNAgent, as only action selector is differs.
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


class ActorCriticAgent(BaseAgent):
    """
    Policy agent which returns policy and value tensors from observations. Value are stored in agent's state
    and could be reused for rollouts calculations by ExperienceSource.
    """
    def __init__(self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v, values_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        return np.array(actions), agent_states
