from AC2 import AC2_brain
from AC2 import environment
from AC2.ExperienceBuffer import ExperienceBuffer
from utils.AppSetting import AppSetting
import torch
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# def main():
#     env = environment.Env(reset_on_close=True, random_ofs_on_reset= False)    
#     agent = AC2_brain.PolicyGradientAgent(lr=0.0001,input_dims=env.reset().shape[0],gamma=1)    
#     # 論文有提到製作緩衝區
#     buffer = ExperienceBuffer(max_size=100000)
    
#     n_games = 3000
#     scores =[]
    
#     # 緩衝區的最少數量
#     REPLAY_INITIAL = 10000
    
    
#     # 配置
#     setting = AppSetting.get_ooas_setting()
    
#     for i in range(n_games):
#         done = False
#         observation = env.reset()
#         score = 0
#         while not done:
#             action = agent.choose_action(observation)            
#             oberservation_, reward, done, info = env.step(action)
#             score +=reward            
#             # 記錄(商品步數)
#             buffer.add((observation, action, reward ,oberservation_,info)) #G
#             observation = oberservation_
        
#         # 當緩衝區的資料足夠多的時候
#         if buffer.size() < REPLAY_INITIAL:
#             continue
        
#         # 每做完一次批次更新 # 論文中明確提到抽樣學習
#         batch = buffer.sample(setting['BATCH_SIZE'])

#         # # 假設 states 和 possible_actions 已經定義
#         # state_batch = torch.Tensor([s for (s,a,r,n_s) in batch]).to(agent.policy.device) #L
#         possible_actions = [0,1,2]  # 這裡填入所有可能的行動

#         # # 假設G_values是先前計算好的
#         # G_values = agent.calculate_G(batch)  # 假設batch是先前抽取的經驗

       

#         for index,_expeience in enumerate(batch):
#             best_action = agent.find_optimal_action(env,_expeience, possible_actions)
#             print('*'*120)
#             # agent.update_policy(state, best_action)
        
# main()

from AC2.AC2_brain import LSTMModel
import matplotlib.pyplot as plt



def count():
    env = environment.Env(reset_on_close=True, random_ofs_on_reset= False)


    features = []
    features.append(env.reset())
    for i in range(31):    
        obs, reward, done, info = env.step(np.random.choice([0,1,2]))
        features.append(obs)

    
    # 模型參數設定
    input_size = 5  # 特徵數
    hidden_size = 8  # LSTM 單元數
    num_layers = 3  # LSTM 層數
    num_classes = 3  # 輸出類別數
    dropout_rate = 0.2  # Dropout 比率
    
    # 假設數據
    batch_size = 32  # 批次大小
    seq_length = 50  # 時間序列長度

    # 生成隨機特徵和標籤
    features = torch.tensor(features)
    labels = torch.randint(low=-1, high=2, size=(batch_size,))  # 生成 -1, 0, 1 的隨機標籤

    # 將標籤轉換為 one-hot 編碼
    labels_one_hot = torch.nn.functional.one_hot(labels + 1, num_classes=3)  # 將標籤 +1 以匹配索引 0, 1, 2
    

    # 創建數據集
    dataset = TensorDataset(features, labels_one_hot)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 創建模型
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)

    # 訓練模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    
    for inputs, targets in dataloader:
        outputs = model(inputs)
        print(outputs)



count()