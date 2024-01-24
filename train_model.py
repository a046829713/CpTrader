from utils.AppSetting import AppSetting
import os
import gym
import numpy as np
import torch
import torch.optim as optim
import time
from DQN.lib import environ, models, common
from DQN import ptan
from DQN.lib.DataFeature import DataFeature
from tensorboardX import SummaryWriter
from datetime import datetime
from DQN.lib.analyze import analyze

# 問題紀錄:
# 1.為甚麼每次都要從5-6根K棒開始運行程序?
# 2.如果更改特徵的話,會不會影響到整體?

# 代辦事項:
# 驗證程序(run_model)
# 現貨資料爬取
# 多商品訓練的轉換,=> 1.訓練時候隨機轉換商品 2.EPSILON_STEPS 要隨著商品總數量做變化


# 當運行tensorboard的時候,必須要確保路徑的正確性
# tensorboard --logdir="C:\runs\20231019-154810-conv-"

TARGET_NET_SYNC = 1000
GAMMA = 1.0
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
REWARD_STEPS = 2
EPSILON_STOP = 0.1
CHECKPOINT_EVERY_STEP = 100
VALIDATION_EVERY_STEP = 100000
WRITER_EVERY_STEP = 100

setting = AppSetting.get_DQN_setting()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EPSILON_STEPS = 2000000 
    EPSILON_STEPS = 1000000
    env = environ.StocksEnv(bars_count=setting['BARS_COUNT'],reset_on_close = setting['RESET_ON_CLOSE'],state_1d=setting['STATE_1D'], reward_on_close=setting['REWARD_ON_CLOSE'], volumes=setting['VOLUMES_TURNON'])
    symbol_code = list(env._prices.keys())[0]
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)

    if setting['STATE_1D']:
        net = models.DQNConv1D(env.observation_space.shape,
                               env.action_space.n).to(device)
        writer = SummaryWriter(
            log_dir=os.path.join(
                'C:\\', 'runs', datetime.strftime(
                    datetime.now(), "%Y%m%d-%H%M%S") + '-conv-'))
    else:
        net = models.SimpleFFDQN(
            env.observation_space.shape[0], env.action_space.n).to(device)
        writer = SummaryWriter(comment="-simple-")

    
    # 用來分析 net    
    # print("參數分析:",analyze(net).GetAllParameters())
    
    
    tgt_net = ptan.agent.TargetNet(net)

    
    # 貪婪的選擇器
    selector = ptan.actions.EpsilonGreedyActionSelector(
        setting['EPSILON_START'])

    agent = ptan.agent.DQNAgent(net, selector, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, steps_count=REWARD_STEPS)

    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=setting['LEARNING_RATE'])

    # main training loop
    # 加載檢查點如果存在的話
    checkpoint_path = r''
    # checkpoint_path = 
    if checkpoint_path and os.path.isfile(checkpoint_path) :
        print("資料繼續運算模式")
        saves_path = checkpoint_path.split('\\')
        saves_path = os.path.join(saves_path[0],saves_path[1])
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_idx = checkpoint['step_idx']        
        best_mean_val = checkpoint['best_mean_val']
        
    else:
        print("建立新的儲存點")
        
        # 用來儲存的位置
        saves_path = os.path.join(setting['SAVES_PATH'], datetime.strftime(
            datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(setting['BARS_COUNT']) + 'k-' + str(setting['REWARD_ON_CLOSE'])  +'-' + symbol_code)

        os.makedirs(saves_path, exist_ok=True)
        step_idx = 0
        best_mean_val = None
    

    with common.RewardTracker(writer, np.inf, group_rewards=2) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)

            # 探索率
            selector.epsilon = max(
                EPSILON_STOP, setting['EPSILON_START'] - step_idx / EPSILON_STEPS)

            # [(-2.5305491551459296, 10)]
            # 跑了一輪之後,清空原本的數據,並且取得獎勵
            new_rewards = exp_source.pop_rewards_steps()

            if new_rewards:
                reward_tracker.reward(
                    new_rewards[0], step_idx, selector.epsilon)
            
            if len(buffer) < REPLAY_INITIAL:
                continue

            # if step_idx % setting['EVAL_EVERY_STEP'] == 0:
            #     mean_vals = []
                
            #     with torch.no_grad():  # 禁用梯度計算
            #         for _ in range(setting['NUM_EVAL_EPISODES']):
            #             # 更新驗證資料
            #             eval_states = common.update_eval_states(
            #                 buffer,setting['STATES_TO_EVALUATE'] )
            #             # 計算平均
            #             mean_val = common.calc_values_of_states(
            #                 eval_states, net, device=device)                        
            #             mean_vals.append(mean_val)

            #     mean_of_means = np.mean(mean_vals)

            #     writer.add_scalar("values_mean", mean_of_means, step_idx)
            #     if best_mean_val is None or best_mean_val < mean_of_means:
            #         if best_mean_val is not None:
            #             print("%d: Best mean value updated %.3f -> %.3f" %
            #                   (step_idx, best_mean_val, mean_of_means))
            #         best_mean_val = mean_of_means
            #         torch.save(net.state_dict(), os.path.join(
            #             saves_path, "mean_val-%.3f.data" % mean_of_means))

            optimizer.zero_grad()
            batch = buffer.sample(setting['BATCH_SIZE'])

            loss_v = common.calc_loss(
                batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)

            
            if step_idx % WRITER_EVERY_STEP == 0:
                writer.add_scalar("Loss_Value", loss_v.item(), step_idx)

            loss_v.backward()
            optimizer.step()

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()


            # 保存檢查點的函數
            def save_checkpoint(state, filename):
                torch.save(state, filename)

            
            print("目前次數:",step_idx,"保存點:",CHECKPOINT_EVERY_STEP)
            # 在主訓練循環中的合適位置插入保存檢查點的代碼
            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                idx = step_idx // CHECKPOINT_EVERY_STEP
                checkpoint = {
                    'step_idx': step_idx,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'selector_state': selector.epsilon,
                    'best_mean_val': best_mean_val,
                }
                save_checkpoint(checkpoint, os.path.join(saves_path, f"checkpoint-{idx}.pt"))
