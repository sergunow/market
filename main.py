from datetime import datetime
import multiprocessing
import time
import random
import gym
import json
import datetime as dt

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

import random

import keras
import numpy as np
from keras import layers
from keras.layers import Dense, Flatten, LSTM, Dropout, Input, ConvLSTM2D, BatchNormalization, MaxPooling1D, \
    TimeDistributed, Conv2D, Conv3D, MaxPooling3D
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent, DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from stable_baselines import PPO2, TRPO
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from tqdm import tqdm

from env.BinanceEnv import BinanceEnv
import pandas as pd
from os import system

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


def main():
    # env = BinanceEnv()
    env = make_vec_env(BinanceEnv, n_envs=8)
    signals = pd.read_csv('./signals.csv')
    # # signals.head()

    def dummy_expert(_obs):
        # print(env.simulator.DataSimulator.current_step)
        state_id = env.simulator.DataSimulator.state_id
        step_signals = signals[signals['open_id'] == state_id]
        if len(step_signals) != 0:
            if step_signals.iloc[0]['type'] == 'long':
                return 0
            else:
                return 3
        else:
            step_signals = signals[signals['close_id'] == state_id]
            if len(step_signals) != 0:
                if step_signals.iloc[0]['type'] == 'long':
                    return 1
                else:
                    return 4
        return 6

    # generate_expert_traj(dummy_expert, 'trading_signals', env, n_episodes=40)
    # dataset = ExpertDataset(expert_path='trading_signals.npz', batch_size=128)
    # env.is_testing = True
    model = PPO2('MlpLstmPolicy', env, verbose=1)
    # model.pretrain(dataset, n_epochs=20000)
    # model.save('trading')
    # # As an option, you can train the RL agent
    start = datetime.now()
    model.learn(total_timesteps=1000)
    print(datetime.now() - start)
    # model.save('trading')
    # env.is_testing = True
    # model = PPO2.load('trading')
    # env = model.get_env()
    reward_sum = 0.0
    for _ in range(0, 10):
        obs = env.reset()
        for _ in range(10000):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()

