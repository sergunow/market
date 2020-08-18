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
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail import generate_expert_traj, ExpertDataset

from env.BinanceEnv import BinanceEnv
import pandas as pd

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


def get_model(states, actions):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(30, 3), input_shape=(1, 61, 3, 1), padding="same", return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D((1, 2, 1)))
    model.add(ConvLSTM2D(filters=64, kernel_size=(10, 3), padding="same", return_sequences=True))
    model.add(MaxPooling3D((1, 2, 1)))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True))
    model.add(MaxPooling3D((1, 2, 1)))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    ))
    model.add(Flatten())
    model.add((Dense(units=1024, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(units=actions, activation='linear'))
    print(model.summary())

    return model


def main():
    env = BinanceEnv()
    signals = pd.read_csv('./signals.csv')
    signals.head()
    i = 0
    def dummy_expert(_obs):
        env.render()
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

    # generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_episodes=10)

    # dataset = ExpertDataset(expert_path='dummy_expert_cartpole.npz',
    #                         traj_limitation=1, batch_size=128)
    # # env.is_testing = True
    # model = PPO2('MlpPolicy', env, verbose=1)
    # model.pretrain(dataset, n_epochs=5000)
    #
    # # As an option, you can train the RL agent
    # model.learn(total_timesteps=250000)
    model = PPO2.load('ppo2')
    # Test the pre-trained model
    # env = model.get_env()
    obs = env.reset()

    reward_sum = 0.0
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
