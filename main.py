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
# from stable_baselines import PPO2, TRPO
# from stable_baselines.common.policies import MlpPolicy

from env.BinanceEnv import BinanceEnv

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


def get_model(states, actions):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(30, 3), input_shape=(1, 29, 60, 3), padding="same", return_sequences=True))
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
    model = get_model(env.observation_space.shape[0], env.action_space.n)
    policy = EpsGreedyQPolicy()
    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=env.action_space.n)
    sarsa.compile('adam', metrics=['mse', 'accuracy'])
    # sarsa.load_weights('sarsa_weights_bnb_07.h5f')
    env.is_testing = False
    sarsa.fit(env, nb_steps=10000000, visualize=False, verbose=1)
    sarsa.save_weights('sarsa_weights_bnb_07_1.h5f', overwrite=True)
    # sarsa.load_weights('sarsa_weights_bnb_07_1.h5f')
    # env.simulator = False
    env.is_testing = True
    scores = sarsa.test(env, nb_episodes=1, visualize=False)
    print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))

    _ = sarsa.test(env, nb_episodes=10, visualize=True)
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=10000000)
    # model.save("trpo_cartpole")
    #
    # del model  # remove to demonstrate saving and loading
    #
    # model = TRPO.load("trpo_cartpole")
    #
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()


if __name__ == '__main__':
    main()
