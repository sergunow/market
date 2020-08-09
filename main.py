import time
import random
import gym
import json
import datetime as dt

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

import random
import numpy as np
from keras.layers import Dense, Flatten, LSTM, Dropout, Input, ConvLSTM2D, BatchNormalization, MaxPooling1D, \
    TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

from env.BinanceEnv import BinanceEnv


def agent(states, actions):
    model = Sequential()
    # model.add(Flatten(input_shape=(1, states)))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(actions, activation='linear'))
    # print(model.summary())

    # model.add(LSTM(units=512, return_sequences=True, input_shape=(1, states)))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=512, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=512, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=512))
    # model.add(Dropout(0.2))
    # model.add(Dense(actions, activation='relu'))
    model.add(Input(shape=(None, 60, 3, 1)))
    model.add(ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(
        filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
    ))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(
        filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
    ))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(
        filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
    ))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    print(model.summary())

    return model


def main():
    # binance = DataReader()
    env = BinanceEnv()
    # binance.get_recent_trades()
    # env.next_observation()
    # binance_market = BinanceMarket()
    # binance_market.long()
    # time.sleep(3)
    # binance_market.close_long()
    # time.sleep(3)
    # binance_market.short()
    # time.sleep(3)
    # binance_market.close_short()
    # binance_market.update_positions()
    # print(binance_market.balance)

    # episodes = 10
    # for episode in range(1, episodes + 1):
    #     # At each begining reset the game
    #     state = env.reset()
    #     # set done to False
    #     done = False
    #     # set score to 0
    #     score = 0
    #     # while the game is not finished
    #     while not done:
    #         # visualize each step
    #         env.render()
    #         # choose a random action
    #         action = random.randint(0, 5)
    #         # execute the action
    #         n_state, reward, done, info = env.step(action)
    #         # keep track of rewards
    #         score += reward
    #     print('episode {} score {}'.format(episode, score))

    model = agent(env.observation_space.shape, env.action_space.n)
    policy = EpsGreedyQPolicy()
    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=env.action_space.n)
    sarsa.compile(optimizer="adadelta", metrics=['mse', 'accuracy'])
    # sarsa.load_weights('sarsa_weights_bnb_07.h5f')
    env.is_testing = False
    sarsa.fit(env, nb_steps=100000, visualize=False, verbose=1)
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


if __name__ == '__main__':
    main()
