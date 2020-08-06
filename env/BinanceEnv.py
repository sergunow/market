import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

from binance_data import BinanceReader
import math

from simulator.simulator import Simulator


class BinanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BinanceEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1000, high=20000,
                                            shape=(84,), dtype=np.float16)
        self.binance = BinanceReader()
        self.simulator = Simulator()
        self.current_step = 0
        self.last_price = 0.0

    def next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        asks, bids = self.simulator.get_order_book()
        data = list()
        for item in asks:
            data.append(item[1])
        asks = np.asarray(data)
        data = list()
        for item in bids:
            data.append(item[1])
        bids = np.asarray(data)
        recent_trades = self.simulator.get_recent_trades()
        data = list()
        for item in recent_trades:
            data.append([item[1], item[2]])
        recent_trades = np.asarray(data)
        self.last_price = self.simulator.get_last_price()
        max_short, max_long = self.simulator.get_max_short_and_long()
        obs = np.append(asks, bids)
        obs = np.append(obs, recent_trades)
        obs = np.append(obs, self.simulator.profit)
        obs = np.append(obs, self.simulator.balance)
        obs = np.append(obs, np.asarray(max_long))
        obs = np.append(obs, np.asarray(max_short))
        return obs

    def reset(self):
        self.simulator = Simulator()
        self.current_step = random.randint(0, int(self.simulator.max_steps / 1.3))
        self.simulator.current_step = self.current_step
        return self.next_observation()

    def _take_action(self, action_type):
        # Set the current price to a random price within the time step
        # print('\n action = ', action_type)
        if action_type == 0:
            self.simulator.long(10)
        elif action_type == 1:
            self.simulator.close_long()
        elif action_type == 2:
            self.simulator.short(10)
        elif action_type == 3:
            self.simulator.close_short()
        # self.render()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.simulator.update_account()
        print(f'Step: {self.current_step}')
        print(f'Current price: {self.last_price}')
        print(f'Balance: {self.simulator.balance}')
        print(f'Available balance: {self.simulator.available_balance}')
        print(
            f'Total trades:: {self.simulator.count_trades}')
        print(
            f'Current profit: {self.simulator.profit} (Max profit: {self.simulator.max_profit})')

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.simulator.current_step += 1
        reward = float(self.simulator.reward)
        done = False
        if self.simulator.reward < - 100 or self.current_step + 1 == self.simulator.max_steps:
            done = True

        obs = self.next_observation()

        return obs, reward, done, {}
