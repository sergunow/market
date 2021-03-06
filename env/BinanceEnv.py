import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from decimal import Decimal

from binance_data import BinanceReader
import math

from simulator.simulator import Simulator


class BinanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BinanceEnv, self).__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(65,), dtype=np.float16)
        self.binance = BinanceReader()
        self.simulator = Simulator()
        self.current_step = 0
        self.initial_step = 0
        self.last_price = 0.0
        self.is_testing = True

    def next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        # asks, bids = self.simulator.get_order_book()
        # data = list()
        # for item in asks:
        #     data.append(item[1])
        # asks = np.asarray(data)
        # data = list()
        # for item in bids:
        #     data.append(item[1])
        # bids = np.asarray(data)
        recent_trades = self.simulator.get_recent_trades()
        # data = list()
        # for item in recent_trades:
        #     data.append([item[1], item[2]])
        # recent_trades = np.asarray(data)
        self.last_price = self.simulator.get_last_price()
        max_short, max_long = self.simulator.get_max_short_and_long()
        obs = np.append(recent_trades, self.simulator.profit)
        # obs = np.append(obs, recent_trades)
        # obs = np.append(obs, self.simulator.profit)
        obs = np.append(obs, self.simulator.balance)
        obs = np.append(obs, self.last_price)
        obs = np.append(obs, np.asarray(max_long))
        obs = np.append(obs, np.asarray(max_short))
        return np.asarray(obs, dtype=np.float)

    def reset(self):
        self.simulator = Simulator()
        if self.is_testing:
            self.current_step = 70000
        else:
            self.current_step = random.randint(0, int(self.simulator.max_steps / 1.3))
        self.simulator.current_step = self.current_step
        self.initial_step = self.current_step
        return self.next_observation()

    def _take_action(self, action_type):
        # Set the current price to a random price within the time step
        # print('\n action = ', action_type)
        if action_type == 0:
            volume = Decimal(abs(self.simulator.get_volume()))
            self.simulator.long(volume)
        elif action_type == 1:
            self.simulator.close_long(True)
        elif action_type == 2:
            self.simulator.close_long(False)
        elif action_type == 3:
            volume = Decimal(abs(self.simulator.get_volume()))
            self.simulator.short(volume)
        elif action_type == 4:
            self.simulator.close_short(True)
        elif action_type == 5:
            self.simulator.close_short(False)
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
        if self.simulator.profit < - 100 \
                or self.current_step + 1 == self.simulator.max_steps \
                or self.current_step - self.initial_step > 10000:
            done = True

        obs = self.next_observation()

        return obs, reward, done, {}
