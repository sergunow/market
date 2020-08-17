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
                                            shape=(29, 60, 3), dtype=np.float16)
        self.binance = BinanceReader()
        self.simulator = Simulator()
        self.current_step = 0
        self.initial_step = 0
        self.last_price = 0.0
        self.previous_price = 0.0
        self.is_testing = True

    def next_observation(self):
        asks, bids, sum_asks, sum_bids = self.simulator.get_order_book()
        recent_trades = self.simulator.get_recent_trades()
        self.last_price = self.simulator.get_last_price()
        self.previous_price = self.simulator.get_previous_price()
        # indicators = np.asarray([sum_asks, sum_bids, ((self.last_price - self.previous_price) / self.previous_price)],
        #                         dtype=np.float)
        # indicators = indicators.reshape([1] + list(indicators.shape))
        # indicators = indicators.reshape([1] + list(indicators.shape))
        obs = np.append(asks, bids, axis=0)
        obs = np.append(obs, recent_trades, axis=0)
        obs = obs.reshape((29, 60, 3))
        # obs = obs.reshape([1] + list(obs.shape))
        return np.asarray(obs, dtype=np.float)

    def reset(self):
        self.simulator = Simulator()
        if self.is_testing:
            self.current_step = 70000
        else:
            self.current_step = random.randint(0, int(self.simulator.DataSimulator.max_steps / 1.3))
        self.simulator.DataSimulator.current_step = self.current_step
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
        self.simulator.DataSimulator.current_step += 1
        reward = float(self.simulator.reward)
        done = False
        if self.simulator.profit < - 100 \
                or self.current_step + 1 == self.simulator.DataSimulator.max_steps \
                or self.current_step - self.initial_step > 10000:
            done = True

        obs = self.next_observation()

        return obs, reward, done, {}
