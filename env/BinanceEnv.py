import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime
from binance_data import BinanceReader
import math

from simulator.simulator import Simulator


class BinanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BinanceEnv, self).__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(125,), dtype=np.float16)
        self.is_simulation = True
        self.binance = BinanceReader()
        self.simulator = Simulator(simulation=self.is_simulation)
        self.current_step = 0
        self.initial_step = 0
        self.last_price = 0.0
        self.previous_price = 0.0
        self.is_testing = False
        self.start_time = datetime.now()

    def next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        asks, bids, sum_asks, sum_bids = self.simulator.get_order_book()
        recent_trades, sum_buy, sum_sell = self.simulator.get_recent_trades()
        if recent_trades.shape[0] < 20:
            recent_trades = np.append(recent_trades, np.zeros((20 - recent_trades.shape[0], 2)), axis=0)
        if asks.shape[0] < 20:
            asks = np.append(asks, np.zeros((20 - asks.shape[0], 2)), axis=0)
        if bids.shape[0] < 20:
            bids = np.append(bids, np.zeros((20 - bids.shape[0], 2)), axis=0)
        self.last_price = self.simulator.get_last_price()
        self.previous_price = self.simulator.get_previous_price()
        indicators = np.asarray(
            [sum_asks, sum_bids, ((self.last_price - self.previous_price) / self.previous_price), sum_buy, sum_sell],
            dtype=np.float)
        obs = np.append(asks, bids)
        obs = np.append(obs, recent_trades)
        obs = np.append(obs, indicators)
        if obs.shape[0] < 125:
            obs = np.append(obs, np.zeros((125 - obs.shape[0])))
        return np.asarray(obs, dtype=np.float)

    def reset(self):
        self.simulator = Simulator()
        if self.is_simulation:
            if self.is_testing:
                self.current_step = 197932
            else:
                self.current_step = random.randint(0, self.simulator.DataSimulator.max_steps - 10000)
            self.simulator.DataSimulator.current_step = self.current_step
            self.initial_step = self.current_step
        else:
            self.current_step = 0
        return self.next_observation()

    def _take_action(self, action_type):
        # Set the current price to a random price within the time step
        # print('\n action = ', action_type)
        if action_type == 0:
            volume = abs(self.simulator.get_volume())
            self.simulator.long(volume)
        elif action_type == 1:
            self.simulator.close_long(True)
        elif action_type == 2:
            self.simulator.close_long(False)
        elif action_type == 3:
            volume = abs(self.simulator.get_volume())
            self.simulator.short(volume)
        elif action_type == 4:
            self.simulator.close_short(True)
        elif action_type == 5:
            self.simulator.close_short(False)
        # self.render()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.simulator.update_account()
        print(f'Step: {self.simulator.DataSimulator.state_id}')
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
        if self.current_step - self.initial_step == 0:
            self.start_time = datetime.now()
        if self.current_step % 1000 == 0:
            print(datetime.now() - self.start_time)
            self.start_time = datetime.now()
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
