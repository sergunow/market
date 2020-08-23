import psycopg2
from decimal import Decimal

from binance_data import BinanceReader
import numpy as np
import pandas as pd
import math
from scipy.stats import zscore


class DataSimulator:
    def __init__(self, simulation=True, step=0):
        self.simulation = simulation
        self.state_id = 0
        self.max_steps = 0
        self.states = list()
        self.recent_trades = list()
        self.asks = list()
        self.bids = list()
        self.current_step = step
        self.binance = BinanceReader()
        self.last_price = 0.0
        self.previous_price = self.last_price
        self.get_last_price()

    def get_last_price(self):
        if self.simulation:
            if len(self.states) == 0:
                self.states = pd.read_csv('/home/sergunow/PycharmProjects/market visaul/simulator/states.csv')
                self.max_steps = len(self.states)
            self.state_id = self.states.iloc[self.current_step]['id']
            self.previous_price = self.last_price
            self.last_price = self.states.iloc[self.current_step]['price']
            if self.previous_price == 0.0:
                self.previous_price = self.last_price
            return self.last_price
        else:
            self.previous_price = self.last_price
            self.last_price = self.binance.get_last_price()
            if self.previous_price == 0:
                self.previous_price = self.last_price
            return self.last_price

    def get_previous_price(self):
        return self.previous_price

    def get_recent_trades(self):
        if self.simulation:
            if len(self.recent_trades) == 0:
                self.recent_trades = pd.read_csv(
                    '/home/sergunow/PycharmProjects/market visaul/simulator/recent_trades.csv')
                self.recent_trades[['volume']] = self.recent_trades[['volume']].apply(zscore)
            data = self.recent_trades[self.recent_trades['state_id'] == self.state_id]
            return np.asarray(data[['volume', 'type_transaction']]), data[data['type_transaction'] == 0][
                'volume'].sum(), data[data['type_transaction'] == 1]['volume'].sum()
        else:
            if len(self.recent_trades) == 0:
                self.recent_trades = pd.read_csv(
                    '/home/sergunow/PycharmProjects/market visaul/simulator/recent_trades.csv')
            mean_price = np.asarray(self.recent_trades['price']).mean()
            std_price = np.asarray(self.recent_trades['price']).std()
            # mean_volume = np.asarray(self.recent_trades['volume']).mean()
            # std_volume = np.asarray(self.recent_trades['volume']).std()
            data = self.binance.get_recent_trades()
            data['price'] = data['price'].apply(lambda x: (x - mean_price) / std_price)
            # data['volume'] = data['volume'].apply(lambda x: (x - mean_volume) / std_volume)
            return np.asarray(data[['volume', 'type_transaction']]), data[data['type_transaction'] == 0][
                'volume'].sum(), data[data['type_transaction'] == 1]['volume'].sum()

    def get_order_book(self):
        if self.simulation:
            if len(self.asks) == 0:
                self.asks = pd.read_csv('/home/sergunow/PycharmProjects/market visaul/simulator/asks.csv')
                self.asks[['volume']] = self.asks[['volume']].apply(zscore)
                self.asks['type'] = 0
            if len(self.bids) == 0:
                self.bids = pd.read_csv('/home/sergunow/PycharmProjects/market visaul/simulator/bids.csv')
                self.bids[['volume']] = self.bids[['volume']].apply(zscore)
                self.bids['type'] = 1
            return np.asarray(self.asks[self.asks['state_id'] == self.state_id][['volume', 'type']]), \
                   np.asarray(self.bids[self.bids['state_id'] == self.state_id][['volume', 'type']]), \
                   self.asks[self.asks['state_id'] == self.state_id]['volume'].sum(), \
                   self.bids[self.bids['state_id'] == self.state_id]['volume'].sum()
        else:
            if len(self.asks) == 0:
                self.asks = pd.read_csv('/home/sergunow/PycharmProjects/market visaul/simulator/asks.csv')
                self.asks['type'] = 0
            if len(self.bids) == 0:
                self.bids = pd.read_csv('/home/sergunow/PycharmProjects/market visaul/simulator/bids.csv')
                self.bids['type'] = 1
            asks_mean_price = np.asarray(self.asks['price']).mean()
            asks_std_price = np.asarray(self.asks['price']).std()
            asks_mean_volume = np.asarray(self.asks['volume']).mean()
            asks_std_volume = np.asarray(self.asks['volume']).std()
            bids_mean_price = np.asarray(self.bids['price']).mean()
            bids_std_price = np.asarray(self.bids['price']).std()
            bids_mean_volume = np.asarray(self.bids['volume']).mean()
            bids_std_volume = np.asarray(self.bids['volume']).std()
            asks, bids = self.binance.get_order_book()
            asks['price'] = asks['price'].apply(lambda x: (x - asks_mean_price) / asks_std_price)
            asks['volume'] = asks['volume'].apply(lambda x: (x - asks_mean_volume) / asks_std_volume)
            bids['price'] = bids['price'].apply(lambda x: (x - bids_mean_price) / bids_std_price)
            bids['volume'] = bids['volume'].apply(lambda x: (x - bids_mean_volume) / bids_std_volume)
            return np.asarray(asks), np.asarray(bids), asks['volume'].sum(), bids['volume'].sum()


class Simulator:
    def __init__(self, simulation=True, step=0):
        self.DataSimulator = DataSimulator(simulation=simulation, step=step)
        self.history = pd.DataFrame(
            columns=['type', 'open_price', 'close_price', 'volume', 'profit', 'max_profit', 'max_down'])
        self.active_trades = pd.DataFrame(
            columns=['type', 'open_price', 'volume', 'profit', 'max_profit', 'max_down'])
        self.balance = 10000
        self.available_balance = self.balance
        self.initial_balance = self.balance
        self.profit = 0.0
        self.count_trades = 0
        self.max_profit = 0.0
        self.max_down = 0.0
        self.reward = 0.0
        self.current_price = 0.0
        self.commission = 0.00075
        self.simulation = simulation
        self.current_step = step
        self.state_id = 0
        self.max_steps = 0
        self.states = list()
        self.recent_trades = list()
        self.asks = list()
        self.bids = list()
        self.reset()
        self.binance = BinanceReader()
        self.get_last_price()
        self.history_id = 0

    def reset(self):
        self.balance = 10000
        self.available_balance = 10000
        self.profit = 0
        self.count_trades = 0
        self.max_down = 0
        self.max_profit = 0
        self.reward = 0.0

    def add_trade_to_history(self, type, open_price, close_price, volume, profit, max_profit, max_down):
        self.history = self.history.append(
            {'type': str(type), 'open_price': open_price, 'close_price': close_price, 'volume': volume,
             'profit': profit, 'max_profit': max_profit,
             'max_down': max_down}, ignore_index=True)

    def get_history(self):
        return self.history.tail(20)

    def get_award(self, profit):
        history = self.get_history()
        if len(history) > 0:
            count_positive = 0
            profit = 0.0
            count_negative = 0
            loss = 0.0
            for i in range(len(history)):
                if history.iloc[i]['profit'] > 0:
                    count_positive += 1
                    profit += history.iloc[i]['profit']
                else:
                    count_negative += 1
                    loss += history.iloc[i]['profit']
            self.reward = (count_positive / len(history)) * profit + (count_negative / len(history)) * loss

    def update_transactions(self, active_trades, last_price):
        for i in range(len(active_trades)):
            if active_trades.iloc[i]['type'] == 'long':
                profit = (last_price - active_trades.iloc[i]['open_price']) * active_trades.iloc[i]['volume']
                max_profit = active_trades.iloc[i]['max_profit']
                max_down = active_trades.iloc[i]['max_down']
                if profit > max_profit:
                    max_profit = profit
                if profit < max_down:
                    max_down = profit
                self.active_trades.iloc[i, self.active_trades.columns.get_loc('profit')] = profit
                self.active_trades.iloc[i, self.active_trades.columns.get_loc('max_profit')] = max_profit
                self.active_trades.iloc[i, self.active_trades.columns.get_loc('max_down')] = max_down
            if active_trades.iloc[i]['type'] == 'short':
                max_profit = active_trades.iloc[i]['max_profit']
                max_down = active_trades.iloc[i]['max_down']
                profit = - (last_price - active_trades.iloc[i]['open_price']) * active_trades.iloc[i]['volume']
                if profit > max_profit:
                    max_profit = profit
                if profit < max_down:
                    max_down = profit
                self.active_trades.iloc[i, self.active_trades.columns.get_loc('profit')] = profit
                self.active_trades.iloc[i, self.active_trades.columns.get_loc('max_profit')] = max_profit
                self.active_trades.iloc[i, self.active_trades.columns.get_loc('max_down')] = max_down

    def update_account(self):
        active_trades = self.get_active_trades()
        last_price = self.get_last_price()
        self.update_transactions(active_trades, last_price)
        all_long_cost = 0.0
        all_short_cost = 0.0
        for i in range(len(active_trades)):
            if active_trades.iloc[i]['type'] == 'long':
                all_long_cost = all_long_cost + active_trades.iloc[i]['open_price'] * active_trades.iloc[i]['volume'] \
                                + (last_price - active_trades.iloc[i]['open_price']) * active_trades.iloc[i]['volume']
            if active_trades.iloc[i]['type'] == 'short':
                all_short_cost = all_short_cost + active_trades.iloc[i]['open_price'] * active_trades.iloc[i]['volume'] \
                                 + (active_trades.iloc[i]['open_price'] - last_price) * active_trades.iloc[i]['volume']
        self.balance = self.available_balance + all_short_cost + all_long_cost
        self.profit = self.balance - self.initial_balance
        if self.profit > self.max_profit:
            self.max_profit = self.profit
        if self.profit < self.max_down:
            self.max_down = self.profit

    def get_active_trades(self, profit_sort=True):
        if profit_sort:
            return self.active_trades.sort_values(by='profit', ascending=False)
        else:
            return self.active_trades.sort_values(by='profit', ascending=True)

    def get_last_price(self):
        return self.DataSimulator.get_last_price()

    def get_previous_price(self):
        return self.DataSimulator.get_previous_price()

    def get_recent_trades(self):
        return self.DataSimulator.get_recent_trades()

    def get_order_book(self):
        return self.DataSimulator.get_order_book()

    def long(self, volume):
        self.update_account()
        last_price = self.get_last_price()
        trade_cost = (last_price * volume) + (last_price * volume) * self.commission
        if self.available_balance >= trade_cost:
            self.active_trades = self.active_trades.append(
                {'type': 'long', 'open_price': last_price, 'volume': volume, 'profit': 0.0, 'max_profit': 0.0,
                 'max_down': 0.0}, ignore_index=True)
            self.available_balance -= trade_cost
            self.count_trades += 1
            self.update_account()

    def close_long(self, profit_sort=True):
        self.update_account()
        last_price = self.get_last_price()
        active_trades = self.get_active_trades(profit_sort)
        long_cost = 0.0
        long_positions = list()
        for i in range(len(active_trades)):
            if active_trades.iloc[i]['type'] == 'long':
                long_positions.append(active_trades.iloc[i])
        if len(long_positions) != 0:
            close_trade = long_positions[0]
            long_cost = long_cost + close_trade['open_price'] * close_trade['volume'] + (
                    last_price - close_trade['open_price']) * close_trade['volume']
            profit = (last_price - close_trade['open_price']) * close_trade['volume']
            commission = (last_price * close_trade['volume']) * self.commission
            commission_open_trade = (close_trade['open_price'] * close_trade['volume']) * self.commission
            self.available_balance += long_cost - commission
            self.active_trades = self.active_trades.drop(close_trade.name)
            self.count_trades += 1
            self.update_account()
            self.add_trade_to_history('long', close_trade['open_price'], last_price, close_trade['volume'],
                                      profit - (commission + commission_open_trade),
                                      close_trade['profit'] - (commission + commission_open_trade),
                                      close_trade['max_profit'] - (commission + commission_open_trade))
            self.get_award(profit - (commission + commission_open_trade))

    def close_short(self, profit=True):
        self.update_account()
        last_price = self.get_last_price()
        active_trades = self.get_active_trades(profit)
        short_cost = 0.0
        short_positions = list()
        for i in range(len(active_trades)):
            if active_trades.iloc[i]['type'] == 'short':
                short_positions.append(active_trades.iloc[i])
        if len(short_positions) != 0:
            close_trade = short_positions[0]
            short_cost = short_cost + close_trade['open_price'] * close_trade['volume'] + (
                    close_trade['open_price'] - last_price) * close_trade['volume']
            profit = -(last_price - close_trade['open_price']) * close_trade['volume']
            commission = (last_price * close_trade['volume']) * self.commission
            commission_open_trade = (close_trade['open_price'] * close_trade['volume']) * self.commission
            self.available_balance += short_cost - commission
            self.active_trades = self.active_trades.drop(close_trade.name)
            self.count_trades += 1
            self.update_account()
            self.add_trade_to_history('short', close_trade['open_price'], last_price, close_trade['volume'],
                                      profit - (commission + commission_open_trade),
                                      close_trade['profit'] - (commission + commission_open_trade),
                                      close_trade['max_profit'] - (commission + commission_open_trade))
            self.get_award(profit - (commission + commission_open_trade))

    def short(self, volume):
        self.update_account()
        last_price = self.get_last_price()
        trade_cost = (last_price * volume) + (last_price * volume) * self.commission
        if self.available_balance >= trade_cost:
            self.active_trades = self.active_trades.append(
                {'type': 'short', 'open_price': last_price, 'volume': volume, 'profit': 0.0, 'max_profit': 0.0,
                 'max_down': 0.0}, ignore_index=True)
            self.available_balance -= trade_cost
            self.count_trades += 1
            self.update_account()

    def get_twr(self, f):
        history = self.get_history()
        twr = 0.0
        if len(history) > 20:
            twr = 1.0
            max_down = history.sort_values(by='max_down', ascending=False).iloc[0]['max_down']
            for i in range(len(history)):
                twr = twr * (1 + f * (-history.iloc[i]['profit']) / max_down)
        return twr

    def get_optimal_f(self):
        twr = 0.0
        optimal_f = 0.0
        for i in np.arange(0.05, 1, 0.05):
            step_twr = self.get_twr(i)
            if step_twr > twr:
                twr = step_twr
                optimal_f = i
        return optimal_f

    def get_volume(self):
        f = self.get_optimal_f()
        history = self.get_history()
        if f != 0.0 and len(history) > 20:
            max_down = history.sort_values(by='max_down', ascending=False).iloc[0]['max_down']
            return float(float(self.balance) / (max_down / (- 1 * f))) / 1000.00
        else:
            return 1
