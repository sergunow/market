import psycopg2
from decimal import Decimal

from binance_data import BinanceReader
import numpy as np
import pandas as pd


class Simulator:
    def __init__(self, simulation=True, step=0):
        self.balance = 10000
        self.available_balance = self.balance
        self.initial_balance = self.balance
        self.profit = 0.0
        self.count_trades = 0
        self.max_profit = 0.0
        self.max_down = 0.0
        self.reward = 0.0
        self.current_price = 0.0
        self.commission = Decimal(0.00075)
        self.simulation = simulation
        self.current_step = step
        self.state_id = 0
        self.max_steps = 0
        self.states = list()
        self.recent_trades = list()
        self.asks = list()
        self.bids = list()
        self._connect()
        self.reset()
        self.binance = BinanceReader()
        self.get_last_price()
        self.get_account_info()
        self.history_id = 0

    def _connect(self):
        self.con = psycopg2.connect(
            database="orderbook",
            user="admin",
            password="l82Z01vdQl",
            host="127.0.0.1",
            port="5432"
        )

    def reset(self):
        cur = self.con.cursor()
        sql = 'update accounts set balance = 10000, free_balance = 10000, profit = 0, count_trades = 0, max_down = 0, ' \
              'max_profit = 0 where id = 1; '
        cur.execute("ROLLBACK")
        cur.execute(sql)
        sql = 'TRUNCATE history;'
        cur.execute("ROLLBACK")
        cur.execute(sql)
        sql = 'TRUNCATE active_trades;'
        cur.execute("ROLLBACK")
        cur.execute(sql)
        self.reward = 0.0

    def add_trade_to_history(self, type, open_price, close_price, volume, profit, max_profit, max_down):
        type = "'" + str(type) + "'"
        cur = self.con.cursor()
        sql = 'insert into history (type, open_price, close_price, volume, profit, max_profit, max_down) ' \
              'VALUES (' + str(type) + ', ' + str(open_price) + ', ' + str(close_price) + ', ' + str(volume) \
              + ', ' + str(profit) + ', ' + str(max_profit) + ', ' + str(max_down) + ');'
        cur.execute("ROLLBACK")
        cur.execute(sql)

    def get_history(self):
        cur = self.con.cursor()
        sql = 'select id, profit, max_profit, max_down from history order by id;'
        cur.execute("ROLLBACK")
        cur.execute(sql)
        return cur.fetchall()

    def get_award(self, profit):
        # if profit <= 0:
        #     self.reward -= abs(float(profit))
        # else:
        #     self.reward += float(profit)
        # sum_profit = 0.0
        # active_trades = self.get_active_trades()
        # if len(active_trades) > 0:
        #     for item in active_trades:
        #         sum_profit += float(item[4])
        #     self.reward += sum_profit / len(active_trades)
        history = self.get_history()
        if len(history) > 0:
            if history[0][0] != self.history_id:
                if float(history[0][2]) != 0:
                    if float(history[0][1]) > 0:
                        self.reward += float(history[0][1]) / float(history[0][2])
                    else:
                        self.reward -= float(history[0][1]) / float(history[0][2])
                    self.history_id = history[0][0]

    def get_account_info(self):
        cur = self.con.cursor()
        sql = 'select * from accounts where id = 1;'
        cur.execute(sql)
        data = cur.fetchall()[0]
        self.balance = data[1]
        self.available_balance = data[2]
        self.profit = data[3]
        self.count_trades = data[4]
        self.max_down = data[5]
        self.max_profit = data[6]

    def update_transactions(self, active_trades, last_price):
        for item in active_trades:
            if item[1] == 'long':
                profit = (last_price - item[2]) * item[3]
                max_profit = item[5]
                max_down = item[6]
                if profit > max_profit:
                    max_profit = profit
                if profit < max_down:
                    max_down = profit
                # if profit < -2:
                #     self.close_long()
                cur = self.con.cursor()
                sql = 'update active_trades set profit = {0}, max_profit = {1}, max_down = {2} where id = {3};'. \
                    format(profit, max_profit, max_down, item[0])
                cur.execute(sql)
            if item[1] == 'short':
                max_profit = item[5]
                max_down = item[6]
                profit = - (last_price - item[2]) * item[3]
                if profit > max_profit:
                    max_profit = profit
                if profit < max_down:
                    max_down = profit
                # if profit < -2:
                #     self.close_short()
                cur = self.con.cursor()
                sql = 'update active_trades set profit = {0}, max_profit = {1}, max_down = {2} where id = {3};' \
                    .format(profit, max_profit, max_down, item[0])
                cur.execute(sql)

    def update_account(self):
        active_trades = self.get_active_trades()
        last_price = self.get_last_price()
        self.update_transactions(active_trades, last_price)
        all_long_cost = Decimal(0.0)
        all_short_cost = Decimal(0.0)
        for item in active_trades:
            if item[1] == 'long':
                all_long_cost = all_long_cost + item[2] * item[3] + (
                        last_price - item[2]) * item[3]
            if item[1] == 'short':
                all_short_cost = all_short_cost + item[2] * item[3] + (
                        item[2] - last_price) * item[3]
        self.balance = self.available_balance + all_short_cost + all_long_cost
        self.profit = self.balance - self.initial_balance

        cur = self.con.cursor()
        sql = 'update accounts set balance = {0}, free_balance = {1}, profit = {2}, count_trades = {3}, ' \
              'max_down = {4}, max_profit = {5} where id = 1;'.format(
            self.balance, self.available_balance, self.profit, self.count_trades, self.max_down, self.max_profit)
        cur.execute(sql)

    def get_active_trades(self, profit=True):
        sort = ''
        if profit:
            sort = 'desc'
        cur = self.con.cursor()
        sql = 'select * from active_trades order by profit {0};'.format(sort)
        cur.execute(sql)
        return cur.fetchall()

    def get_max_short_and_long(self):
        max_long = 0.0
        max_short = 0.0

        cur = self.con.cursor()
        sql = 'select * from active_trades where type = {0} order by profit desc;'.format("'long'")
        cur.execute(sql)
        longs = cur.fetchall()
        if len(longs) > 0:
            max_long = longs[0][4]
        cur = self.con.cursor()
        sql = 'select * from active_trades where type = {0} order by profit desc;'.format("'short'")
        cur.execute(sql)
        shorts = cur.fetchall()
        if len(shorts) > 0:
            max_short = shorts[0][4]
        return max_long, max_long

    def get_last_price(self):
        if self.simulation:
            if len(self.states) == 0:
                cur = self.con.cursor()
                sql = 'select * from states;'
                cur.execute("ROLLBACK")
                cur.execute(sql)
                self.states = cur.fetchall()
                self.max_steps = len(self.states)
            self.state_id = self.states[self.current_step][0]
            return self.states[self.current_step][2]
        else:
            return self.binance.get_last_price()

    def get_recent_trades(self):
        if self.simulation:
            if len(self.recent_trades) == 0:
                cur = self.con.cursor()
                sql = 'select price, volume, type_transaction, state_id from time_and_sales;'
                cur.execute("ROLLBACK")
                cur.execute(sql)
                data = np.asarray(cur.fetchall())
                dict = {}
                i = 0
                for item in data:
                    dict[i] = {
                        'price': item[0],
                        'volume': item[1],
                        'isBuyerMaker': item[2],
                        'state_id': item[3]
                    }
                    i += 1
                self.recent_trades = pd.DataFrame.from_dict(dict, 'index')
            return np.asarray(self.recent_trades[self.recent_trades['state_id'] == self.state_id][
                                  ['price', 'volume', 'isBuyerMaker']])
        else:
            return np.asarray(self.binance.get_recent_trades())

    def get_order_book(self):
        if self.simulation:
            if len(self.asks) == 0:
                cur = self.con.cursor()
                sql = 'select price, volume, state_id from asks;'
                cur.execute("ROLLBACK")
                cur.execute(sql)
                data = cur.fetchall()
                dict = {}
                i = 0
                for item in data:
                    dict[i] = {
                        'price': item[0],
                        'volume': item[1],
                        'state_id': item[2]
                    }
                    i += 1
                self.asks = pd.DataFrame.from_dict(dict, 'index')
            if len(self.bids) == 0:
                cur = self.con.cursor()
                sql = 'select price, volume, state_id from bids;'
                cur.execute("ROLLBACK")
                cur.execute(sql)
                data = cur.fetchall()
                dict = {}
                i = 0
                for item in data:
                    dict[i] = {
                        'price': item[0],
                        'volume': item[1],
                        'state_id': item[2]
                    }
                    i += 1
                self.bids = pd.DataFrame.from_dict(dict, 'index')
            return np.asarray(self.asks[self.asks['state_id'] == self.state_id][['price', 'volume']]), \
                   np.asarray(self.bids[self.bids['state_id'] == self.state_id][['price', 'volume']])
        else:
            return np.asarray(self.binance.get_order_book())

    def long(self, volume):
        self.get_account_info()
        last_price = self.get_last_price()
        trade_cost = (last_price * volume) + (last_price * volume) * self.commission
        transaction_type = "'long'"
        if self.available_balance >= trade_cost:
            cur = self.con.cursor()
            sql = 'insert into active_trades (type, open_price, volume, profit, max_profit, max_down) ' \
                  'VALUES ({0}, {1}, {2}, {3}, {4}, {5});'.format(transaction_type, last_price, volume, 0.0, 0.0, 0.0)
            cur.execute("ROLLBACK")
            cur.execute(sql)

            self.available_balance -= trade_cost
            self.count_trades += 1
            self.update_account()

    def close_long(self, profit=True):
        self.get_account_info()
        last_price = self.get_last_price()
        active_trades = self.get_active_trades(profit)
        long_cost = Decimal(0.0)
        long_positions = list()
        for item in active_trades:
            if item[1] == 'long':
                long_positions.append(item)
        if len(long_positions) != 0:
            close_trade = long_positions[0]
            long_cost = long_cost + close_trade[2] * close_trade[3] + (
                    last_price - close_trade[2]) * close_trade[3]
            profit = (last_price - close_trade[2]) * close_trade[3]
            commission = (last_price * close_trade[3]) * self.commission
            commission_open_trade = (close_trade[2] * close_trade[3]) * self.commission
            self.available_balance += long_cost - commission
            cur = self.con.cursor()
            sql = 'delete from active_trades where id = {0};'.format(close_trade[0])
            cur.execute("ROLLBACK")
            cur.execute(sql)
            self.count_trades += 1
            self.update_account()
            if close_trade[5] == 0:
                a = 1
            self.add_trade_to_history('long', close_trade[2], last_price, close_trade[3],
                                      profit - (commission + commission_open_trade),
                                      close_trade[5] - (commission + commission_open_trade),
                                      close_trade[6] - (commission + commission_open_trade))
            self.get_award(profit - (commission + commission_open_trade))

    def close_short(self, profit=True):
        self.get_account_info()
        last_price = self.get_last_price()
        active_trades = self.get_active_trades(profit)
        short_cost = Decimal(0.0)
        short_positions = list()
        for item in active_trades:
            if item[1] == 'short':
                short_positions.append(item)
        if len(short_positions) != 0:
            close_trade = short_positions[0]
            short_cost = short_cost + close_trade[2] * close_trade[3] + (
                    close_trade[2] - last_price) * close_trade[3]
            profit = -(last_price - close_trade[2]) * close_trade[3]
            commission = (last_price * close_trade[3]) * self.commission
            commission_open_trade = (close_trade[2] * close_trade[3]) * self.commission
            self.available_balance += short_cost - commission
            cur = self.con.cursor()
            sql = 'delete from active_trades where id = {0};'.format(close_trade[0])
            cur.execute("ROLLBACK")
            cur.execute(sql)
            self.count_trades += 1
            self.update_account()
            self.add_trade_to_history('short', close_trade[2], last_price, close_trade[3],
                                      profit - (commission + commission_open_trade),
                                      close_trade[5] - (commission + commission_open_trade),
                                      close_trade[6] - (commission + commission_open_trade))
            self.get_award(profit - (commission + commission_open_trade))

    def short(self, volume):
        self.get_account_info()
        last_price = self.get_last_price()
        trade_cost = (last_price * volume) + (last_price * volume) * self.commission
        transaction_type = "'short'"
        if self.available_balance >= trade_cost:
            cur = self.con.cursor()
            sql = 'insert into active_trades (type, open_price, volume, profit, max_profit, max_down) ' \
                  'VALUES ({0}, {1}, {2}, {3}, {4}, {5});'.format(transaction_type, last_price, volume, 0.0, 0.0, 0.0)
            cur.execute("ROLLBACK")
            cur.execute(sql)

            self.available_balance -= trade_cost
            self.count_trades += 1
            self.update_account()

    def get_twr(self, f):
        history = self.get_history()
        twr = 0.0
        if len(history) > 20:
            twr = 1.0
            max_down = 0.0
            for item in history:
                if float(item[1]) < max_down:
                    max_down = float(item[1])
            for item in history:
                twr = twr * (1 + f * (float(-item[1])) / max_down)
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

    def get_max_down_trade(self):
        cur = self.con.cursor()
        sql = 'select id, profit, max_profit, max_down from history order by profit limit 1;'
        cur.execute("ROLLBACK")
        cur.execute(sql)
        return cur.fetchall()

    def get_volume(self):
        f = self.get_optimal_f()
        history = self.get_history()
        if f != 0.0 and len(history) > 20:
            max_down = float(self.get_max_down_trade()[0][2])
            return float(float(self.balance) / (max_down / (- 1 * f))) / 1000.00
        else:
            return 10
