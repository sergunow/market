import psycopg2
from decimal import Decimal

from binance_data import BinanceReader
import numpy as np


class Simulator:
    def __init__(self, simulation=False, step=0):
        self.balance = 10000
        self.available_balance = self.balance
        self.initial_balance = self.balance
        self.profit = 0.0
        self.count_trades = 0
        self.max_profit = 0.0
        self.max_down = 0.0
        self.current_price = 0.0
        self.commission = Decimal(0.00075)
        self.simulation = simulation
        self.current_step = step
        self.state_id = 0
        self.max_steps = 0
        self._connect()
        self.reset()
        self.binance = BinanceReader()
        self.get_last_price()
        self.get_account_info()

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

    def add_trade_to_history(self, type, open_price, close_price, volume, profit):
        type = "'" + str(type) + "'"
        cur = self.con.cursor()
        sql = 'insert into history (type, open_price, close_price, volume, profit) VALUES (' + str(
            type) + ', ' + str(
            open_price) + ', ' + str(close_price) + ', ' + str(volume) + ', ' + str(profit) + ');'
        cur.execute("ROLLBACK")
        cur.execute(sql)

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
                cur = self.con.cursor()
                sql = 'update active_trades set profit = {0} where id = {1};'.format(profit, item[0])
                cur.execute(sql)
            if item[1] == 'short':
                profit = - (last_price - item[2]) * item[3]
                cur = self.con.cursor()
                sql = 'update active_trades set profit = {0} where id = {1};'.format(profit, item[0])
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

    def get_active_trades(self):
        cur = self.con.cursor()
        sql = 'select * from active_trades order by id;'
        cur.execute(sql)
        return cur.fetchall()

    def get_last_price(self):
        if self.simulation:
            cur = self.con.cursor()
            sql = 'select * from states'
            cur.execute("ROLLBACK")
            cur.execute(sql)
            states = cur.fetchall()
            self.max_steps = len(states)
            self.state_id = states[self.current_step][0]
            return states[self.current_step][2]
        else:
            return self.binance.get_last_price()

    def get_recent_trades(self):
        if self.simulation:
            cur = self.con.cursor()
            sql = 'select price, volume, type_transaction from time_and_sales where state_id = {0};'.format(self.state_id)
            cur.execute("ROLLBACK")
            cur.execute(sql)
            return np.asarray(cur.fetchall())
        else:
            return np.asarray(self.binance.get_recent_trades())

    def get_order_book(self):
        if self.simulation:
            cur = self.con.cursor()
            sql = 'select price, volume from asks where state_id = {0};'.format(self.state_id)
            cur.execute("ROLLBACK")
            cur.execute(sql)
            asks = cur.fetchall()

            cur = self.con.cursor()
            sql = 'select price, volume from bids where state_id = {0};'.format(self.state_id)
            cur.execute("ROLLBACK")
            cur.execute(sql)
            bids = cur.fetchall()

            return np.asarray(asks), np.asarray(bids)
        else:
            return np.asarray(self.binance.get_order_book())

    def long(self, volume):
        self.get_account_info()
        last_price = self.get_last_price()
        trade_cost = (last_price * volume) + (last_price * volume) * self.commission
        transaction_type = "'long'"
        if self.available_balance >= trade_cost:
            cur = self.con.cursor()
            sql = 'insert into active_trades (type, open_price, volume, profit) VALUES ({0}, {1}, {2}, {3});'.format(
                transaction_type, last_price, volume, 0.0)
            cur.execute("ROLLBACK")
            cur.execute(sql)

            self.available_balance -= trade_cost
            self.count_trades += 1
            self.update_account()

    def close_long(self):
        self.get_account_info()
        last_price = self.get_last_price()
        active_trades = self.get_active_trades()
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
            self.available_balance += long_cost - (last_price * close_trade[3]) * self.commission
            cur = self.con.cursor()
            sql = 'delete from active_trades where id = {0};'.format(close_trade[0])
            cur.execute("ROLLBACK")
            cur.execute(sql)
            self.count_trades += 1
            self.update_account()
            self.add_trade_to_history('long', close_trade[2], last_price, close_trade[3], profit)

    def close_short(self):
        self.get_account_info()
        last_price = self.get_last_price()
        active_trades = self.get_active_trades()
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
            self.available_balance += short_cost - (last_price * close_trade[3]) * self.commission
            cur = self.con.cursor()
            sql = 'delete from active_trades where id = {0};'.format(close_trade[0])
            cur.execute("ROLLBACK")
            cur.execute(sql)
            self.count_trades += 1
            self.update_account()
            self.add_trade_to_history('short', close_trade[2], last_price, close_trade[3], profit)

    def short(self, volume):
        self.get_account_info()
        last_price = self.get_last_price()
        trade_cost = (last_price * volume) + (last_price * volume) * self.commission
        transaction_type = "'short'"
        if self.available_balance >= trade_cost:
            cur = self.con.cursor()
            sql = 'insert into active_trades (type, open_price, volume, profit) VALUES ({0}, {1}, {2}, {3});'.format(
                transaction_type, last_price, volume, 0.0)
            cur.execute("ROLLBACK")
            cur.execute(sql)

            self.available_balance -= trade_cost
            self.count_trades += 1
            self.update_account()


if __name__ == '__main__':
    simulator = Simulator()
    simulator.current_step = 10
    simulator.get_last_price()
    a, c = simulator.get_order_book()
    b = simulator.get_recent_trades()
    a = np.asarray(a).flatten()
    i = 0
