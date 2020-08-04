import psycopg2
from binance.client import Client
import time
from tqdm import tqdm


class Record:

    def __init__(self):
        self.con = psycopg2.connect(
            database="orderbook",
            user="admin",
            password="l82Z01vdQl",
            host="127.0.0.1",
            port="5432"
        )
        try:
            self.client = Client('7PN2Xro2pKvURLggEqvvs7lBthYhhHEExe8bYAQgAk8ERIZRDrhVH7YYSNr9BJae',
                                 'RkNjG8TzScHs9VwP7bM2LqeW4gYwJlpb8i7wpdVGG4klUnbHjeHs5BhW4CifouU1')
        except Exception:
            time.sleep(10)
            self.client = Client('7PN2Xro2pKvURLggEqvvs7lBthYhhHEExe8bYAQgAk8ERIZRDrhVH7YYSNr9BJae',
                                 'RkNjG8TzScHs9VwP7bM2LqeW4gYwJlpb8i7wpdVGG4klUnbHjeHs5BhW4CifouU1')

    def get_order_book(self):
        data = {}
        while len(data) == 0:
            try:
                data = self.client.get_order_book(symbol='BNBUSDT', limit=20)
            except Exception:
                continue
            return data

    def get_recent_trades(self):
        data = {}
        while len(data) == 0:
            try:
                data = self.client.get_recent_trades(symbol='BNBUSDT', limit=20)
            except Exception:
                continue
            return data

    def get_last_price(self):
        data = {}
        while len(data) == 0:
            try:
                data = self.client.get_ticker(symbol='BNBUSDT')
            except Exception:
                continue
            return float(data['lastPrice'])

    def save_data(self):
        last_price = self.get_last_price()
        recent_trades = self.get_recent_trades()
        order_book = self.get_order_book()

        cur = self.con.cursor()
        sql = 'insert into states (time, price) values (now(), {0});'.format(last_price)
        cur.execute("ROLLBACK")
        cur.execute(sql)

        sql = 'select max(id) from states;'
        cur.execute(sql)
        id = cur.fetchall()[0][0]

        for trades in recent_trades:
            sql = 'insert into time_and_sales (price, volume, type_transaction, state_id) VALUES ({0}, {1}, {2}, {3} );'.format(
                trades['price'], trades['qty'], int(trades['isBuyerMaker']), id)
            cur.execute(sql)

        for asks in order_book['asks']:
            sql = 'insert into asks (price, volume, state_id) VALUES ({0}, {1}, {2});'.format(asks[0], asks[1], id)
            cur.execute(sql)

        for bids in order_book['bids']:
            sql = 'insert into bids (price, volume, state_id) VALUES ({0}, {1}, {2});'.format(bids[0], bids[1], id)
            cur.execute(sql)




if __name__ == '__main__':
    record = Record()
    i = 0
    while True:
        try:
            record.save_data()
            print(i)
            i += 1
        except Exception:
            print('Exception')
            continue
