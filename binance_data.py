import psycopg2
from binance.client import Client
import time
import pandas as pd
import numpy as np
from decimal import Decimal

class BinanceReader:
    def __init__(self):
        try:
            self.client = Client('7PN2Xro2pKvURLggEqvvs7lBthYhhHEExe8bYAQgAk8ERIZRDrhVH7YYSNr9BJae',
                                 'RkNjG8TzScHs9VwP7bM2LqeW4gYwJlpb8i7wpdVGG4klUnbHjeHs5BhW4CifouU1')
        except Exception:
            time.sleep(10)
            self.client = Client('7PN2Xro2pKvURLggEqvvs7lBthYhhHEExe8bYAQgAk8ERIZRDrhVH7YYSNr9BJae',
                                 'RkNjG8TzScHs9VwP7bM2LqeW4gYwJlpb8i7wpdVGG4klUnbHjeHs5BhW4CifouU1')

    def get_last_price(self):
        data = {}
        while len(data) == 0:
            try:
                data = self.client.get_ticker(symbol='BNBUSDT')
            except Exception:
                continue
            return Decimal(data['lastPrice'])

    def get_order_book(self):
        data = {}
        while len(data) == 0:
            try:
                data = self.client.get_order_book(symbol='BNBUSDT', limit=20)
            except Exception:
                continue
            return data['asks'], data['bids']

    def get_recent_trades(self):
        data = {}
        while len(data) == 0:
            try:
                data = self.client.get_recent_trades(symbol='BNBUSDT', limit=20)
            except Exception:
                continue
            dict = {}
            i = 0
            for item in data:
                dict[i] = {
                    'price': item['price'],
                    'volume': item['qty'],
                    'isBuyerMaker': item['isBuyerMaker']
                }
                i += 1
            data = pd.DataFrame.from_dict(dict, 'index')
            return np.asarray(data)