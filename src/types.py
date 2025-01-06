import json
from datetime import datetime,timedelta
from os import error
from shutil import Error

from nbclient.client import timestamp
from tqdm import tqdm
from typing import Dict, List, Any, Union, TypedDict
from dataclasses import dataclass, field
import os
import pandas as pd
import tarfile

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
FEATURES_KEYS = ['trade_price', 'trade_volume', 'spread', 'quote_volume_imbalance']


@dataclass
class Tick:
    # Timestamp and stock symbol
    timestamp: int
    stock: str

    # Bid and ask prices and volumes
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float

    # Trade price and volume
    trade_price: float
    trade_volume: float


    def __post_init__(self):
        if not (self.ask_price == None or self.bid_price == None) :
            self.spread = self.ask_price - self.bid_price
            self.quote_volume_imbalance = (self.ask_volume - self.bid_volume) / (self.ask_volume + self.bid_volume)
        else :
            self.spread = None
            self.quote_volume_imbalance = None


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tick":
        return cls(
            timestamp=data["timestamp"],
            stock=data["stock"],
            bid_price=data["bid_price"],
            bid_volume=data["bid_volume"],
            ask_price=data["ask_price"],
            ask_volume=data["ask_volume"],
            trade_price=data["trade_price"],
            trade_volume=data["trade_volume"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stock": self.stock,
            "bid_price": self.bid_price,
            "bid_volume": self.bid_volume,
            "ask_price": self.ask_price,
            "ask_volume": self.ask_volume,
            "trade_price": self.trade_price,
            "trade_volume": self.trade_volume,
            "spread": self.spread,
            "quote_volume_imbalance": self.quote_volume_imbalance
        }


    
    @property
    def features(self) -> List[float]:
        d = self.to_dict()
        return [d[k] for k in FEATURES_KEYS]
    
@dataclass
class Period:
    start: int
    end: int
    stocks : List[str]
    tick_data: List[Tick] = field(default_factory=list)


    @property
    def per_stock_ticks(self) -> Dict[str, List[Tick]]:
        per_stock = {stock : [] for stock in self.stocks}
        for tick in self.tick_data:
            per_stock[tick.stock].append(tick)
        return per_stock

    def get_stock_fv(self, stock: str) -> np.ndarray:
        ticks_per_stocks = self.per_stock_ticks
        stock_data = np.array([t.features for t in ticks_per_stocks[stock] ])
        '''
        stock_data_df = pd.DataFrame(stock_data,columns=FEATURES_KEYS)
        stock_data_df_rel_change = stock_data_df.apply(pct_change_ignore_nan)
        #rel_changes = np.diff(stock_data, axis=0) / (stock_data[:-1, :] + epsilon)
        #rel_changes = np.nan_to_num(rel_changes, nan=0.0, posinf=0.0, neginf=0.0)

        # Small constant to prevent division by zero

        # Example: Adding epsilon directly
        mean_rel_changes = stock_data_df_rel_change.mean(skipna=True)
        print('hhhhh',mean_rel_changes)
        '''
        means = []


        for col in range(len(FEATURES_KEYS)):
            if stock_data.ndim == 1:
                print(stock_data)
            x = stock_data[:, col]

            x = x.astype(float)
            x = x[~np.isnan(x)]
            x = np.diff(x) / x[:-1]

            means.append(np.mean(x))

        return means

    @property
    def fv(self):
        # Compute the stock feature values
        per_stock_fv = np.array([self.get_stock_fv(stock) for stock in self.stocks])

        # Example: Adding epsilon directly
        flattened_fv = np.concatenate(per_stock_fv)
        return flattened_fv




    def plot_ssv(self, ax, color='b'):
        # Create a bar chart
        barlist = ax.bar(FEATURES_KEYS, self.ssv)

        # Set the color of the bars
        barlist[0].set_color('r')
        barlist[1].set_color('g')
        barlist[2].set_color('b')
        barlist[3].set_color('y')

        # Add a title and labels
        ax.set_title(f'Period {self.start}-{self.end}')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Trade Price', 'Trade Volume', 'Spread', 'Quote Volume Imbalance'])
        ax.set_ylabel('Value')

        ax.grid()

@dataclass
class Market:

    def __init__(self,start,end,year_path):
        date_format1 = "%Y/%m/%d"
        self.error_tick = 0
        self.error_log = []
        self.tick_data = []
        self.start_date = datetime.strptime(start, date_format1)
        self.end_date = datetime.strptime(end, date_format1)
        for entry in tqdm(os.listdir(year_path)):
            with tarfile.open(f'{year_path}/{entry}', 'r') as tar:
                for member in (tar.getmembers()):
                    day_data = (member.name)
                    stock = day_data.split('/')[7]
                    type = day_data.split('/')[6]
                    date_format2 = "%Y/%m/%d"

                    date = datetime.strptime('/'.join(day_data.split('/')[-1].split('-')[:3]), date_format2)
                    if (day_data.split('.')[-1]) == 'parquet' and date > self.start_date and date < self.end_date:
                        df = pd.read_parquet(tar.extractfile(day_data))
                        self.extend_from_pandas(df, stock, type)

        self.tick_data = [tick for tick in self.tick_data if tick.timestamp is not None]

        self.stocks  = list(set(tick.stock for tick in self.tick_data))



    @classmethod
    def from_csv(cls, path: str) -> "Market":
        df = pd.read_csv(path)
        return cls.from_pandas(df)
    
    @classmethod
    def from_jsonl(cls, path: str):
        with open(path, 'r') as f:
            tick_data = [Tick.from_dict(json.loads(line)) for line in f]
        return cls(tick_data=tick_data)
        # df = pd.read_json(path, lines=True)
        # return cls.from_pandas(df)

    def from_pandas(cls, df) -> "Market":
        tick_data = [Tick.from_dict(row.to_dict()) for _, row in df.iterrows()]
        return cls(tick_data=tick_data)


    def extend_from_pandas(self,df,stock,type):
        self.tick_data.extend([Tick.from_dict(self.to_dict(row,stock, type)) for _, row in df.iterrows() if valid_row(row,type)])


    def get_periods(self, period_length) -> List[Period]:
        periods = []

        if not self.tick_data:
            return periods

        data = sorted(self.tick_data, key=lambda t: t.timestamp)
        start = self.start_date
        end = self.start_date + period_length

        current_period_data = []
        idx = 0

        while True:
            if idx >= len(data):
                break

            tick = data[idx]

            if tick.timestamp < end:


                current_period_data.append(tick)
                idx += 1
            elif tick.timestamp > start  and tick.timestamp < self.end_date:
                if current_period_data:
                    periods.append(Period(start=start, end=end, tick_data=current_period_data, stocks= self.stocks))
                start = end
                end += period_length
                current_period_data = []

                if end >= self.end_date:
                    end = self.end_date
            elif tick.timestamp > self.end_date:
                if current_period_data:
                    periods.append(Period(start=start, end=end, tick_data=current_period_data, stocks = self.stocks))
                break

        return periods

    def to_dict(self, row, stock: str, type: str) -> Dict[str, Any]:
        if type == 'bbo':
            try:
                ret =  {
                    "timestamp": convert_xltime_to_date(row['xltime']),
                    "stock": stock,
                    "bid_price": row['bid-price'],
                    "bid_volume": row['bid-volume'],
                    "ask_price": row['ask-price'],
                    "ask_volume": row['ask-volume'],
                    "trade_price": None,
                    "trade_volume": None

                }
                return ret
            except (ValueError, TypeError, ZeroDivisionError) as e:
                self.error_log.append([e,row])
                self.error_tick += 1
                ret =  {
                    "timestamp": None,
                    "stock": None,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": None,
                    "trade_volume": None

                }
                return ret
        else:
            try :
                return {
                    "timestamp": convert_xltime_to_date(row['xltime']),
                    "stock": stock,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": row['trade-price'],
                    "trade_volume": row['trade-volume']

                }
            except (ValueError, TypeError, ZeroDivisionError) as e:
                self.error_log.append([e, row])
                self.error_tick += 1
                return {
                    "timestamp": None,
                    "stock": None,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": None,
                    "trade_volume": None,

                }



    def compute_correlation_matrix(self, period_length_seconds: int) -> np.ndarray:
        period_length = (timedelta(seconds=period_length_seconds))
        periods = self.get_periods(period_length)
        fvs = []
        for period in periods:
            array = period.fv
            array = np.array(array, dtype=np.float64)

            array[np.isinf(array)] = np.nan
            fvs.append(array)
        matrice_masque = np.ma.masked_invalid(fvs)


        # Calculer la matrice de corrélation
        corr_matrix = np.ma.corrcoef(matrice_masque)
        return  corr_matrix.data

    def get_fvs(self, period_length_seconds: int):
        period_length = (timedelta(seconds=period_length_seconds))
        periods = self.get_periods(period_length)
        fvs = []
        for period in periods:
            array = period.fv


            array[np.isinf(array)] = np.nan
            fvs.append(array)
        return  fvs


    
    def build_graph(self) -> nx.Graph:
        ...

    def __getitem__(self, key):
        return self.tick_data[key]



def convert_xltime_to_date(xl_time):
    excel_start_date = datetime(1899, 12, 30)  # Excel incorrectly considers 1900 as a leap year
    return excel_start_date + timedelta(days=float(xl_time))


def forward_fill(array):
    """
    Remplit les valeurs None dans un tableau NumPy avec la dernière valeur valide trouvée dans les lignes précédentes.

    :param array: np.ndarray, tableau NumPy contenant des listes avec des valeurs potentiellement None.
    :return: np.ndarray, tableau avec les valeurs None remplacées.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("L'entrée doit être un tableau NumPy")

    # Copie pour éviter de modifier l'original
    filled_array = array.copy()

    # Parcourir chaque ligne
    for i in range(1, filled_array.shape[0]):
        for j in range(len(filled_array[i])):
            if filled_array[i][j] is None:
                # Chercher dans les lignes précédentes
                for k in range(i - 1, -1, -1):
                    if filled_array[k][j] is not None:
                        filled_array[i][j] = filled_array[k][j]
                        break
    return filled_array


def pct_change_ignore_nan(series):
    """
    Calcule le pourcentage de variation en utilisant la première valeur précédente valide.

    :param series: pd.Series - Série Pandas avec des NaN.
    :return: pd.Series - Série avec les pourcentages de variation.
    """
    result = pd.Series(index=series.index, dtype=float)
    last_valid = np.nan  # Stocke la dernière valeur valide

    for i in range(len(series)):
        current = series.iloc[i]
        if not pd.isna(current):
            if not pd.isna(last_valid):
                result.iloc[i] = (current - last_valid) / last_valid
            last_valid = current  # Mettre à jour avec la dernière valeur valide

    return result

def is_before_8h30(xl_time):
    """
    Vérifie si l'heure est avant 8h30.
    """
    date_time = convert_xltime_to_date(xl_time)
    return date_time.time() < datetime.strptime('08:00', '%H:%M').time()

def valid_row(row,type):
    if (type=='trade'):
        if  (row['trade-stringflag'] == 'theoricalprice' or row['trade-stringflag'] == 'late0day|offbook'):
            return False

    if is_before_8h30(row['xltime']):
        return False

    return True