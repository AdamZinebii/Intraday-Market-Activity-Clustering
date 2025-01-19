import json
from datetime import datetime, timedelta
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
DATE_FORMAT = "%Y/%m/%d"


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
        if not (self.ask_price == None or self.bid_price == None):
            self.spread = self.ask_price - self.bid_price
            self.quote_volume_imbalance = (self.ask_volume - self.bid_volume) / (self.ask_volume + self.bid_volume)
        else:
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
    stocks: List[str]
    tick_data: List[Tick] = field(default_factory=list)

    @property
    def per_stock_ticks(self) -> Dict[str, List[Tick]]:
        per_stock = {stock: [] for stock in self.stocks}
        for tick in self.tick_data:
            per_stock[tick.stock].append(tick)
        return per_stock

    def get_stock_fv(self, stock: str) -> np.ndarray:
        ticks_per_stocks = self.per_stock_ticks
        stock_data = np.array([t.features for t in ticks_per_stocks[stock]])
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

    def get_stock_fv_ip(self, stock: str) -> np.ndarray:
        ticks_per_stocks = self.per_stock_ticks
        stock_data = np.array([t.features for t in ticks_per_stocks[stock]])
        stock_data = np.where(stock_data == None, np.nan, stock_data).astype(float)
        mean_trade_price = np.nanmean(stock_data[:, 0])  # Mean of trade prices, ignoring NaNs
        sum_trade_volume = np.nansum(stock_data[:, 1])  # Sum of trade volume, ignoring NaNs
        mean_spread = np.nanmean(stock_data[:, 2])  # Mean of spread, ignoring NaNs
        mean_quote_volume_imbalance = np.nanmean(stock_data[:, 3])  # Mean of quote volume imbalance, ignoring NaNs

        # Combine into a new array
        fv_vals_stock = np.array([mean_trade_price, sum_trade_volume, mean_spread, mean_quote_volume_imbalance])

        return fv_vals_stock

    @property
    def fv(self):
        # Compute the stock feature values
        per_stock_fv = np.array([self.get_stock_fv(stock) for stock in self.stocks])

        # Example: Adding epsilon directly
        flattened_fv = np.concatenate(per_stock_fv)
        return flattened_fv

    @property
    def fv_inter(self):
        # Compute the stock feature values
        per_stock_fv = np.array([self.get_stock_fv_ip(stock) for stock in self.stocks])

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
    tick_data: List[Tick]
    start_date: int
    end_date: int

    @property
    def stocks(self):
        return list(set(tick.stock for tick in self.tick_data))

    @classmethod
    def loader(cls, start, end, year_path):
        error_tick = 0
        error_log = []
        tick_data = []
        start_date = datetime.strptime(start, DATE_FORMAT)
        end_date = datetime.strptime(end, DATE_FORMAT)
        for entry in tqdm(os.listdir(year_path)):
            with tarfile.open(f'{year_path}/{entry}', 'r') as tar:
                for member in (tar.getmembers()):
                    print(member)
                    day_data = (member.name)
                    stock = day_data.split('/')[7]
                    type = day_data.split('/')[6]

                    date = datetime.strptime('/'.join(day_data.split('/')[-1].split('-')[:3]), DATE_FORMAT)
                    if (day_data.split('.')[-1]) == 'parquet' and date > start_date and date < end_date:
                        df = pd.read_parquet(tar.extractfile(day_data))
                        tick_data.extend(Market.extend_from_pandas(df, stock, type))

        tick_data = [tick for tick in tick_data if tick.timestamp is not None]
        stocks = list(set(tick.stock for tick in tick_data))

        return cls(tick_data=tick_data, start_date=start_date.timestamp(), end_date=end_date.timestamp())

    def from_csv(cls, start_date, end_date, path: str) -> "Market":
        df = pd.read_csv(path)
        return cls.from_pandas(df, start_date=start_date, end_date=end_date)

    @classmethod
    def from_jsonl(cls, path: str, start_date: str, end_date: str):
        with open(path, 'r') as f:
            tick_data = [Tick.from_dict(json.loads(line)) for line in f]
        return cls(tick_data=tick_data, start_date=datetime.strptime(start_date, DATE_FORMAT).timestamp(),
                   end_date=datetime.strptime(end_date, DATE_FORMAT).timestamp())
        # df = pd.read_json(path, lines=True)
        # return cls.from_pandas(df)

    @classmethod
    def from_pandas(cls, df, start_date, end_date) -> "Market":
        tick_data = [Tick.from_dict(row.to_dict()) for _, row in df.iterrows()]
        return cls(tick_data=tick_data, start_date=datetime.strptime(start_date, DATE_FORMAT).timestamp(),
                   end_date=datetime.strptime(end_date, DATE_FORMAT).timestamp())

    @staticmethod
    def extend_from_pandas(df, stock, type):
        return [Tick.from_dict(Market.to_dict(row, stock, type)) for _, row in df.iterrows() if valid_row(row, type)]

    def get_periods(self, period_length: int) -> List[Period]:
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
            elif tick.timestamp > start and tick.timestamp < self.end_date:
                if current_period_data:
                    periods.append(Period(start=start, end=end, tick_data=current_period_data, stocks=self.stocks))
                start = end
                end += period_length
                current_period_data = []

                if end >= self.end_date:
                    end = self.end_date
            elif tick.timestamp > self.end_date:
                if current_period_data:
                    periods.append(Period(start=start, end=end, tick_data=current_period_data, stocks=self.stocks))
                break

        return periods

    @staticmethod
    def to_dict(row, stock: str, type: str) -> Dict[str, Any]:
        error_log = []
        if type == 'bbo':
            try:
                ret = {
                    "timestamp": row['xltime'],
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
                error_log.append([e, row])
                error_tick += 1
                ret = {
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
            try:
                return {
                    "timestamp": row['xltime'],
                    "stock": stock,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": row['trade-price'],
                    "trade_volume": row['trade-volume']

                }
            except (ValueError, TypeError, ZeroDivisionError) as e:
                error_log.append([e, row])
                error_tick += 1
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
        periods = self.get_periods(period_length_seconds)
        fvs = []
        for period in periods:
            array = period.fv
            array = np.array(array, dtype=np.float64)

            array[np.isinf(array)] = np.nan
            fvs.append(array)
        matrice_masque = np.ma.masked_invalid(fvs)

        # Calculer la matrice de corrélation
        corr_matrix = np.ma.corrcoef(matrice_masque)
        return corr_matrix.data

    def compute_correlation_matrix_inter(self, period_length_seconds: int) -> np.ndarray:
        fvs = self.get_fvs_inter(period_length_seconds)

        # Calculer la matrice de corrélation
        corr_matrix = np.ma.corrcoef(fvs)
        return corr_matrix.data

    def get_fvs(self, period_length: int):
        periods = self.get_periods(period_length)
        fvs = []
        for period in periods:
            array = period.fv

            array[np.isinf(array)] = np.nan
            fvs.append(array)
        return fvs

    def get_fvs_inter(self, period_length: int):
        periods = self.get_periods(period_length)
        fvs = []
        for i in range(len(periods[1:])):
            array = (periods[i].fv_inter - periods[i - 1].fv_inter) / periods[i - 1].fv_inter
            fvs.append(array)
        return fvs

    def build_graph(self, period_length_seconds: int, threshold=0.2, inter=False) -> nx.Graph:
        """
            This method builds a graph from the correlation matrix of the state vectors of the days.
            Each period is represented as a node in the graph and the edges are weighted by the correlation, with no edges
            when correlation is below the threshold.

            params:
                period_length: int - the length of the periods to consider.
                threshold: float - the minimum correlation value for an edge to be added to the graph
                inter: Boolean - if true use the inter correlation approximation
            return: nx.Graph - the graph representing the correlation between the periods.
        """
        if inter:
            corr_matrix = self.compute_correlation_matrix_inter(period_length_seconds)
        else:
            corr_matrix = self.compute_correlation_matrix(period_length_seconds)

        n = corr_matrix.shape[0]
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > threshold:
                    G.add_edge(i, j, weight=corr)
        # Display Graph
        self.plot_graph(G)
        return G

    def plot_graph(self, G: nx.Graph):
        """
            This method plots the graph G.
            params:
                G: nx.Graph - the graph to plot
        """
        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

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


def valid_row(row, type):
    if (type == 'trade'):
        if (row['trade-stringflag'] == 'theoricalprice' or row['trade-stringflag'] == 'late0day|offbook'):
            return False

    if is_before_8h30(row['xltime']):
        return False

    return True