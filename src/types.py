import json
from datetime import datetime, timedelta
from os import error
from shutil import Error
import time

from nbclient.client import timestamp
from tqdm import tqdm
from typing import Dict, List, Any, Union, TypedDict, Literal
from dataclasses import dataclass, field
import os
import pandas as pd
import tarfile

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from scipy.linalg import eigh

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
            self.spread = float(self.ask_price) - float(self.bid_price)
            if (self.ask_volume + self.bid_volume) != 0:
                self.quote_volume_imbalance = (float(self.ask_volume) - float(self.bid_volume)) / (self.ask_volume + self.bid_volume) 
            else:
                self.quote_volume_imbalance = None
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
    feature_vector : List[float] = field(default_factory=list)
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
        if stock_data.shape[0] == 0:
            return np.array([0, 0, 0, 0])   
        stock_data = np.where(stock_data == None, np.nan, stock_data).astype(float)
        mean_trade_price = np.nanmean(stock_data[:, 0])  # Mean of trade prices, ignoring NaNs
        sum_trade_volume = np.nansum(stock_data[:, 1])  # Sum of trade volume, ignoring NaNs
        mean_spread = np.nanmean(stock_data[:, 2])  # Mean of spread, ignoring NaNs
        mean_quote_volume_imbalance = np.nanmean(stock_data[:, 3])  # Mean of quote volume imbalance, ignoring NaNs

        # Combine into a new array
        fv_vals_stock = np.array([mean_trade_price, sum_trade_volume, mean_spread, mean_quote_volume_imbalance])

        return fv_vals_stock

    @property
    def fv_inter(self):
        # Compute the stock feature values
        per_stock_fv = np.array([self.get_stock_fv_ip(stock) for stock in self.stocks])

        # Example: Adding epsilon directly
        flattened_fv = np.concatenate(per_stock_fv)
        return flattened_fv
      
    def stocks(self):
        return set(t.stock for t in self.tick_data)
        
    def get_stock_fv(self, stock: str) -> np.ndarray:
        stock_data = np.array([t.features for t in self.per_stock_ticks.get(stock, [])])
        rel_changes = np.diff(stock_data, axis=0) / stock_data[:-1, :]
        return np.mean(rel_changes, axis=0)

    @property
    def fv(self):
        # Compute the stock feature values
        per_stock_fv = self.fv_inter.reshape(-1, len(FEATURES_KEYS))
        return np.mean(per_stock_fv, axis=0)

       

    def plot_fv(self, ax):
        # Create a bar chart
        barlist = ax.bar(FEATURES_KEYS, self.fv)

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
        #ax.set_ylim(-1, 1)

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
    
    def get_periods_fast(self, period_length: int) -> List[Period]:
        if not self.tick_data:
            print("No tick data available")
            return []

        # Sort and filter ticks within the valid date range
        sorted_ticks = sorted(self.tick_data, key=lambda t: t.timestamp)
        valid_ticks = [t for t in sorted_ticks if t.timestamp <= self.end_date]

        periods = []
        current_start = None
        current_end = None
        current_data = []

        for tick in tqdm(valid_ticks):
            # Calculate the period this tick belongs to
            delta = tick.timestamp - self.start_date
            period_num = delta // period_length
            period_start = self.start_date + period_num * period_length
            period_end = period_start + period_length

            # Ensure the period does not exceed the end date
            if period_end > self.end_date:
                period_end = self.end_date

            # Check if this tick is part of the current period
            if (period_start, period_end) == (current_start, current_end):
                current_data.append(tick)
            else:
                # Save the current period if it has data
                if current_data:
                    periods.append(Period(
                        start=current_start,
                        end=current_end,
                        tick_data=current_data,
                        stocks=self.stocks
                    ))
                # Start a new period
                current_start, current_end = period_start, period_end
                current_data = [tick]

        # Add the last collected period
        if current_data:
            periods.append(Period(
                start=current_start,
                end=current_end,
                tick_data=current_data,
                stocks=self.stocks
            ))

        nperiods = affect_fvs(periods)
        return nperiods

    @staticmethod
    def to_dict(row, stock: str, type: str) -> Dict[str, Any]:
        error_log = []
        def safe_float(value):
            """Converts value to float if possible, returns None otherwise."""
            try:
                return float(value) if value is not None else None
            except (ValueError, TypeError):
                return None

        if type == 'bbo':
            try:
                return {
                    "timestamp": xltime_to_timestamp(row.get('xltime')),
                    "stock": stock,
                    "bid_price": safe_float(row.get('bid-price')),
                    "bid_volume": safe_float(row.get('bid-volume')),
                    "ask_price": safe_float(row.get('ask-price')),
                    "ask_volume": safe_float(row.get('ask-volume')),
                    "trade_price": None,
                    "trade_volume": None
                }
            except Exception as e:
                error_log.append([e, row])
                return {
                    "timestamp": None,
                    "stock": None,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": None,
                    "trade_volume": None
                }
        else:
            try:
                return {
                    "timestamp": xltime_to_timestamp(row.get('xltime')),
                    "stock": stock,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": safe_float(row.get('trade-price')),
                    "trade_volume": safe_float(row.get('trade-volume'))
                }
            except Exception as e:
                error_log.append([e, row])
                return {
                    "timestamp": None,
                    "stock": None,
                    "bid_price": None,
                    "bid_volume": None,
                    "ask_price": None,
                    "ask_volume": None,
                    "trade_price": None,
                    "trade_volume": None
                }

    @staticmethod
    def _compute_correlation_matrix_intra(periods: Period) -> np.ndarray:
        # periods = self.get_periods(period_length_seconds)
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

    @staticmethod
    def _compute_correlation_matrix_inter(periods: Period) -> np.ndarray:
        fvs = [period.feature_vector for period in periods]
        fvs = np.array(fvs)
        fvs[np.isinf(fvs)] = np.nan
        #for i in range(1,len(fvs)):
         #   nan_mask = np.isnan(fvs[i])
          #  fvs[i,nan_mask] = fvs[i-1,nan_mask]
        n = len(fvs)
        corr_matrix = np.full((n,n), 1.0)
        for i in range(n):
            for j in range(n):
                valid_mask = ~np.isnan(fvs[i]) & ~np.isnan(fvs[j])
                if np.any(valid_mask):  # Ensure there's at least one valid pair
                    corr_matrix[i, j] = np.corrcoef(fvs[i][valid_mask], fvs[j][valid_mask])[0, 1]
                    if np.isnan(corr_matrix[i, j]):
                        print('erere',i,j)
                        print(fvs[i],fvs[j])
        return corr_matrix

    
    @staticmethod
    def compute_correlation_matrix(periods: Period, method: Literal['intra', 'inter'] = 'inter') -> np.ndarray:
        if method == 'intra':
            return Market._compute_correlation_matrix_intra(periods)
        elif method == 'inter':
            return Market._compute_correlation_matrix_inter(periods)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_fvs(self, periods: Period):
        fvs = []
        for period in periods:
            array = period.fv

            array[np.isinf(array)] = np.nan
            fvs.append(array)
        return fvs

    def get_fvs_inter(self, periods: Period):
        fvs = []
        for i in range(1, len(periods)):
            array = (periods[i].fv_inter - periods[i -1].fv_inter) / periods[i-1].fv_inter
            fvs.append(array)
        return fvs

    def build_graph(
        self, 
        periods: Period, 
        threshold: float = 0.2, 
        inter: bool = False, 
        filter_type: str = None
    ) -> nx.Graph:
        """
        Builds a graph from the filtered correlation matrix.
        
        Args:
            periods: The periods to consider.
            threshold: Minimum correlation for an edge (default: 0.2).
            inter: If True, use inter-correlation; else intra-correlation.
            filter_type: Type of filter ('delta', 's', 'g', or None).
            
        Returns:
            nx.Graph: Graph representing filtered correlations.
        """
        # Step 1: Compute correlation matrix
        method = 'inter' if inter else 'intra'
        corr_matrix = self.compute_correlation_matrix(periods, method=method)
        
        # Step 2: Apply filter if specified
        if filter_type == 'delta':
            corr_matrix = self.filter_delta(corr_matrix)
        elif filter_type == 's':
            corr_matrix = self.filter_s(corr_matrix)
        elif filter_type == 'g':
            corr_matrix = self.filter_g(corr_matrix)
        
        # Step 3: Ensure symmetry and PSD
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        corr_matrix = self.make_psd(corr_matrix)
        
        # Step 4: Build graph using vectorized operations
        n = corr_matrix.shape[0]
        G = nx.Graph()
        i, j = np.triu_indices(n, k=1)
        edges = corr_matrix[i, j]
        # Apply threshold to filter positive edges
        mask = edges > threshold
        
        G.add_edges_from(zip(i[mask], j[mask]))
        for u, v in zip(i[mask], j[mask]):
            G[u][v]['weight'] = corr_matrix[u, v]
        
        self.plot_graph(G)
        return G
    

    def filter_delta(self, corr_matrix):
        """
        Subtracts the identity matrix from the correlation matrix.
        """
        return corr_matrix - np.eye(corr_matrix.shape[0])
    
    def filter_s(self, corr_matrix):
        """
        Filters the random noise component using Random Matrix Theory (RMT).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        threshold = self.rmt_threshold(corr_matrix.shape[0])
        filtered_eigenvalues = np.where(eigenvalues > threshold, eigenvalues, 0)
        return eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T
    
    def filter_g(self, corr_matrix):
        """
        Subtracts both the random noise and the global (market) mode.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        max_idx = np.argmax(eigenvalues)
        
        # Always initialize filtered_eigenvalues
        filtered_eigenvalues = eigenvalues.copy()
        
        # Zero out the global mode if the eigenvector is "market-like"
        if np.all(eigenvectors[:, max_idx] > 0):
            filtered_eigenvalues[max_idx] = 0  # Zero out the global mode
        
        return eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

    def rmt_threshold(self, size):
        """
        Returns the RMT threshold for the eigenvalues of a random matrix.
        """
        q = size / size  # Adjust for T/N ratio if needed
        return (1 + np.sqrt(q)) ** 2
    
    def make_psd(self, matrix):
        eigenvalues, eigenvectors = eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Clamp negative eigenvalues to 0
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


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

def xltime_to_timestamp(xltime, mac=False):
    # Définir l'origine Excel
    excel_epoch = datetime(1904, 1, 1) if mac else datetime(1899, 12, 30)
    
    # Ajouter les jours (XLTime) à l'origine
    excel_date = excel_epoch + timedelta(days=xltime)
    
    # Convertir en timestamp Unix
    return int(excel_date.timestamp())


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
        check time of the data is before 8:30:00
    """
    date_time = convert_xltime_to_date(xl_time)
    return date_time.time() < datetime.strptime('08:00', '%H:%M').time() or date_time.time() >= datetime.strptime('16:30', '%H:%M').time()


def valid_row(row, type):
    if (type == 'trade'):
        if (row['trade-stringflag'] == 'theoricalprice' or row['trade-stringflag'] == 'late0day|offbook'):
            return False

    if is_before_8h30(row['xltime']):
        return False

    return True

def affect_fvs(periods):
    new_periods = []
    #new_periods.append(Period(start=periods[0].start, end=periods[0].end, tick_data=periods[0].tick_data, stocks=periods[0].stocks))
    for i in range(1,len(periods)):
        array = (periods[i].fv_inter - periods[i - 1].fv_inter) / periods[i - 1].fv_inter
        
        new_periods.append(Period(start=periods[i].start, end=periods[i].end, tick_data=periods[i].tick_data, stocks=periods[i].stocks, feature_vector=array))

    return new_periods
