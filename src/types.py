import json
from typing import Dict, List, Any, Union, TypedDict
from dataclasses import dataclass, field

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

FV_KEYS = ['trade_price', 'trade_volume', 'spread', 'quote_volume_imbalance']

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

    # Spread and quote volume imbalance
    spread: float = None
    quote_volume_imbalance: float = None

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

    def __post_init__(self):
        self.spread = self.ask_price - self.bid_price
        self.quote_volume_imbalance = (self.ask_volume - self.bid_volume) / (self.ask_volume + self.bid_volume)

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
    def fv(self) -> List[float]:
        d = self.to_dict()
        return [d[k] for k in FV_KEYS]
    
@dataclass
class Period:
    start: int
    end: int
    tick_data: List[Tick] = field(default_factory=list)

    @property
    def ssv(self):
        fvs = np.array([t.fv for t in self.tick_data])
        return np.mean(fvs, axis=0)

    def plot_ssv(self, ax, color='b'):
        # Create a bar chart
        barlist = ax.bar(FV_KEYS, self.ssv)

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
    tick_data: List[Tick] = field(default_factory=list)
        
    @classmethod
    def from_pandas(cls, df) -> "Market":
        tick_data = [Tick.from_dict(row.to_dict()) for _, row in df.iterrows()]
        return cls(tick_data=tick_data)

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

    def get_periods(self, period_length: int) -> List[Period]:
        periods = []
        if not self.tick_data:
            return periods

        data = sorted(self.tick_data, key=lambda t: t.timestamp)
        start = data[0].timestamp
        end = start + period_length

        current_period_data = []
        for tick in data:
            if tick.timestamp < end:
                current_period_data.append(tick)
            else:
                if current_period_data:
                    periods.append(Period(start=start, end=end, tick_data=current_period_data))
                start = end
                end += period_length
                current_period_data = [tick]

        if current_period_data:
            periods.append(Period(start=start, end=end, tick_data=current_period_data))

        return periods
    
    #NOTE This methods is for test purposes, it should be implemented
    def compute_correlation_matrix(self, X: List[Period]) -> np.ndarray:
        n = len(X)  # Number of periods
        # Generate a random symmetric matrix with values between -1 and 1
        random_matrix = np.random.uniform(-1, 1, size=(n, n))
        # Ensure symmetry
        corr_matrix = (random_matrix + random_matrix.T) / 2
        # Set diagonal values to 1 (perfect self-correlation)
        np.fill_diagonal(corr_matrix, 1)
        return corr_matrix
        
    
    def build_graph(self, X: List[Period], threshold=0.2) -> nx.Graph:
        """
            This method builds a graph from the correlation matrix of the state vectors of the days.
            Each period is represented as a node in the graph and the edges are weighted by the correlation, with no edges
            when correlation is below the threshold.

            params:
                period_length: int - the length of the periods to consider.
                threshold: float - the minimum correlation value for an edge to be added to the graph
            return: nx.Graph - the graph representing the correlation between the periods. 
        """
        periods = X
        corr_matrix = self.compute_correlation_matrix(X)
     
        n = len(periods)
        G = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > threshold: # J'ai mis la valeur absolue, Avis aux experts?
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
    
    