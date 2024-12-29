import json
from typing import Dict, List, Any, Union, TypedDict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

import numpy as np

FV_KEYS = ['trade_price', 'trade_volume', 'spread', 'quote_volume_imbalance']

class Tick(TypedDict):
    timestamp: int
    stock: str
    trade_price: float
    trade_volume: float
    spread: float
    quote_volume_imbalance: float
    
@dataclass
class Period:
    start: int
    end: int
    tick_data: List[Tick] = field(default_factory=list)

    @property
    def ssv(self):
        fvs = np.array([[t[k] for k in FV_KEYS] for t in self.tick_data])
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
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
        return cls(tick_data=data)

    def get_periods(self, period_length: int) -> List[Period]:
        periods = []
        if not self.tick_data:
            return periods

        data = sorted(self.tick_data, key=lambda t: t["timestamp"])
        start = data[0]["timestamp"]
        end = start + period_length
        current_period_data = []

        for tick in data:
            if tick["timestamp"] < end:
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

    def __getitem__(self, key):
        return self.tick_data[key]