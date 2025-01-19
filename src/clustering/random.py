from .base import BaseClustering
from typing import List

from ..types import Period

import numpy as np

class RandomClustering(BaseClustering):
    def fit(self, X: List[Period], num_clusters: int) -> "BaseClustering":
        self.periods = X[1:]
        self.labels = np.random.randint(0, num_clusters, len(X)-1)
        return self