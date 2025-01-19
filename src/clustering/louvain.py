from .base import BaseClustering
from typing import List

from ..types import Period

import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx


class LouvainClustering(BaseClustering):
    def fit(self, X: List[Period], G: nx.Graph) -> "BaseClustering":
        self.periods = X[1:]

        # Create a graph from the periods and compute the Louvain communities
        communities = nx.community.louvain_communities(G, weight='weight', seed=42)
        num_periods = len(X) -1
        labels = [-1] * num_periods  
        for community_id, community in enumerate(communities):
            for period in community:
                labels[period] = community_id
        
        self.labels = np.array(labels)

        return self
    