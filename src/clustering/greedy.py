from .base import BaseClustering
from typing import List

from ..types import Period

import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx


class GreedyClustering(BaseClustering):
    def _fit(self, X: List[Period], **kwargs) -> List[int]:

        # Create a graph from the periods and compute the Louvain communities
        communities = nx.community.greedy_modularity_communities(**kwargs, weight="weight")
        num_periods = len(X) - 1
        labels = [-1] * num_periods  
        for community_id, community in enumerate(communities):
            for period in community:
                if period <= num_periods - 1:
                    labels[period] = community_id
        
        return np.array(labels)
    