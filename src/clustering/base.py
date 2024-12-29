from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt

from src.types import Period, FV_KEYS

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

import numpy as np

class BaseClustering(ABC):
    periods: List[Period] = None
    labels: np.ndarray = None

    # NOTE: This should populate self.labels and self.periods
    @abstractmethod
    def fit(self, X: List[Period], **kwargs) -> "BaseClustering":
        """ 
        Fit the clustering algorithm to the data

        Parameters
        ----------
        X : List[Period]
            List of periods to fit the clustering algorithm to
        
        Returns
        -------
        np.ndarray
            Array of indices of the cluster for each period
        """
        pass

    # NOTE: This can be overwritten if the clustering algorithm has a predict method, 
    # otherwise it will use the default implementation: euclidean_distances of SSVs
    def predict(self, X: List[Period]) -> np.ndarray:
        """
        Predict the closest cluster for each period in X

        Parameters
        ----------
        X : List[Period]
            List of periods to predict the closest cluster for
        
        Returns
        -------
        np.ndarray
            Array of indices of the closest cluster for each period
        """
        input_ssv = np.array([x.ssv for x in X])
        return np.argmin(euclidean_distances(self.ssv, input_ssv), axis=0)
    
    @property
    def clusters(self):
        """Clusters as a list of tuples (index, periods)"""
        return np.array([[self.periods[j] for j in np.where(self.labels == i)[0]] for i in range(self.n_clusters)])
    
    @property
    def n_clusters(self):
        """Number of clusters"""
        return len(np.unique(self.labels))
    
    @property
    def n_features(self):
        """Number of features"""
        return self.periods[0].shape[1]

    @property
    def centers(self) -> np.ndarray:
        centers = []
        for c in self.clusters:
            centroid = np.mean([p.ssv for p in c], axis=0)
            centers.append(centroid)
        return centers

    @property
    def ssv(self) -> np.ndarray:
        return self.centers
    
    def get_transition_probability_matrix(self) -> np.ndarray:
        """
        Get the transition matrix between clusters

        Returns
        -------
        np.ndarray
            Transition matrix between clusters
        """
        # Sort by the start time of the Period
        sorted_periods_with_labels = sorted(list(zip(self.periods, self.labels)), key=lambda x: x[0].start)

        transitions = np.zeros((self.n_clusters, self.n_clusters))
        for i in range(len(sorted_periods_with_labels) - 1):
            curr_state, next_state = sorted_periods_with_labels[i][1], sorted_periods_with_labels[i + 1][1]
            transitions[curr_state, next_state] += 1

        # Final state to final state
        transitions[next_state, next_state] += 1

        # Normalize the rows
        transitions = transitions / transitions.sum(axis=1, keepdims=True)

        return transitions
    
    @property
    def transition_matrix(self) -> np.ndarray:
        return self.get_transition_probability_matrix()
    
    def score(self, X: List[Period]) -> float:
        """
        Score the clustering algorithm on the data

        Parameters
        ----------
        X : List[Period]
            List of periods to score the clustering algorithm on
        
        Returns
        -------
        float
            Score of the clustering algorithm
        """
        return cosine_similarity(self.ssv, [x.ssv for x in X]).mean()
    
    def plot_ssv(self):
        fig, ax = plt.subplots(nrows=self.n_clusters, figsize=(10, 5 * self.n_clusters))

        for i, ssv in enumerate(self.ssv):
            # Create a bar chart
            barlist = ax[i].bar(FV_KEYS, ssv)

            # Set the color of the bars
            barlist[0].set_color('r')
            barlist[1].set_color('g')
            barlist[2].set_color('b')
            barlist[3].set_color('y')

            ax[i].set_xticks(range(4))
            ax[i].set_xticklabels(['Trade Price', 'Trade Volume', 'Spread', 'Quote Volume Imbalance'])
            ax[i].set_ylabel('Value')
            ax[i].set_title(f'Cluster {i}')

        #plt.legend([f'Cluster {i}' for i in range(self.n_clusters)])
        plt.suptitle('Clusters SSV')

        plt.tight_layout()
        plt.show()