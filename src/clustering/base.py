from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt

from src.types import Period, FV_KEYS

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

import numpy as np
import networkx as nx
from datetime import datetime, timezone

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
        BaseClustering
            The fitted clustering algorithm instance
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
        cluster_list = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels == i)[0]  # Indices of periods in cluster i
            cluster_periods = [self.periods[j] for j in cluster_indices]  # Periods corresponding to those indices
            cluster_list.append(cluster_periods)
        return cluster_list
    
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
        BaseClustering
            The fitted clustering algorithm instance
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
        cluster_list = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels == i)[0]  # Indices of periods in cluster i
            cluster_periods = [self.periods[j] for j in cluster_indices]  # Periods corresponding to those indices
            cluster_list.append(cluster_periods)
        return cluster_list
    
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

    def plot_community_graph(self, G: nx.Graph, method: str) -> None:
        """
        Plot the community graph colored by community (cluster) labels.

        Parameters
        ----------
        G : nx.Graph
            The graph to visualize.
        
        Returns
        -------
        None
        """
        # Create a color map based on the community labels
        colors = [self.labels[i] for i in range(len(self.labels))]  # Community label for each period

        # Generate positions for the nodes using a layout
        pos = nx.spring_layout(G)  # You can use other layouts like nx.kamada_kawai_layout(G), nx.circular_layout(G), etc.

        # Plot the graph
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, node_size=500, node_color=colors, with_labels=True, cmap=plt.cm.get_cmap("tab10"), 
                font_size=10, font_weight='bold', edge_color='gray', width=0.5)

        plt.title(f"Community Graph: {method}", fontsize=15)
        plt.show()

    def plot_cluster(self, G: nx.Graph, periods: List['Period'], method: str) -> None:
        """
        Plot the graph with clusters as fully connected subgraphs and nodes colored by their time period.

        Parameters
        ----------
        G : nx.Graph
            The original graph.
        periods : List[Period]
            List of Period objects corresponding to the nodes in the graph.
        method : str
            Name of the community detection method.

        Returns
        -------
        None
        """
        # Step 1: Perform community detection using the provided labels (self.labels)
        labels = self.labels  # Assuming self.labels contains community labels for the nodes

        # Step 2: Create a new graph where each cluster is fully connected
        cluster_graph = nx.Graph()
        node_to_cluster = {}
        for i, community in enumerate(set(labels)):
            cluster_nodes = [node for node, label in enumerate(labels) if label == community]
            cluster_graph.add_nodes_from(cluster_nodes)
            cluster_graph.add_edges_from([(u, v) for u in cluster_nodes for v in cluster_nodes if u != v])
            for node in cluster_nodes:
                node_to_cluster[node] = i

        # Step 3: Assign colors based on the time period
        def get_time_of_day_color(period: 'Period') -> tuple:
            # Use the start timestamp of the period to determine the color
            timestamp = period.start
            
            # Convert the timestamp to a timezone-aware datetime object and extract the hour
            dt = datetime.fromtimestamp(timestamp)
            
            hour = dt.hour
            minute = dt.minute
            
            # Determine the corresponding 10-minute period index (0 to 5 for each hour)
            time_slot = (hour * 60 + minute) // 10  # Divide the day into 10-minute intervals (144 intervals in total)
            
            # Define the color gradient: red in the morning to green at 5 PM (17:00)
            if time_slot <= 51:  # First 51 intervals (00:00 to 8:30 AM)
                return plt.cm.RdYlGn(time_slot / 51)  # Red to Yellow for morning (Red in morning)
            else:  # From 8:30 AM to 5 PM (51-102)
                return plt.cm.RdYlGn((102 - time_slot) / 51)  # Green for afternoon (green at 5pm)

        node_colors = [get_time_of_day_color(periods[node]) for node in cluster_graph.nodes()]
        
        # Step 4: Generate positions for the nodes
        pos = nx.spring_layout(cluster_graph)

        # Step 5: Plot the graph
        plt.figure(figsize=(12, 10))
        nx.draw(
            cluster_graph,
            pos,
            node_size=500,
            node_color=node_colors,
            with_labels=True,
            font_size=10,
            edge_color="gray",
        )
        plt.title(f"Clustered Graph - {method}", fontsize=15)
        plt.show()