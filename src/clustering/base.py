from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from src.types import Period, FEATURES_KEYS

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
        input_fv = np.array([x.fv for x in X])
        return np.argmin(euclidean_distances(self.ssv, input_fv), axis=0)
    
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
    def clusters_sizes(self):
        return [len(cluster) for cluster in self.clusters]
    
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
            centroid = np.mean([p.fv for p in c], axis=0)
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
        return cosine_similarity(self.ssv, [x.fv for x in X]).mean()
    
    def plot_ssv(self):
        fig, ax = plt.subplots(nrows=self.n_clusters, figsize=(10, 5 * self.n_clusters))

        for i, ssv in enumerate(self.ssv):
            # Create a bar chart
            barlist = ax[i].bar(FEATURES_KEYS, ssv)

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
        # Ensure labels are the same length as the number of nodes in G
        if len(self.labels) != len(G.nodes):
            raise ValueError("The length of 'self.labels' must be equal to the number of nodes in G")

        # Create a color map based on the community labels
        colors = [self.labels[i] for i in range(len(self.labels))]  # Community label for each node

        # Generate positions for the nodes using a layout
        pos = nx.spring_layout(G)  # You can use other layouts like nx.kamada_kawai_layout(G), nx.circular_layout(G), etc.

        # Plot the graph
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, node_size=500, node_color=colors, with_labels=True, cmap=plt.cm.get_cmap("tab10"), 
                font_size=10, font_weight='bold', edge_color='gray', width=0.5)

        plt.title(f"Community Graph: {method}", fontsize=15)
        plt.show()

    def plot_cluster(self, method: str) -> None:
        """
        Plot the graph with clusters as fully connected subgraphs and nodes colored by their time period.

        Parameters
        ----------
        method : str
            Name of the community detection method.

        Returns
        -------
        None
        """

        labels = self.labels  
        cluster_graph = nx.Graph()  
        node_to_cluster = {}  

        periods = self.periods

        # Create clusters in the graph
        for i in set(labels):
            cluster_nodes = [idx for idx, label in enumerate(labels) if label == i]
            cluster_graph.add_nodes_from(cluster_nodes)
            cluster_graph.add_edges_from([(u, v) for u in cluster_nodes for v in cluster_nodes if u < v])
            
            for node in cluster_nodes:
                node_to_cluster[node] = i

        def get_time_of_day(period: 'Period') -> float:
            timestamp = period.start
            dt = datetime.fromtimestamp(timestamp)
            return dt.hour + dt.minute / 60  

        # Map node colors based on time of day
        time_of_day_values = [get_time_of_day(periods[node]) for node in cluster_graph.nodes]
        cmap = plt.cm.RdYlGn.reversed()  
        norm = mcolors.Normalize(vmin=8, vmax=16.5)  # Set max value to 16.5 (4:30 PM)  

        node_colors = [cmap(norm(value)) for value in time_of_day_values]

        # Adjust the layout with increased spacing
        pos = nx.spring_layout(cluster_graph, seed=42, k=1.0)  # Increase 'k' to spread nodes further apart
        fig, ax = plt.subplots(figsize=(14, 12))  

        # Draw the graph
        nx.draw(
            cluster_graph,
            pos,
            node_size=600,
            node_color=node_colors,
            with_labels=True,
            font_size=10,
            edge_color="gray",
            ax=ax,  
        )
            
        # Add a color bar for the time of day
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
        cbar.set_label("Time of Day (Market Hours)", fontsize=12)
        cbar.set_ticks([8, 10, 12, 14, 16, 16.5])  # Add 16.5 for 4:30 PM
        cbar.set_ticklabels(["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "4:30 PM"])  
        
        # Add title and display the plot
        ax.set_title(f"Clustered Graph - {method}", fontsize=18)
        plt.show()
    
    def plot_cluster_2(self, method: str) -> None:
        """
        Plot the graph with clusters as fully connected subgraphs and nodes colored by their time period.
        Clusters are spatially separated to prevent overlap.
        """
        labels = self.labels  
        cluster_graph = nx.Graph()  
        node_to_cluster = {}  
        periods = self.periods

        # Create clusters with fully connected subgraphs
        for i in set(labels):
            cluster_nodes = [idx for idx, label in enumerate(labels) if label == i]
            cluster_graph.add_nodes_from(cluster_nodes)
            cluster_graph.add_edges_from([(u, v) for u in cluster_nodes for v in cluster_nodes if u < v])
            for node in cluster_nodes:
                node_to_cluster[node] = i

        # Time-of-day color mapping
        def get_time_of_day(period: 'Period') -> float:
            dt = datetime.fromtimestamp(period.start)
            return dt.hour + dt.minute / 60  

        time_values = [get_time_of_day(periods[node]) for node in cluster_graph.nodes]
        cmap = plt.cm.RdYlGn.reversed()
        norm = mcolors.Normalize(vmin=8, vmax=16.5)
        node_colors = [cmap(norm(v)) for v in time_values]

        # Create cluster-aware layout
        def cluster_layout(graph, cluster_mapping):
            pos = {}
            clusters = sorted(set(cluster_mapping.values()))
            
            # Arrange clusters in a grid pattern
            grid_size = int(np.ceil(np.sqrt(len(clusters))))
            
            # Generate cluster positions using cluster IDs
            cluster_positions = {}
            for idx, cid in enumerate(clusters):
                row = idx // grid_size
                col = idx % grid_size
                cluster_positions[cid] = (col * 5, row * 5)  # 5-unit spacing
            
            # Create layout for each cluster
            for cid in clusters:
                cluster_nodes = [n for n in graph.nodes if cluster_mapping[n] == cid]
                subgraph = graph.subgraph(cluster_nodes)
                
                # Use circular layout for individual clusters
                sub_pos = nx.circular_layout(subgraph, scale=1.5)
                
                # Offset to cluster position
                for node, (x, y) in sub_pos.items():
                    pos[node] = (x + cluster_positions[cid][0], 
                                y + cluster_positions[cid][1])
            return pos

        # Create figure and plot
        fig, ax = plt.subplots(figsize=(16, 14))
        pos = cluster_layout(cluster_graph, node_to_cluster)

        # Draw with improved visualization parameters
        nx.draw_networkx_nodes(
            cluster_graph, pos,
            node_size=400,
            node_color=node_colors,
            edgecolors='black',
            linewidths=0.5,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            cluster_graph, pos,
            alpha=0.15,
            edge_color='gray',
            width=0.8,
            ax=ax
        )

        # Color bar setup
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Time of Day (Market Hours)", fontsize=12)
        cbar.set_ticks([8, 10, 12, 14, 16, 16.5])
        cbar.set_ticklabels(["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "4:30 PM"])

        # Final touches
        ax.set_title(f"Clustered Graph - {method}", fontsize=20, pad=20)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    def summarize_clusters(self):
        print(f"Number of periods: {len(self.periods)}")
        print(f"Number of clusters: {len(self.clusters)}")
        print(f"Cluster sizes: {[len(cluster) for cluster in self.clusters]}")
        print(f"Cluster labels: {self.labels}")
        #print(f"Cluster centers: {clustering.cluster_centers}")
        print(f"Transition matrix: \n{self.transition_matrix}")

   
    def power_law(self):
        """
            Returns the relative 
        """
        #  We define the model as: x^alpha ~ p(x)
        model = lambda x, alpha: x**(-alpha)
        alpha, pvalue = curve_fit(model, xdata=self.clusters_sizes)

        return alpha, pvalue




