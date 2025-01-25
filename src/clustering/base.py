from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.special import zeta
import powerlaw

import seaborn as sns

from src.types import Period, FEATURES_KEYS

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

import numpy as np
import networkx as nx
from datetime import datetime, timezone

import os

class BaseClustering(ABC):
    periods: List[Period] = None
    labels: np.ndarray = None

    @abstractmethod
    def _fit(self, X: List[Period], **kwargs) -> List[int]:
        """
        Fit the clustering algorithm to the data
        Parameters
        ----------
        X : List[Period]
            List of periods to fit the clustering algorithm to
        Returns
        -------
        List[int]
            List of cluster labels for each period
        """
        pass

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
        X = X[1:]
        # Store the periods and labels
        self.periods = X
        self.labels = self._fit(X, **kwargs)
        return self


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
            labels_array = np.array(self.labels)
            cluster_indices = np.where(labels_array == i)[0]  # Indices of periods in cluster i
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
            tab = [p.fv for p in c]
            centroid = np.mean(tab, axis=0)
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
    
    def plot_transition_matrix(self, path: str = None):
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            self.transition_matrix, 
            cmap='Blues',
            vmin=0, vmax=1,
            annot=True, fmt=".2f",
            xticklabels=range(1, self.n_clusters+1),
            yticklabels=range(1, self.n_clusters+1)
        )
        plt.xlabel('$state_{t+1}$')
        plt.ylabel('$state_{t}$')
        plt.title('Transition Matrix')

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300)

        plt.show()
    
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
    
    def plot_ssv(self, path: str = None):
        # Determine the grid layout for subplots
        cols = 3  # Number of plots per row
        rows = (self.n_clusters + cols - 1) // cols  # Compute the number of rows needed
        
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))  # Adjust the figure size for better visuals
        ax = ax.flatten()  # Flatten the grid into a 1D array for easier iteration
        
        for i, ssv in enumerate(self.ssv):
            if np.isnan(ssv).any():  # Skip if ssv contains NaN
                continue

            # Create a bar chart
            barlist = ax[i].bar(FEATURES_KEYS, ssv)

            # Set the color of the bars
            barlist[0].set_color('r')
            barlist[1].set_color('g')
            barlist[2].set_color('b')
            barlist[3].set_color('y')

            ax[i].set_xticks(range(4))
            ax[i].set_xticklabels(['Trade Price', 'Trade Volume', 'Spread', 'QVI'])
            ax[i].set_ylabel('Value')
            ax[i].set_title(f'Cluster {i}')

        # Remove empty subplots if any
        for j in range(len(self.ssv), len(ax)):
            fig.delaxes(ax[j])  # Remove unused axes

        plt.suptitle('Clusters SSV')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

        # Save the plot
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300)

        plt.show()

    
    def plot_community_graph(
            self, 
            method: str,
            path: str = None
        ) -> None:
        """
        Plot the graph with clusters as fully connected subgraphs and nodes colored by their time period.
        Clusters are spatially separated to prevent overlap. Additionally, save and print stats of the number
        of periods per hour. Nodes with times after 16:00 are excluded from the plot.
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
        
        # Exclude nodes with times after 16:00 (4:00 PM)
        filtered_nodes = [node for node, time in zip(cluster_graph.nodes, time_values) if time <= 16]
        cluster_graph = cluster_graph.subgraph(filtered_nodes).copy()
        time_values = [get_time_of_day(periods[node]) for node in cluster_graph.nodes]

        cmap = plt.cm.RdYlGn.reversed()
        norm = mcolors.Normalize(vmin=8, vmax=16)
        node_colors = [cmap(norm(v)) for v in time_values]

        # Save and print statistics of periods per hour
        period_counts = {}
        for node in cluster_graph.nodes:
            dt = datetime.fromtimestamp(periods[node].start)
            time_slot = dt.strftime('%H:%M')
            if time_slot not in period_counts:
                period_counts[time_slot] = 0
            period_counts[time_slot] += 1

        # Sort and print the stats
        print("Count of periods per time:")
        for time_slot, count in sorted(period_counts.items()):
            print(f"{time_slot} : {count}")

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
                cluster_nodes = [n for n in graph.nodes if cluster_mapping.get(n) == cid]
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
        cbar.set_ticks([8, 10, 12, 14, 16])
        cbar.set_ticklabels(["8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM"])

        # Final touches
        ax.set_title(f"Clustered Graph - {method}", fontsize=20, pad=20)
        ax.set_axis_off()
        plt.tight_layout()

        # Save the plot
        if path:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300)

        plt.show()

    def summarize_clusters(self):
        print(f"Number of periods: {len(self.periods)}")
        print(f"Number of clusters: {len(self.clusters)}")
        print(f"Cluster sizes: {[len(cluster) for cluster in self.clusters]}")
        print(f"Cluster labels: {self.labels}")
        #print(f"Cluster centers: {clustering.cluster_centers}")
        print(f"Transition matrix: \n{self.transition_matrix}")

    def plot_power_law(self, path: str = None):
        """
            Plots the power law distribution of the cluster sizes
        """
        # Step 1: Load or generate data
        data = np.array(self.clusters_sizes)

        # Step 2: Fit the power-law distribution
        fit = powerlaw.Fit(data)  # Automatically finds x_min and alpha
        alpha = fit.alpha
        x_min = int(fit.xmin)
        #loglikelihood = fit.loglikelihood
        pvalue = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)[1]

        print(f"Estimated alpha: {alpha}")
        print(f"Estimated x_min: {x_min}")
        #print(f"Log-likelihood: {loglikelihood}")
        print(f"P-value: {pvalue}")

        # Step 3: Plot the ICDF (log-log scale)
        values, counts = np.unique(data, return_counts=True)
        cdf = np.cumsum(counts) / np.sum(counts)
        icdf = cdf[::-1]

        # Define the ICDF function for the power-law distribution
        above_xmin = values[values >= x_min]   
        fitted_values, fitted_ccdf = fit.ccdf()

        plt.figure(figsize=(10, 6))

        plt.loglog(values, icdf, label="Empirical CCDF", marker='o', linestyle='none')
        plt.loglog(fitted_values, fitted_ccdf, label="Fitted CCDF", linestyle='--')

        plt.xlabel("x")
        plt.ylabel("$P(X \geq x)$")

        plt.legend()

        plt.title(f"Power-law distribution of cluster sizes: $\\alpha={alpha:.2f}$, $x_{{\\min}}={x_min}$, $p_{{\\text{{value}}}}={pvalue:.5f}$")

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300)
        plt.show()



