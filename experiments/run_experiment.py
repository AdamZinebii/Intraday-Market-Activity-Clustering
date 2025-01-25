""""Run experiments for the clustering algorithms on the CAC40 dataset, reproducing the results of the paper."""
import os
import sys

# Get the path to the `src` directory
src_path = os.path.relpath('..')
sys.path.append(src_path)

from typing import List
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from src.types import *
from src.clustering.louvain import LouvainClustering
from src.clustering.greedy import GreedyClustering
from src.clustering.likelihood import LikelihoodClustering

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_experiment(market: Market, periods: List[Period], results_path: str, year: int, period_length: int):
    """Run the clustering experiment for the given market and periods."""
    # Initialize the graph for the graph-based clustering algorithms
    state_graph_s = market.build_graph(periods, threshold=0.5, inter=True, filter_type='s')
    state_graph_g = market.build_graph(periods, threshold=0.5, inter=True, filter_type='g')
    state_graph_delta = market.build_graph(periods, threshold=0.5, inter=True, filter_type='delta')

    graphs = {
    's': state_graph_s,
    'g': state_graph_g,
    'delta': state_graph_delta
    }

    for filter_type, G in graphs.items():
        # ======================== Louvain Clustering ========================

        print(f"Filter type: {filter_type}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        
        # Clustering process
        clustering = LouvainClustering().fit(periods, G=G)

        # Get the clusters
        clusters = clustering.clusters

        print(f"Number of periods: {len(periods)}")
        print(f"Number of clusters: {len(clusters)}")
        print(f"Cluster sizes: {[len(cluster) for cluster in clusters]}")
        print(f"Transition matrix: \n{clustering.transition_matrix}")

        # Results
        clustering.plot_community_graph(method=f'Louvain - Filtering Method: {filter_type}', path=results_path+f"Louvain_{year}/louvain_cluster_{period_length}_{filter_type}.png")
        clustering.plot_ssv(path=results_path+f"Louvain_{year}/louvain_ssv_{period_length}_{filter_type}.png")
        clustering.plot_transition_matrix(path=results_path+f"Louvain_{year}/louvain_transition_{period_length}_{filter_type}.png")
        clustering.plot_power_law(path=results_path+f"Louvain_{year}/louvain_power_law_{period_length}_{filter_type}.png")

        # ======================== Greedy Clustering ========================

        print(f"Filter type: {filter_type}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        
        # Clustering process
        clustering = GreedyClustering().fit(periods,  G=G)

        # Get the clusters
        clusters = clustering.clusters

        print(f"Number of periods: {len(periods)}")
        print(f"Number of clusters: {len(clusters)}")
        print(f"Cluster sizes: {[len(cluster) for cluster in clusters]}")
        print(f"Transition matrix: \n{clustering.transition_matrix}")

        # Results
        clustering.plot_community_graph(method=f'Greedy - Filtering Method: {filter_type}', path=results_path+f"Greedy_{year}/greedy_cluster_{period_length}_{filter_type}.png")
        clustering.plot_ssv(path=results_path+f"Greedy_{year}/greedy_ssv_{period_length}_{filter_type}.png")
        clustering.plot_transition_matrix(path=results_path+f"Greedy_{year}/greedy_transition_{period_length}_{filter_type}.png")
        clustering.plot_power_law(path=results_path+f"Greedy_{year}/greedy_power_law_{period_length}_{filter_type}.png")

    # ======================== Likelihood Clustering ======================== 
    clustering = LikelihoodClustering().fit(periods, population_size=200, num_clusters=len(periods) // 10)

    print('-' * 50)
    clustering.summarize_clusters()
    print('-' * 50)

    # Results
    clustering.plot_community_graph(method='Likelihood', path=results_path+f"Likelihood_{year}/likelihood_cluster_{period_length}.png")
    clustering.plot_ssv(path=results_path+f"Likelihood_{year}/likelihood_ssv_{period_length}.png")
    clustering.plot_transition_matrix(path=results_path+f"Likelihood_{year}/likelihood_transition_{period_length}.png")
    clustering.plot_power_law(path=results_path+f"Likelihood_{year}/likelihood_power_law_{period_length}.png")

if __name__ == '__main__':
    # Year and periods settings 
    years = [2007, 2010]
    periods_length = [30, 60] * 60 

    for year in years:
        data_path = os.path.relpath(f'../data/CAC40/FR_{year}')
        results_path = os.path.relpath(f'../results/')
        for period_length in periods_length:
            print(f'Running experiment for {year} with period length {period_length} days')
            # Load data for the corresponding year
            market = Market.loader(f'{year}/03/01',f'{year}/04/10', data_path)   
            periods = market.get_periods(period_length=period_length)

            # Run the experiment
            run_experiment(market, periods, results_path, year, period_length)