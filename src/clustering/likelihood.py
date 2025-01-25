from .base import BaseClustering
from typing import List

from src.types import Period, Market

import numpy as np
import multiprocessing as mp
import random
from scipy.stats import zscore

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

class PGASolver:
    def __init__(self, likelihood_fn):
        self.likelihood_fn = likelihood_fn

    @staticmethod
    def _initialize_population(size, num_intervals, num_clusters):
        return [np.random.randint(0, num_clusters, num_intervals).tolist() for _ in range(size)]
    
    @staticmethod
    def _evaluate_population(likelihood_fn, population, C):
        with mp.Pool(mp.cpu_count()) as pool:
            scores = pool.starmap(likelihood_fn, [(ind, C) for ind in population])
        return scores
    
    @staticmethod
    def _selection(population, scores, num_parents):
        selected_indices = np.argsort(scores)[-num_parents:]
        return [population[i] for i in selected_indices]
    
    @staticmethod
    def _crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    @staticmethod
    def _mutate(individual, num_clusters, mutation_rate=0.1):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = random.randint(0, num_clusters - 1)
        return individual

    def solve(
        self,
        C: np.ndarray, 
        num_clusters: int = 3, 
        population_size: int = 20, 
        generations: int = 200, 
        mutation_rate: float = 0.09
    ) -> tuple:
        num_intervals = len(C)
        population = PGASolver._initialize_population(population_size, num_intervals, num_clusters)

        best_score = float('-inf')
        best_solution = None
        
        for g in tqdm(range(generations), desc='Solving with PGA...', total=generations):
            scores = PGASolver._evaluate_population(self.likelihood_fn, population, C)
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_solution = population[best_idx]
            
            # Sélection
            parents = PGASolver._selection(population, scores, num_parents=population_size // 2)
            next_population = []
            
            # Croisement
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = PGASolver._crossover(parents[i], parents[i + 1])
                    next_population.append(child1)
                    next_population.append(child2)
            
            # Mutation
            population = [PGASolver._mutate(ind, num_clusters, mutation_rate) for ind in next_population]

        return best_solution, best_score
    
    def __call__(
        self,
        C: np.ndarray, 
        num_clusters: int = 3, 
        population_size: int = 20, 
        generations: int = 50, 
        mutation_rate: float = 0.1
    ) -> tuple:
        return self.solve(C, num_clusters, population_size, generations, mutation_rate)
    
solver_map = {
    'pga': PGASolver
}
    
def likelihood_gm(S, C):
    L = 0
    for cluster in set(S):
        indices = np.where(np.array(S) == cluster)[0]
        if len(indices) > 1:
            n_s = len(indices)
            c_s = np.sum(C[np.ix_(indices, indices)])
            L += 0.5 * (np.log(n_s / c_s) + (n_s - 1) * np.log((n_s**2 - n_s) / (n_s**2 - c_s)))
    return L
    
likelihood_map = {
    'gm': likelihood_gm
}

class LikelihoodClustering(BaseClustering):
    def __init__(self, solver='pga', likelihood_fn='gm'):
        super().__init__()
        likelihood_fn = likelihood_map[likelihood_fn]
        self.solver = solver_map[solver](likelihood_fn)

    def _fit(self, X: List[Period], **kwargs) ->  List[int]:

        corr = Market.compute_correlation_matrix(X)

        assert len(X) == corr.shape[0], 'Number of periods must match the number of rows in the correlation matrix'

        best_solution, best_score = self.solver(corr, **kwargs)
        print(f'Best solution : {best_solution}')
        print(f'Best score : {best_score:.4f}')
        return best_solution