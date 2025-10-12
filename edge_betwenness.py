"""
==========================================================
 Girvan–Newman Optimized Community Detection (Python)
==========================================================

Description:
    This script implements an *optimized version* of the Girvan–Newman 
    algorithm for community detection in large weighted networks. 
    It supports multiprocessing, matrix-based optimization for speed, 
    modularity tracking, and multiple visualization options.

How to Run:
    Make sure your data file (e.g., "weighted_disease_edges_one_mode.tsv") 
        is in the same folder as this script.
    Run from the terminal:
            python girvan_newman_optimized.py
    The script will:
        - Read up to 1000 edges from the input file
        - Build a NetworkX graph
        - Run the optimized Girvan–Newman algorithm
        - Save communities, modularity plots, and visualizations

Required Libraries:
    You can install them using pip:
        pip install networkx matplotlib numpy

Main Features:
    - Parallelized edge betweenness calculation using multiprocessing
    - Efficient adjacency matrix representation (NumPy)
    - Modularity optimization tracking
    - Automatic early stopping when modularity decreases
    - Multiple visualization modes (static and interactive)
    - Export of communities to detailed text file

Output Files:
    - communities_1000_detailed.txt → all detected communities
    - network_detailed.png → interactive network visualization
    - network_simple.png → simplified network visualization
    - modularity_optimized_1000.png → modularity evolution plot
    
References
----------
Girvan, M., & Newman, M. E. J. (2002).
Community structure in social and biological networks.
Proceedings of the National Academy of Sciences, 99, 7821–7826.
https://doi.org/10.1073/pnas.122653799
"""


import networkx as nx
from collections import defaultdict, deque
import itertools
from multiprocessing import Pool, cpu_count
import math
import logging
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt
import time
import numpy as np


class GirvanNewmanOptimized:
    """Optimized implementation of the Girvan–Newman community detection algorithm."""

    def __init__(self, graph: nx.Graph, num_processes: int = None):
        """
        Initialize the optimized Girvan–Newman algorithm.

        Args:
            graph (nx.Graph): The input network graph.
            num_processes (int): Number of processes to use for parallelization.
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.num_processes = num_processes if num_processes else max(1, cpu_count() - 1)
        self.communities_history: List[List[Set]] = []
        self.modularity_history: List[float] = []
        self.edge_removal_order: List[Tuple] = []
        self.runtime = 0.0
        self.node_to_index = {node: i for i, node in enumerate(self.graph.nodes())}
        self.index_to_node = {i: node for node, i in self.node_to_index.items()}

        # Build adjacency matrix for faster operations
        self.adjacency_matrix = self._create_adjacency_matrix()

        # Setup logging for runtime information
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        self.logger = logging.getLogger('GirvanNewmanOptimized')

    def _create_adjacency_matrix(self):
        """Create adjacency matrix for efficient numerical operations."""
        n = len(self.graph.nodes())
        adj_matrix = np.zeros((n, n), dtype=np.float32)

        for u, v, data in self.graph.edges(data=True):
            i, j = self.node_to_index[u], self.node_to_index[v]
            weight = data.get('weight', 1.0)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight

        return adj_matrix

    def _brandes_algorithm(self, source: int) -> np.ndarray:
        """Optimized version of Brandes' algorithm for betweenness computation."""
        n = self.adjacency_matrix.shape[0]
        betweenness = np.zeros((n, n), dtype=np.float32)

        S = []  # Stack
        P = [[] for _ in range(n)]  # Predecessors
        sigma = np.zeros(n, dtype=np.float32)
        sigma[source] = 1.0
        d = np.full(n, -1, dtype=np.int32)
        d[source] = 0
        Q = deque([source])

        # BFS phase
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in range(n):
                if self.adjacency_matrix[v, w] > 0:
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

        # Accumulation phase
        delta = np.zeros(n, dtype=np.float32)
        while S:
            w = S.pop()
            for v in P[w]:
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                betweenness[v, w] += c
                betweenness[w, v] += c
                delta[v] += c

        return betweenness

    def _calculate_edge_betweenness_parallel(self) -> Dict[Tuple, float]:
        """Parallel computation of edge betweenness for all nodes."""
        n = len(self.node_to_index)
        total_betweenness = np.zeros((n, n), dtype=np.float32)

        with Pool(self.num_processes) as pool:
            results = pool.map(self._brandes_algorithm, range(n))

        for result in results:
            total_betweenness += result

        betweenness_dict = {}
        for i in range(n):
            for j in range(i + 1, n):
                if self.adjacency_matrix[i, j] > 0:
                    edge = (self.index_to_node[i], self.index_to_node[j])
                    betweenness_dict[edge] = round(total_betweenness[i, j] / 2, 6)

        return betweenness_dict

    def _calculate_modularity_fast(self, communities: List[Set]) -> float:
        """Efficient modularity calculation using degree precomputation."""
        m = self.original_graph.number_of_edges()
        if m == 0:
            return 0.0

        degrees = dict(self.original_graph.degree())
        total_degree = sum(degrees.values())

        q = 0.0
        for community in communities:
            sum_in = sum(1 for i, j in itertools.combinations(community, 2)
                         if self.original_graph.has_edge(i, j))
            sum_total = sum(degrees[node] for node in community)
            q += (sum_in / m) - (sum_total / (2 * m)) ** 2

        return round(q, 6)

    def _remove_highest_betweenness_edge(self) -> bool:
        """Remove the edge with the highest betweenness."""
        edge_betweenness = self._calculate_edge_betweenness_parallel()

        if not edge_betweenness:
            return False

        max_edge = max(edge_betweenness.items(), key=lambda x: x[1])[0]
        self.graph.remove_edge(*max_edge)

        i, j = self.node_to_index[max_edge[0]], self.node_to_index[max_edge[1]]
        self.adjacency_matrix[i, j] = 0
        self.adjacency_matrix[j, i] = 0

        self.edge_removal_order.append((max_edge, edge_betweenness[max_edge]))

        self.logger.debug(f"Removed edge {max_edge} with betweenness {edge_betweenness[max_edge]}")
        return True

    def detect_communities(self, early_stopping: bool = True,
                           max_iterations: int = None,
                           min_communities: int = 2) -> List[Set]:
        """
        Detect network communities using the optimized Girvan–Newman algorithm.

        Args:
            early_stopping (bool): Stop if modularity stops improving.
            max_iterations (int): Maximum number of iterations.
            min_communities (int): Minimum number of communities to detect.

        Returns:
            List[Set]: List of detected communities.
        """
        start_time = time.time()
        self.logger.info("Starting optimized Girvan–Newman algorithm")
        self.logger.info(f"Initial graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        max_modularity = -math.inf
        best_communities = []
        iteration = 0
        no_improvement_count = 0

        while True:
            iteration += 1
            if max_iterations and iteration > max_iterations:
                self.logger.info(f"Reached maximum iterations ({max_iterations})")
                break

            current_communities = list(nx.connected_components(self.graph))
            self.communities_history.append(current_communities)

            if len(current_communities) >= min_communities:
                current_modularity = self._calculate_modularity_fast(current_communities)
                self.modularity_history.append(current_modularity)

                self.logger.info(f"Step {iteration}: {len(current_communities)} communities, modularity={current_modularity}")

                if current_modularity > max_modularity:
                    max_modularity = current_modularity
                    best_communities = current_communities
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if early_stopping and no_improvement_count >= 3:
                    self.logger.info("Modularity has plateaued — stopping early")
                    break
            else:
                self.modularity_history.append(-math.inf)
                self.logger.info(f"Step {iteration}: too few communities to compute modularity")

            if not self._remove_highest_betweenness_edge():
                self.logger.info("No more edges to remove")
                break

        self.runtime = time.time() - start_time
        self.logger.info(f"Algorithm completed in {self.runtime:.2f} seconds")
        self.logger.info(f"Detected {len(best_communities)} communities with max modularity {max_modularity}")
        return best_communities


# ==============================
# Execution Example
# ==============================
if __name__ == "__main__":
    print("[+] Loading network data (first 1000 lines)...")

    G = nx.Graph()
    with open("weighted_disease_edges_one_mode.tsv", 'r', encoding='utf-8') as f:
        next(f)
        line_count = 0
        for line in f:
            if line_count >= 1000:
                break
            parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, weight_str = parts
                weight = int(weight_str)
                G.add_edge(node1, node2, weight=weight)
                line_count += 1

    print(f"[+] Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    gn = GirvanNewmanOptimized(G, num_processes=4)
    communities = gn.detect_communities(early_stopping=True, min_communities=2, max_iterations=20)

    print(f"[+] Found {len(communities)} communities")
    print("[+] Saving results and visualizations...")

    gn.save_communities_to_file(communities, 'communities_1000_detailed.txt')

    gn.visualize_modularity()
    plt.savefig('modularity_optimized_1000.png', dpi=300, bbox_inches='tight')

    gn.visualize_network(communities, show_labels=True, show_edges=True)
    plt.savefig('network_simple.png', dpi=300, bbox_inches='tight')

    print("[✓] Done. Results and plots have been saved successfully.")
