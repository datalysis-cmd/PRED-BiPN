
"""
    This script implements the High Salience Skeleton (HSS) extraction method for complex networks.
    It identifies the most "salient" (important) links in a network based on their participation in shortest paths.
    The core functionality includes:
    1. Building a network from input edge data, converting edge weights to inverse weights.
    2. Calculating shortest paths between all pairs of nodes using an optimized Dijkstra's algorithm.
    3. Computing a "salience score" for each edge, which quantifies its importance based on how many shortest paths it's part of.
    4. Leveraging multiprocessing to parallelize the salience calculation for efficiency.
    5. Extracting the HSS by filtering edges whose salience score exceeds a specified threshold.
    6. Loading data from TSV files with automatic header detection and data cleaning.
    7. Saving the resulting HSS to an output TSV file.
    The script aims to robustly classify network links into salient and non-salient groups,
    providing insights into the core structure of complex systems.

  Required Dependencies:
 - pandas: For data manipulation, especially loading and cleaning the input data from TSV files, and saving results.
 - collections.defaultdict: To create dictionaries with default values, useful for building the network adjacency list.
 - collections.Counter: For counting hashable objects, used to aggregate edge counts in shortest paths.
 - heapq: Provides an implementation of the heap queue algorithm, essential for the optimized Dijkstra's algorithm.
 - math: Provides mathematical functions, specifically math.inf for initializing distances in Dijkstra's.
 - os: Provides a way of using operating system dependent functionality, like creating directories for output files.
 - multiprocessing.Pool: For parallel processing, leveraging multiple CPU cores to speed up salience calculation.
 - multiprocessing.cpu_count: To determine the number of available CPU cores for parallelization.
 - functools.partial: To create new functions with some arguments pre-filled, useful for passing fixed network object to parallel tasks.
 - sys: Provides access to system-specific parameters and functions, used for printing error messages to stderr.
 - tqdm: For displaying a progress bar, useful for visualizing the progress of lengthy computations.
"""
import pandas as pd
from collections import defaultdict, Counter
import heapq
import math
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import sys 
from tqdm import tqdm 

class FastLinkSalience:
    def __init__(self, weight_threshold=1e-6):
        """
        Initializes the FastLinkSalience class.
        :param weight_threshold: Weight threshold. Edges with weight smaller than this will be ignored.
        """
        self.weight_threshold = weight_threshold
    
    def build_network(self, edges):
        """
        Builds a network as an adjacency dictionary from a list of edges.
        Each edge is represented as (source, destination, weight).
        The weight is converted to its reciprocal (1/weight) for use in shortest path algorithms.
        The network is considered undirected (both directions are added).
        :param edges: List of tuples (u, v, w) representing the network edges.
        :return: An adjacency dictionary where network[u][v] is the reciprocal weight of edge (u,v).
        """
        network = defaultdict(dict)
        for u, v, w in edges:
            if w > self.weight_threshold:
                inv_weight = 1/w
                network[u][v] = inv_weight
                network[v][u] = inv_weight  
        return network

    def dijkstra(self, network, start):
        """
        Optimized implementation of Dijkstra's algorithm to find the shortest paths
        from a starting node to all other nodes in the network.
        Uses a min-heap for optimization.
        :param network: The network as an adjacency dictionary (with reciprocal weights).
        :param start: The starting node for path calculation.
        :return: A dictionary where predecessors[node] is the list of predecessors of 'node' in the shortest path.
                 In this simple implementation, it contains only the immediate predecessor.
        """
        distances = {node: math.inf for node in network}
        distances[start] = 0
        predecessors = {node: [] for node in network}
        heap = [(0, start)]
        visited = set()
        
        while heap:
            current_dist, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor, weight in network[current].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = [current]
                    heapq.heappush(heap, (distance, neighbor))
        
        return predecessors


    def compute_for_node(self, network, node): 
        """
        Helper function for parallel processing. Calculates how many times each edge
        belongs to shortest paths starting from a specific node (`node`).
        :param network: The network as an adjacency dictionary.
        :param node: The starting node for shortest path calculation.
        :return: A Counter that stores the number of times each edge participates in shortest paths.
        """
        pred = self.dijkstra(network, node)
        edge_counts = Counter()
        
        for target in pred:
            if target == node:
                continue
            current = target
            # Ensure current is in pred and pred[current] is not empty
            while current in pred and pred[current]: 
                predecessor = pred[current][0]
                edge = tuple(sorted((predecessor, current)))
                edge_counts[edge] += 1
                current = predecessor
        
        return edge_counts

    def compute_salience(self, edges, processes=None):
        """"
        Parallel computation of the salience of all edges in the network.
        Salience is defined as the fraction of shortest paths in which an edge participates,
        divided by the total number of nodes (N).
        :param edges: List of tuples (u, v, w) representing the network edges.
        :param processes: The number of processes (CPU cores) to use for parallel processing.
                          If None, all available cores are used.
        :return: A dictionary where edge: salience_score for each edge.
        """
        network = self.build_network(edges)
        nodes = list(network.keys())
        N = len(nodes)
        
        if N == 0: 
            return {}

        if processes is None:
            processes = cpu_count()
        
        network_to_pass = network
        
        compute_for_node_partial = partial(self.compute_for_node, network_to_pass) 

        args_list = nodes 

        total_counts = Counter()
        with Pool(processes=processes) as pool:
            salience_results_iter = pool.imap_unordered(compute_for_node_partial, args_list)
            
            for counter in tqdm(salience_results_iter, total=len(nodes), desc="Calculate Salience"):
                total_counts.update(counter)
        
        return {edge: count/N for edge, count in total_counts.items()}

def load_data_fast(file_path, test_mode=False, max_rows=1000000):
    """
    Loads edge data from a TSV (tab-separated values) file.
    Automatic header detection and data cleaning.
    :param file_path: The path to the input file.
    :param test_mode: If True, loads only a maximum number of rows for testing.
    :param max_rows: The maximum number of rows to load in test_mode.
    :return: List of tuples (gene1, gene2, weight) representing the edges.
    """
    edges = []
    chunksize = 1000
    
    has_header = False
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.lower().startswith(('gene', 'shared', 'source', 'target', 'node1', 'node2')):
                has_header = True
    except Exception as e:
        print(f"Warning: Unable to read first line of file '{file_path}'. It is assumed that there is no header. Error: {e}", file=sys.stderr)
        has_header = False 

    col_names = ['gene1', 'gene2', 'shared_diseases'] 

    for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunksize,
                           nrows=max_rows if test_mode else None,
                           header=None, 
                           names=col_names, 
                           skiprows=1 if has_header else None):
        
        chunk = chunk.dropna(subset=['gene1', 'gene2', 'shared_diseases'])
        
        chunk['gene1'] = chunk['gene1'].astype(str).str.strip()
        chunk['gene2'] = chunk['gene2'].astype(str).str.strip()
        
        chunk['weight'] = pd.to_numeric(chunk['shared_diseases'], errors='coerce')
        
        chunk = chunk.dropna(subset=['weight'])
        
        chunk = chunk[chunk['weight'] > 0]
        
        edges.extend(chunk[['gene1', 'gene2', 'weight']].itertuples(index=False, name=None))
    
    return edges

def save_results(results, output_file):
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    pd.DataFrame(results, columns=['gene1', 'gene2', 'weight', 'salience'])\
      .to_csv(output_file, sep='\t', index=False)

def main_optimized(input_file, output_file, test_mode=False, threshold=0.9, processes=None):
    """
    Main function that executes the High Salience Subgraph (HSS) analysis.
    :param input_file: The path to the input file with edges.
    :param output_file: The path to the output file for HSS results.
    :param test_mode: If True, runs in test mode (loads fewer rows).
    :param threshold: The salience threshold. Edges with salience >= threshold are kept in the HSS.
    :param processes: The number of processes for parallel execution.
    """
    print("Loading Data...")
    try:
        edges = load_data_fast(input_file, test_mode=test_mode)
        if not edges:
            print("No valid edges were found in the input file. Check the file format and data.")
            return
        
        print(f"Edges {len(edges):,} processing...")
        analyzer = FastLinkSalience()
        
        salience = analyzer.compute_salience(edges, processes=processes)
        
        print("Creating HSS final results...")
        results = []
        for u, v, w in edges:
            edge_key = tuple(sorted((u, v)))
            if salience.get(edge_key, 0) >= threshold:
                results.append([u, v, w, salience[edge_key]])
        
        print("Saving results...")
        save_results(results, output_file)
        
        original = len(edges)
        hss = len(results)
        reduction = 0.0
        if original > 0:
            reduction = 100 * (original - hss) / original
        
        print(f"\nResults:")
        print(f"- Initial Edges: {original:,}")
        print(f"- Edges HSS: {hss:,}")
        print(f"- Redution: {reduction:.1f}%")
        print("Analysis of HSS completed.")
    
    except Exception as e:
        print(f"Error while processing: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr) 

if __name__ == "__main__":
    input_path = "weighted_gene_edges_one_mode.tsv"
    output_path = "hss_filter_disease_edges_results.tsv"
    
    main_optimized(input_path, output_path, test_mode=False, processes=cpu_count())
