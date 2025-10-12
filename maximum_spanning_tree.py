"""
This script implements Kruskal's algorithm to find the Maximum Spanning Tree (MST)
of a graph, specifically for gene interaction networks. It processes a tab-separated
value (TSV) file containing gene pairs and their interaction weights, identifying
the strongest set of connections that link all genes without forming cycles.

Dependencies:
- pandas: For efficient data loading and manipulation.
- numpy: Used implicitly by pandas for numerical operations.

To install the necessary dependencies, run the following command in your terminal:
pip install pandas numpy

References
----------
Gabow, H. N., & Tarjan, R. E. (1988).
A linear-time algorithm for finding a minimum spanning pseudoforest.
Information Processing Letters, 27(5), 259–263.
https://doi.org/10.1016/0020-0190(88)90089-0
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class UnionFind:
    """
    Union-Find data structure for efficient connectivity tracking.
    It supports operations to find the representative (root) of a set and to union two sets.
    """
    def __init__(self, size):
        """
        Initializes the Union-Find structure.
        Each element is initially in its own set (its parent is itself).
        Args:
            size (int): The number of elements to track.
        """
        self.parent = list(range(size))
   
    def find(self, x):
        """
        Finds the root (representative) of the set containing element x.
        Applies path compression for optimization, flattening the tree structure
        during traversal to speed up future lookups.
        Args:
            x (int): The element whose root needs to be found.
        Returns:
            int: The root parent of element x.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """
        Unites the sets containing elements x and y.
        If x and y are already in the same set, no action is taken.
        Otherwise, the root of one set is made a child of the root of the other set,
        effectively merging the two sets.
        Args:
            x (int): The first element.
            y (int): The second element.
        """
        """Union two sets."""
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.parent[y_root] = x_root

def load_data(filepath, test_mode=False, test_size=10000):
    """Load data with optional test mode for first N lines."""
    # Read only first test_size lines if in test mode
    nrows = test_size if test_mode else None
    df = pd.read_csv(filepath, sep='\t', names=['Gene1', 'Gene2', 'Weight'], nrows=nrows)
    
    # Clean and convert weights
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df = df.dropna(subset=['Weight'])
    
    # Create node to ID mapping
    nodes = sorted(set(df['Gene1'].unique()) | set(df['Gene2'].unique()))
    node_to_id = {node: idx for idx, node in enumerate(nodes)}
    
    # Convert to edge list with numeric IDs
    edges = []
    for _, row in df.iterrows():
        u = node_to_id[row['Gene1']]
        v = node_to_id[row['Gene2']]
        edges.append((u, v, row['Weight']))
    
    return edges, nodes

def kruskal_maxst(edges, num_nodes):
    """
    Implements Kruskal's algorithm to find the Maximal Spanning Tree (MaxST).
    It sorts edges by weight in descending order and iteratively adds edges
    to the MaxST if they do not form a cycle with previously added edges.
    Args:
        edges (list): A list of tuples, where each tuple is (u_id, v_id, weight).
        num_nodes (int): The total number of nodes in the graph.
    Returns:
        list: A list of tuples, representing the edges of the MaxST (u_id, v_id, weight).
    """
    # Sort edges by descending weight
    edges_sorted = sorted(edges, key=lambda x: -x[2])
    
    uf = UnionFind(num_nodes)
    maxst_edges = []
    
    for u, v, weight in edges_sorted:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            maxst_edges.append((u, v, weight))
        # Early stop when we have enough edges
        if len(maxst_edges) == num_nodes - 1:
            break
    
    return maxst_edges

def save_maxst_results(maxst_edges, nodes, output_file):
    """
    Saves the edges of the Maximal Spanning Tree to a TSV file.
    It converts numerical node IDs back to their original gene names.
    Args:
        maxst_edges (list): A list of tuples representing the MaxST edges (u_id, v_id, weight).
        nodes (list): A sorted list of unique gene names (used to map IDs back to names).
        output_file (str): The path to the output TSV file.
    """
    """Save MaxST edges with original node names."""
    id_to_node = {idx: node for idx, node in enumerate(nodes)}
    with open(output_file, 'w') as f:
        f.write("Source\tTarget\tWeight\n")  # Add header
        for u, v, weight in maxst_edges:
            f.write(f"{id_to_node[u]}\t{id_to_node[v]}\t{weight}\n")

if __name__ == "__main__":
    # Configuration
    input_file = "weighted_disease_edges_one_mode.tsv"
    output_file = "gene_maxst_edges_TEST_10k.tsv"  # Different name for test output
    
    # Run in test mode (first 10,000 lines)
    print(f"Loading first 10,000 lines from {input_file}...")
    edges, nodes = load_data(input_file, test_mode=True, test_size=200000)
    num_nodes = len(nodes)
    
    print(f"Running Kruskal's algorithm ({num_nodes} nodes, {len(edges)} edges)...")
    maxst_edges = kruskal_maxst(edges, num_nodes)
    
    print(f"Saving results to {output_file}...")
    save_maxst_results(maxst_edges, nodes, output_file)
    
    # Print statistics
    print("\n=== Test Results ===")
    print(f"Total nodes: {num_nodes}")
    print(f"Total edges processed: {len(edges)}")
    print(f"Edges in MaxST: {len(maxst_edges)}")
    print("\nSample edges in MaxST:")
    for u, v, w in maxst_edges[:5]:  # Show first 5 edges
        print(f"{nodes[u]} -- {nodes[v]} (weight: {w})")
