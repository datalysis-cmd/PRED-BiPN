#!/usr/bin/env python3
"""
===========================================================
H-Backbone Network Analysis for Gene Interaction Networks
===========================================================

Implementation of the method from:
Zhang et al. (2018) "Extracting h-Backbone as a Core Structure in Weighted Networks"
Scientific Reports 8:14356

Description:
------------
This script extracts the h-backbone from weighted gene interaction
networks, identifying core edges that are significant either via
h-strength (weight-based) or h-bridge (betweenness-based) metrics.
It calculates network-wide metrics, selects edges satisfying
h-backbone criteria, and outputs a filtered network.

Input:
------
- Tab-separated file (TSV) without headers or with 3 columns:
    gene1    gene2    weight
- gene1, gene2: Node identifiers
- weight: Numeric positive weight of the edge

Output:
-------
1. gene_backbone_results.tsv
    - Contains filtered backbone edges and analysis metadata

Features:
---------
- Computes h-strength: h-index of edge weights
- Computes h-bridge: h-index of edge betweenness centrality
- Edge sampling option for large networks
- Safe handling of missing/invalid weights
- Output includes metadata with original and backbone network stats
- Can limit number of processed rows to reduce memory usage

Required Dependencies:
----------------------
- pandas      : Data handling
- networkx    : Graph construction and analysis
- numpy       : Numerical operations
- warnings    : Suppress runtime warnings
- heapq       : Priority queue for Dijkstra's algorithm
- collections : defaultdict for counting
- typing      : Type hints

Install dependencies using pip:
--------------------------------
pip install pandas networkx numpy

Usage:
------
1. Save this script as 'h_backbone.py'
2. Place your input TSV file in the working directory
3. Run from the command line with required input file:
   python h_backbone.py input_file.tsv

Optional Arguments:
------------------
-m, --max_rows       : Maximum number of rows to process (default=200000)
-s, --sample_size    : Number of nodes to sample for betweenness calculation (default=None)
-o, --output         : Output TSV file path (default="gene_backbone_results.tsv")

Notes:
------
- Only positive edge weights are retained
- Large networks may benefit from setting sample_size < total nodes
- Output TSV contains metadata and edge list suitable for downstream analysis

References
----------
Zhang, R. J., Stanley, H. E., & Ye, F. Y. (2018).
Extracting h-backbone as a core structure in weighted networks.
Scientific Reports, 8, 1–7.
https://doi.org/10.1038/s41598-018-32430-1
"""
#!/usr/bin/env python3
"""
H-Backbone Network Analysis for Gene Interaction Networks
Implementation of the method from:
Zhang et al. (2018) "Extracting h-Backbone as a Core Structure in Weighted Networks"
Scientific Reports 8:14356
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
import heapq
import numpy as np
import warnings
from typing import Tuple, Dict, Set

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module='pytz')
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_network_data(file_path: str, max_rows: int = 200000) -> pd.DataFrame:
    """
    Load gene network data from TSV file
    Args:
        file_path: Path to TSV file (gene1, gene2, weight)
        max_rows: Maximum number of rows to read
    Returns:
        DataFrame with validated network data
    """
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['gene1', 'gene2', 'weight'],
            nrows=max_rows,
            dtype={'gene1': str, 'gene2': str}
        )
        
        # Convert weight to numeric, handling errors
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
        
        # Drop rows with invalid weights or missing data
        df = df.dropna(subset=['gene1', 'gene2', 'weight'])
        df = df[df['weight'] > 0]  # Only keep positive weights
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading network data: {str(e)}")

def build_weighted_network(df: pd.DataFrame) -> nx.Graph:
    """
    Construct weighted network graph from DataFrame
    Args:
        df: DataFrame with gene1, gene2, weight columns
    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['gene1'], row['gene2'], weight=float(row['weight']))
    return G

def compute_edge_betweenness(
    G: nx.Graph, 
    sample_size: int = None,
    weight_attribute: str = 'weight'
) -> Dict[Tuple[str, str], float]:
    """
    Compute edge betweenness centrality with sampling option
    Args:
        G: NetworkX graph
        sample_size: Number of nodes to sample (None for all nodes)
        weight_attribute: Edge attribute to use as weight
    Returns:
        Dictionary of edge betweenness values
    """
    nodes = list(G.nodes())
    N = len(nodes)
    edge_counts = defaultdict(float)
    
    # Use sampling for large networks
    if sample_size and N > sample_size:
        nodes = np.random.choice(nodes, min(sample_size, N), replace=False)
    
    for source in nodes:
        # Dijkstra's algorithm for weighted shortest paths
        pred = {v: [] for v in G}
        dist = {v: float('inf') for v in G}
        sigma = {v: 0.0 for v in G}
        dist[source] = 0.0
        sigma[source] = 1.0
        
        heap = [(0.0, source)]
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v]:
                continue
                
            for w in G.neighbors(v):
                try:
                    edge_weight = G[v][w][weight_attribute]
                    if edge_weight <= 0:
                        continue
                        
                    new_dist = dist[v] + edge_weight
                    
                    if new_dist < dist[w]:
                        dist[w] = new_dist
                        pred[w] = [v]
                        sigma[w] = sigma[v]
                        heapq.heappush(heap, (new_dist, w))
                    elif new_dist == dist[w]:
                        pred[w].append(v)
                        sigma[w] += sigma[v]
                except KeyError:
                    continue
        
        # Backward accumulation of betweenness
        delta = defaultdict(float)
        bfs_order = sorted(
            [v for v in G if dist[v] < float('inf')],
            key=lambda x: -dist[x]
        )
        
        for w in bfs_order:
            for v in pred[w]:
                edge = tuple(sorted((v, w)))
                contribution = (sigma[v] / sigma[w]) * (1 + delta[w])
                edge_counts[edge] += contribution
                delta[v] += contribution
    
    return edge_counts

def compute_h_bridge(
    G: nx.Graph,
    sample_size: int = None
) -> Tuple[int, Dict[Tuple[str, str], float]]:
    """
    Compute h-bridge value for the network
    Args:
        G: NetworkX graph
        sample_size: Number of nodes to sample for betweenness calculation
    Returns:
        Tuple of (h_bridge value, dictionary of bridge values for edges)
    """
    edge_counts = compute_edge_betweenness(G, sample_size)
    N = G.number_of_nodes()
    
    # Calculate bridge values (b(v) = eb(v)/N)
    bridge_values = {e: count/N for e, count in edge_counts.items()}
    
    # Calculate h-index for bridge values
    sorted_bridges = sorted(bridge_values.values(), reverse=True)
    h_bridge = 0
    for i, val in enumerate(sorted_bridges, 1):
        if val >= i:
            h_bridge = i
        else:
            break
    
    return h_bridge, bridge_values

def compute_h_strength(G: nx.Graph) -> int:
    """
    Compute h-strength value for the network
    Args:
        G: NetworkX graph with weight attribute
    Returns:
        h-strength value
    """
    weights = []
    for _, _, d in G.edges(data=True):
        try:
            weight = float(d['weight'])
            if weight > 0:
                weights.append(weight)
        except (ValueError, KeyError):
            continue
    
    if not weights:
        return 0
    
    sorted_weights = sorted(weights, reverse=True)
    h_strength = 0
    for i, val in enumerate(sorted_weights, 1):
        if val >= i:
            h_strength = i
        else:
            break
    
    return h_strength

def extract_h_backbone(
    G: nx.Graph,
    h_bridge: int,
    bridge_values: Dict[Tuple[str, str], float],
    h_strength: int
) -> nx.Graph:
    """
    Extract h-backbone subgraph
    Args:
        G: Original network graph
        h_bridge: Computed h-bridge value
        bridge_values: Dictionary of bridge values for edges
        h_strength: Computed h-strength value
    Returns:
        h-backbone subgraph
    """
    backbone_edges = set()
    
    # Add edges meeting h-bridge criterion
    for edge, bridge in bridge_values.items():
        if bridge >= h_bridge:
            backbone_edges.add(edge)
    
    # Add edges meeting h-strength criterion
    for u, v, d in G.edges(data=True):
        try:
            if float(d['weight']) >= h_strength:
                backbone_edges.add((u, v))
        except (ValueError, KeyError):
            continue
    
    # Create backbone subgraph
    return G.edge_subgraph(backbone_edges).copy()

def analyze_gene_network(
    file_path: str,
    max_rows: int = 1000,
    sample_size: int = None
) -> Tuple[nx.Graph, dict]:
    """
    Complete h-backbone analysis pipeline
    Args:
        file_path: Path to gene network TSV file
        max_rows: Maximum number of rows to process
        sample_size: Sample size for betweenness calculation
    Returns:
        Tuple of (backbone graph, analysis statistics)
    """
    print("Loading and validating network data...")
    df = load_network_data(file_path, max_rows)
    
    if df.empty:
        raise ValueError("No valid edges found in the input file")
    
    print(f"Building network from {len(df)} valid edges...")
    G = build_weighted_network(df)
    
    print("\nCalculating network metrics:")
    print("- Computing h-strength...")
    h_strength = compute_h_strength(G)
    print(f"  h-strength: {h_strength}")
    
    print("- Computing h-bridge...")
    h_bridge, bridge_values = compute_h_bridge(G, sample_size)
    print(f"  h-bridge: {h_bridge}")
    
    print("\nExtracting h-backbone...")
    backbone = extract_h_backbone(G, h_bridge, bridge_values, h_strength)
    
    # Calculate statistics
    stats = {
        'original_nodes': G.number_of_nodes(),
        'original_edges': G.number_of_edges(),
        'backbone_nodes': backbone.number_of_nodes(),
        'backbone_edges': backbone.number_of_edges(),
        'node_reduction': 1 - (backbone.number_of_nodes() / G.number_of_nodes()),
        'edge_reduction': 1 - (backbone.number_of_edges() / G.number_of_edges()),
        'h_strength': h_strength,
        'h_bridge': h_bridge
    }
    
    return backbone, stats

def save_backbone_results(
    backbone: nx.Graph,
    stats: dict,
    output_file: str = "gene_backbone_results.tsv"
) -> None:
    """
    Save backbone results to TSV file
    Args:
        backbone: h-backbone graph
        stats: Analysis statistics
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        # Write metadata header
        f.write("# H-Backbone Network Results\n")
        f.write(f"# Original nodes: {stats['original_nodes']}\n")
        f.write(f"# Original edges: {stats['original_edges']}\n")
        f.write(f"# Backbone nodes: {stats['backbone_nodes']}\n")
        f.write(f"# Backbone edges: {stats['backbone_edges']}\n")
        f.write(f"# Node reduction: {stats['node_reduction']:.1%}\n")
        f.write(f"# Edge reduction: {stats['edge_reduction']:.1%}\n")
        f.write(f"# h-strength: {stats['h_strength']}\n")
        f.write(f"# h-bridge: {stats['h_bridge']}\n")
        f.write("# Edge List:\n")
        f.write("gene1\tgene2\tweight\n")
        
        # Write edge data
        for u, v, d in backbone.edges(data=True):
            f.write(f"{u}\t{v}\t{d['weight']}\n")

def main():
    """Main execution function"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract h-backbone from gene interaction network"
    )
    parser.add_argument(
        "input_file",
        help="TSV file with gene1, gene2, weight columns"
    )
    parser.add_argument(
        "-m", "--max_rows",
        type=int,
        default=200000,
        help="Maximum number of rows to process"
    )
    parser.add_argument(
        "-s", "--sample_size",
        type=int,
        default=None,
        help="Sample size for betweenness calculation (default: None for full calculation)"
    )
    parser.add_argument(
        "-o", "--output",
        default="gene_backbone_results.tsv",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"\nStarting h-backbone analysis for {args.input_file}")
        backbone, stats = analyze_gene_network(
            args.input_file,
            args.max_rows,
            args.sample_size
        )
        
        print("\nAnalysis Results:")
        print(f"- Original network: {stats['original_nodes']} nodes, {stats['original_edges']} edges")
        print(f"- Backbone network: {stats['backbone_nodes']} nodes, {stats['backbone_edges']} edges")
        print(f"- Node reduction: {stats['node_reduction']:.1%}")
        print(f"- Edge reduction: {stats['edge_reduction']:.1%}")
        print(f"- h-strength: {stats['h_strength']}")
        print(f"- h-bridge: {stats['h_bridge']}")
        
        print(f"\nSaving results to {args.output}...")
        save_backbone_results(backbone, stats, args.output)
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
