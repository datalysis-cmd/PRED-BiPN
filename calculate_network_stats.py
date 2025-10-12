"""
Network Statistics Calculator
------------------------------

This script computes key network statistics for one or more edge list files.
It supports both unweighted (2-column) and weighted (3-column) TSV/CSV files.

For each input file, it calculates:
- Number of nodes
- Number of edges
- Graph density
- Average / minimum / maximum node degree
- Average clustering coefficient
- Average shortest path length (in the largest connected component)
- Graph diameter (in the largest connected component)

The results are saved to a CSV file named `network_statistics_filtered.csv`.

---------------------------------
USAGE:
---------------------------------
1. Place this script and your network files (e.g., `.tsv` or `.csv`) in the same folder.
2. Run the script:
   $ python network_stats.py
3. When prompted, enter your file names separated by commas:
   Example: DISPARITY_FILTER.tsv, SDSM.tsv, MY_NETWORK.csv

---------------------------------
REQUIRED LIBRARIES:
---------------------------------
You must have the following Python libraries installed:
    - pandas
    - networkx
    - numpy

To install them, run:
    pip install pandas networkx numpy
"""

import pandas as pd
import networkx as nx
import numpy as np
import os


def compute_network_statistics(file_path):
    """
    Compute key topological statistics for a given network file.

    Parameters
    ----------
    file_path : str
        Path to the input TSV/CSV file containing the edge list.

    Returns
    -------
    dict
        A dictionary with computed network statistics.
    """
    try:
        # Read the file (automatically detects separator)
        df = pd.read_csv(file_path, sep=None, engine='python', header=None)

        # Build edge list (weighted or unweighted)
        if df.shape[1] >= 3:
            edges = [(row[0], row[1], {'weight': row[2]}) for row in df.values]
        else:
            edges = [(row[0], row[1]) for row in df.values if len(row) >= 2]

        # Create the graph
        G = nx.Graph()
        G.add_edges_from(edges)

        # --- Basic stats ---
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = round(nx.density(G), 4)

        degrees = [d for _, d in G.degree()]
        avg_degree = round(np.mean(degrees), 2) if degrees else 0
        min_degree = round(np.min(degrees), 2) if degrees else 0
        max_degree = round(np.max(degrees), 2) if degrees else 0

        avg_clustering = round(nx.average_clustering(G), 4) if num_nodes > 1 else 0

        # --- Connectivity statistics ---
        if num_nodes > 1:
            largest_cc = max(nx.connected_components(G), key=len)
            G_lcc = G.subgraph(largest_cc)
            if len(G_lcc) > 1:
                avg_shortest_path = round(nx.average_shortest_path_length(G_lcc), 2)
                diameter = round(nx.diameter(G_lcc), 2)
            else:
                avg_shortest_path = 0
                diameter = 0
        else:
            avg_shortest_path = 0
            diameter = 0

        return {
            'file': os.path.basename(file_path),
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': avg_degree,
            'min_degree': min_degree,
            'max_degree': max_degree,
            'avg_shortest_path': avg_shortest_path,
            'diameter': diameter,
            'avg_clustering': avg_clustering
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def main():
    """Main function: handles user input and summary output."""
    print("Network Statistics Calculator")
    print("--------------------------------")
    print("Enter the names of your edge list files separated by commas.")
    print("Example: DISPARITY_FILTER.tsv, SDSM.tsv, MY_FILE.csv\n")

    input_files = input("Files: ").strip()

    if not input_files:
        print("No files provided.")
        return

    files = [f.strip() for f in input_files.split(",") if f.strip()]

    stats_list = []

    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue

        print(f"➡️ Processing file: {file}")
        stats = compute_network_statistics(file)
        if stats:
            stats_list.append(stats)

    if not stats_list:
        print("No statistics were computed.")
        return

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_list)
    print("\nResults:\n")
    print(stats_df)

    # Save to CSV
    output_file = "network_statistics_filtered.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"\nStatistics saved to file: {output_file}")


if __name__ == "__main__":
    main()
