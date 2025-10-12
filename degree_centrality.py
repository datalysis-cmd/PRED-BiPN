#!/usr/bin/env python3
"""
Weighted Degree Centrality Analysis for Gene Networks

This script computes the Weighted Degree Centrality for nodes in a gene interaction network.
The weights are given by the 'Shared_Diseases' column in the input TSV file.

Key Features:
- Processes large files using chunking to reduce memory usage.
- Uses multiprocessing to speed up computation of weighted degrees.
- Normalizes the weighted degree by dividing by N-1 (where N is the total number of nodes).
- Saves the results to a CSV file with explanatory comments.

Requirements:
- Python 3.x
- pandas (install via: pip install pandas)
- Multiprocessing (built-in Python library)
- os (built-in Python library)

How to run:
1. Place the input TSV file (with header: Gene1, Gene2, Shared_Diseases) in the same folder.
2. Adjust parameters at the top of the script (CHUNK_SIZE, TEST_LINES, INPUT_FILE).
3. Run the script:
   python weighted_degree.py

Outputs:
- weighted_degree_centrality_test.csv: contains Node, Weighted_Degree, Normalised_Weighted_Degree
  plus explanatory comments at the end of the file.

Reference:
----------
Zakariya Ghalmane, Chantal Cherif, Hocine Cherif & Mohammed El Hassouni,
"Extracting backbones in weighted modular complex networks",
Scientific Reports, (Year).
This paper inspired the approach used here for community detection
and edge-betweenness-based backbone extraction in weighted networks.

"""

import pandas as pd
from multiprocessing import Pool, cpu_count
import os

# -------------------------------
# Parameters (adjust as needed)
# -------------------------------
CHUNK_SIZE = 200000       # Number of rows per chunk when reading the file
TEST_LINES = 200000       # Number of rows to process for testing
INPUT_FILE = "weighted_gene_edges_one_mode.tsv"  # Input TSV file

# -------------------------------
# Function to process a chunk
# -------------------------------
def process_chunk(chunk):
    """
    Calculate the weighted degree for each node in a chunk of data.

    Args:
        chunk (DataFrame): a pandas DataFrame with columns 'Gene1', 'Gene2', 'Shared_Diseases'

    Returns:
        dict: dictionary of nodes and their accumulated weighted degree in this chunk
    """
    degree = {}
    for _, row in chunk.iterrows():
        u, v, weight = row['Gene1'], row['Gene2'], float(row['Shared_Diseases'])
        degree[u] = degree.get(u, 0) + weight
        degree[v] = degree.get(v, 0) + weight
    return degree

# -------------------------------
# Main computation function
# -------------------------------
def compute_weighted_degree_centrality(file_path, test_mode=False):
    """
    Compute Weighted Degree Centrality for all nodes using chunking and multiprocessing.

    Args:
        file_path (str): path to the TSV file
        test_mode (bool): if True, process only TEST_LINES rows for quick testing

    Returns:
        DataFrame: sorted dataframe of nodes and their weighted degree
    """
    print(f"[INFO] Starting weighted degree calculation from file: {file_path}")
    
    degree_counts = {}
    
    # Read file in chunks
    chunks = pd.read_csv(
        file_path,
        sep='\t',
        chunksize=CHUNK_SIZE,
        header=0,  # The file has header
        dtype={'Gene1': str, 'Gene2': str, 'Shared_Diseases': float}
    )
    
    # Use all available CPU cores
    with Pool(cpu_count()) as pool:
        for i, chunk in enumerate(chunks):
            if test_mode and i * CHUNK_SIZE >= TEST_LINES:
                print("[INFO] Test mode active: stopping after limited lines.")
                break
            
            # Compute weighted degree for this chunk in parallel
            chunk_degrees = pool.map(process_chunk, [chunk])
            
            # Accumulate results
            for d in chunk_degrees:
                for node, count in d.items():
                    degree_counts[node] = degree_counts.get(node, 0) + count
    
    # Convert results to DataFrame
    df_degree = pd.DataFrame(list(degree_counts.items()), columns=['Node', 'Weighted_Degree'])
    df_degree = df_degree.sort_values(by='Weighted_Degree', ascending=False)
    
    print(f"[INFO] Weighted degree calculation completed. Total nodes: {len(df_degree)}")
    return df_degree

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    print(f"[INFO] Processing the first {TEST_LINES} lines (test mode)...")
    df_result = compute_weighted_degree_centrality(INPUT_FILE, test_mode=True)
    
    # Normalize weighted degree
    total_nodes = len(df_result)
    if total_nodes > 1:
        df_result['Normalised_Weighted_Degree'] = df_result['Weighted_Degree'] / (total_nodes - 1)
        print("[INFO] Normalization completed.")
    
    # Save results to CSV
    output_file = "weighted_degree_centrality_test.csv"
    df_result.to_csv(output_file, index=False)
    
    # Append explanatory comments to the CSV file
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write("\n# Node: The node in the network (a gene in this case).\n")
        outfile.write("# Weighted_Degree: Sum of weights of all edges connected to this node.\n")
        outfile.write("# Normalised_Weighted_Degree: Weighted_Degree divided by N-1 (total nodes minus 1).\n")
    
    print(f"[INFO] Results saved to {output_file}")
    print("\n[INFO] First 10 records:")
    print(df_result.head(10))
