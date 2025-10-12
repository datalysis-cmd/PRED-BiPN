"""
Description:
------------
This script implements the LANS method for extracting the backbone 
of dense weighted networks. It reads a weighted edge list, builds 
a sparse adjacency matrix, calculates normalized (fractional) edge 
weights for each node, and identifies statistically significant 
edges according to a significance threshold (alpha). The output 
includes both a detailed TSV file and a CSV suitable for Gephi.

Input file format:
------------------
- Tab-separated file with three columns:
    Gene1    Gene2    Shared_Diseases
- Gene1 and Gene2: node identifiers
- Shared_Diseases: weight of the edge between the nodes (can be numeric)

Output:
-------
1. backbone_results.tsv
    - Columns: Gene1, Gene2, Weight, Normalized_Weight
2. backbone_gephi.csv
    - Columns: Source, Target, Weight (Gephi compatible)
3. Optional: additional outputs can be added as needed

Features:
---------
- Handles large datasets efficiently using chunked reading
- Avoids duplicate edges by using upper triangular representation
- Memory usage monitoring and configurable limits
- Parallel processing using multiple CPU cores
- Optional TEST_MODE for processing a subset of rows

Required Dependencies:
----------------------
- pandas   : Efficient data manipulation and reading files in chunks
- scipy    : Sparse matrix operations
- numpy    : Numerical calculations
- psutil   : Monitor memory usage
- tqdm     : Progress bars
- multiprocessing : Parallel processing
- resource : Optional memory limit adjustment on Linux/Unix

Install dependencies using pip:
--------------------------------
pip install pandas scipy numpy psutil tqdm

How to run:
-----------
1. Save this script as 'lans_backbone.py'
2. Place your input TSV file (e.g., 'weighted_disease_edges_one_mode.tsv') in the working directory
3. Adjust optional parameters if needed:
    - CHUNK_SIZE: rows per chunk for reading the file
    - ALPHA: significance threshold for edge filtering
    - N_CORES: number of CPU cores to use
    - MAX_MEMORY: memory limit in MB
    - TEST_MODE and TEST_ROWS for processing subsets during testing
4. Run the script from the command line:
    python lans_backbone.py
5. The script will:
    - Discover unique nodes
    - Build a sparse adjacency matrix
    - Calculate fractional edge weights
    - Extract statistically significant edges
    - Save results in TSV and Gephi CSV formats
    - Print memory usage and timing statistics

Notes:
------
- Memory usage is monitored; if MAX_MEMORY is exceeded, reduce CHUNK_SIZE
- The script automatically ensures the adjacency matrix is symmetric
- Parallel processing significantly speeds up normalization and backbone extraction
"""

import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, triu
import multiprocessing as mp
from functools import partial
import psutil
import os
import gc
import time
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Union
import resource
import numpy as np
import sys

warnings.filterwarnings('ignore')

# Settings
CHUNK_SIZE = 50000
ALPHA = 1.0 # More reasonable significance level
N_CORES = max(1, mp.cpu_count() - 1)
MAX_MEMORY = 16000  # Increased memory limit
TEST_MODE = True # Process entire file
TEST_ROWS = 10000

def increase_memory_limit():
    """Attempts to increase memory limits (Linux/Unix only)."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except Exception as e:
        print(f"Warning: Could not increase memory limit: {e}")

def memory_usage() -> float:
    """Returns current memory usage of the process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def check_memory():
    """Raises an error if memory usage exceeds MAX_MEMORY."""
    if memory_usage() > MAX_MEMORY:
        raise MemoryError(f"Memory limit exceeded {MAX_MEMORY}MB")

def sparse_row_normalize(matrix: csr_matrix, row_index: int) -> csr_matrix:
    """Normalizes a sparse matrix row so that the sum of its elements is 1."""
    row = matrix.getrow(row_index)
    row_sum = row.sum()
    return row / row_sum if row_sum > 0 else csr_matrix(row.shape)

def process_chunk(chunk: pd.DataFrame, gene_map: Dict[str, int]) -> coo_matrix:
    """Processes a chunk of data and returns a COO matrix without duplicate edges."""
    row = chunk['Gene1'].map(gene_map).values
    col = chunk['Gene2'].map(gene_map).values
    data = pd.to_numeric(chunk['Shared_Diseases'], errors='coerce').fillna(0).values
    data = np.clip(data, 0, None)
    
    # Ensure we don't create duplicate edges
    n_genes = len(gene_map)
    return coo_matrix((data, (row, col)), shape=(n_genes, n_genes))

def build_sparse_matrix(input_file: str) -> Tuple[csr_matrix, Dict[str, int]]:
    """Builds a sparse weight matrix from a TSV file."""
    
    print("\nPhase 1: Discovering genes...")
    genes = set()
    
    # First pass: discover all unique genes
    for chunk in tqdm(pd.read_csv(input_file, sep='\t', header=None,
                     names=['Gene1', 'Gene2', 'Shared_Diseases'],
                     chunksize=CHUNK_SIZE),
                     desc="Scanning file"):
        genes.update(chunk['Gene1'].unique())
        genes.update(chunk['Gene2'].unique())
        check_memory()

        if TEST_MODE:
            if len(genes) >= TEST_ROWS:
                break

    gene_map = {gene: idx for idx, gene in enumerate(genes)}
    n_genes = len(gene_map)
    print(f"Found {n_genes} unique genes | Memory: {memory_usage():.2f}MB")

    print("\nPhase 2: Building matrix...")
    matrix = lil_matrix((n_genes, n_genes), dtype='float32')
    rows_processed = 0

    # Second pass: build the matrix
    for chunk in tqdm(pd.read_csv(input_file, sep='\t', header=None,
                                names=['Gene1', 'Gene2', 'Shared_Diseases'],
                                chunksize=CHUNK_SIZE),
                     desc="Loading data"):
        chunk_matrix = process_chunk(chunk, gene_map)
        # Add only upper triangular to avoid duplicates
        matrix += triu(chunk_matrix, format='lil')
        gc.collect()

        if memory_usage() > MAX_MEMORY * 0.7:
            matrix = matrix.tocsr()
            gc.collect()

        if TEST_MODE:
            rows_processed += len(chunk)
            if rows_processed >= TEST_ROWS:
                break

    # Make matrix symmetric by adding its transpose
    matrix = matrix + matrix.T
    return matrix.tocsr(), gene_map

def process_node(args):
    """Worker function to normalize one node row of the matrix."""
    matrix, i = args
    row = sparse_row_normalize(matrix, i)
    return (i, row) if row.nnz > 0 else None

def calculate_fractional_weights(matrix: csr_matrix) -> csr_matrix:
    """Calculates the normalized (fractional) edge weights for each node."""
    print("\nCalculating fractional weights...")
    n_nodes = matrix.shape[0]
    fractional = lil_matrix(matrix.shape, dtype='float32')

    with mp.Pool(processes=N_CORES) as pool:
        tasks = [(matrix, i) for i in range(n_nodes)]
        results = list(tqdm(pool.imap(process_node, tasks, chunksize=100),
                      total=n_nodes, desc="Parallel processing"))

    for result in results:
        if result:
            i, row = result
            fractional[i] = row

    return fractional.tocsr()

def extract_backbone_edges(args: Tuple[csr_matrix, csr_matrix, List[int], float]) -> List[Tuple[int, int, float]]:
    """Extracts statistically significant edges for a subset of nodes."""
    matrix, fractional, nodes, alpha = args
    edges = []

    for i in nodes:
        row = fractional.getrow(i)
        if row.nnz == 0:
            continue

        weights = sorted(row.data, reverse=True)
        n_neighbors = len(weights)

        for j, p_ij in zip(row.indices, row.data):
            # Only process upper triangular to avoid duplicates
            if i < j:
                prob = sum(1 for w in weights if w > p_ij) / n_neighbors
                if prob < alpha:
                    edges.append((i, j, matrix[i, j]))

    return edges

def extract_backbone(matrix: csr_matrix, fractional: csr_matrix, alpha: float) -> List[Tuple[int, int, float]]:
    print("\nExtracting network backbone...")
    n_nodes = matrix.shape[0]
    edges = []

    step = max(1, n_nodes // (N_CORES * 4) + 1)
    chunks = [(matrix, fractional, list(range(i, min(i + step, n_nodes))), alpha)
              for i in range(0, n_nodes, step)]

    with mp.Pool(processes=N_CORES) as pool:
        for result in tqdm(pool.imap_unordered(extract_backbone_edges, chunks),
                         total=len(chunks), desc="Processing nodes"):
            edges.extend(result)
            check_memory()

    return edges

def save_results(edges: List[Tuple[int, int, float]], 
                gene_map: Dict[str, int], 
                fractional: csr_matrix,
                output_file: str = "backbone_results.tsv",
                gephi_file: str = "backbone_gephi.csv"):
    print("\nSaving results...")
    reverse_map = {idx: gene for gene, idx in gene_map.items()}
    
    # Save detailed TSV
    with open(output_file, 'w') as f:
        f.write("Gene1\tGene2\tWeight\tNormalized_Weight\n")
        for i, j, w in tqdm(edges, desc="Saving TSV"):
            f.write(f"{reverse_map[i]}\t{reverse_map[j]}\t{w}\t{fractional[i,j]:.6f}\n")
    
    # Save Gephi format
    pd.DataFrame([(reverse_map[i], reverse_map[j], w) for i,j,w in edges],
                columns=['Source', 'Target', 'Weight']).to_csv(gephi_file, index=False)

    print(f"Results saved to:\n- {output_file}\n- {gephi_file}")
    print(f"Total edges exported: {len(edges)}")

def main(input_file: str):
    start_time = time.time()
    increase_memory_limit()

    try:
        # Verify input file
        print(f"\nProcessing file: {input_file}")
        df_test = pd.read_csv(input_file, sep='\t', header=None, nrows=5)
        print("Input file preview:")
        print(df_test.head())
        
        if TEST_MODE:
            print(f"\nTEST MODE - Processing first {TEST_ROWS} rows")

        matrix, gene_map = build_sparse_matrix(input_file)
        print(f"Matrix built ({matrix.shape[0]} nodes, {matrix.nnz} edges) | Memory: {memory_usage():.2f}MB")

        fractional = calculate_fractional_weights(matrix)
        print(f"Fractional weights calculated | Memory: {memory_usage():.2f}MB")

        backbone_edges = extract_backbone(matrix, fractional, ALPHA)
        print(f"Found {len(backbone_edges)} significant edges (α={ALPHA})")

        save_results(backbone_edges, gene_map, fractional)

    except MemoryError as e:
        print(f"Error: {e}\nAdjust CHUNK_SIZE or MAX_MEMORY parameters")
    except Exception as e:
        print(f"Critical error: {str(e)}", file=sys.stderr)
        raise
    finally:
        print(f"\nTotal time: {(time.time()-start_time)/60:.2f} minutes")
        print(f"Final memory: {memory_usage():.2f} MB")

if __name__ == '__main__':
    mp.freeze_support()
    input_file = "weighted_disease_edges_one_mode.tsv"
    main(input_file)
