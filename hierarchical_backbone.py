"""
This program implements an optimized method for hierarchical backbone extraction from a bipartite network.
It follows the methodology from scientific literature to identify statistically significant
edges and construct a directed, hierarchical network structure.

Key features of this implementation include:
- Use of **CSR matrices** for efficient storage and manipulation of sparse data.
- **Parallel processing** with the `multiprocessing` library to accelerate computationally intensive steps.
- **Memory-efficient data handling** to work with very large networks that might not fit entirely in RAM.

Dependencies:
To run this script, you need to install the following Python libraries. You can install them
using pip:
pip install pandas numpy networkx NEMtropy scipy
"""

import pandas as pd
import numpy as np
import time
import networkx as nx
from NEMtropy import BipartiteGraph
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
from scipy import sparse
import gc
import os
import tempfile
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Helper functions for parallel processing ---

def _process_chunk_load(chunk):
    """
    Processes a chunk of the input file to identify unique nodes from both partitions (A and B).
    """
    # Ensures nodes are treated as strings
    node_A = chunk.iloc[:, 0].astype(str).unique()
    node_B = chunk.iloc[:, 1].astype(str).unique()
    return (set(node_A), set(node_B))

def _build_csr_chunk(chunk, node_A_to_idx, node_B_to_idx, shape):
    """
    Builds a sparse matrix chunk (coo_matrix) from a portion of the bipartite network data.
    """
    # Map nodes to indices
    mapped_rows = chunk.iloc[:, 0].map(node_A_to_idx)
    mapped_cols = chunk.iloc[:, 1].map(node_B_to_idx)

    # Create a single mask for valid entries in both columns
    valid_mask = ~mapped_rows.isna() & ~mapped_cols.isna()
    
    # Filter both rows and cols using the same mask
    rows = mapped_rows[valid_mask].astype(int)
    cols = mapped_cols[valid_mask].astype(int)
    
    data = np.ones(len(rows), dtype=np.int8)
    return coo_matrix((data, (rows, cols)), shape=shape)


def _process_z_score_chunk(chunk, node_params, other_side_params, z_threshold):
    """
    Calculates the Z-scores for a chunk of edges in the projected network.
    """
    # 1. Use other_side_params to calculate sum_squares/cubes for the expected value/variance
    sum_other_side_sq = np.sum(other_side_params**2)
    sum_other_side_cube = np.sum(other_side_params**3)

    rows, cols, data = chunk
    # 2. Use node_params for the nodes in the projected network (rows and columns)
    row_params = node_params[rows]
    col_params = node_params[cols]
    
    # Calculate expected value and variance (BiCM formulas)
    expected = row_params * col_params * sum_other_side_sq
    variance = expected + (row_params * col_params * sum_other_side_cube)
    std_dev = np.sqrt(variance)
    
    # Avoid division by zero
    std_dev[std_dev == 0] = 1e-10 
    
    z_scores = (data - expected) / std_dev
    
    # Mask to filter edges based on the z_threshold
    mask = z_scores > z_threshold
    
    return (rows[mask], cols[mask], data[mask], z_scores[mask])

def _process_degree_chunk(chunk, num_nodes):
    """Calculates the degrees of nodes for a chunk of edges."""
    rows, data = chunk
    chunk_degrees = np.zeros(num_nodes)
    np.add.at(chunk_degrees, rows, data)
    return chunk_degrees

def _process_alpha_chunk(chunk, degrees, k_max):
    """Calculates the hierarchical strength (alpha) for a chunk of edges."""
    rows, cols, data = chunk
    # Avoid division by zero
    row_degrees = degrees[rows] + 1e-10
    col_degrees = degrees[cols] + 1e-10
    
    # Calculate P_uv and P_vu
    P_uv = data / col_degrees
    P_vu = data / row_degrees
    
    # Calculate the normalization factor
    normalization = np.minimum(row_degrees, col_degrees) / k_max
    alpha = normalization * (P_uv - P_vu)
    
    # Handle numerical errors
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    return alpha

def _process_edge_chunk(chunk, node_names):
    """Converts a chunk of edge indices back to node names."""
    rows, cols, alphas = chunk
    return [(node_names[r], node_names[c], a) for r, c, a in zip(rows, cols, alphas)]

# --- 1. Data Loading and Network Projection (Optimized) ---

def load_bipartite_network_optimized(file_path, chunksize=100000):
    """Loads a bipartite network in an optimized manner."""
    start_time = time.time()
    
    # First pass: find unique nodes using parallel processing
    chunks = pd.read_csv(file_path, sep='\t', header=None, chunksize=chunksize)
    with Pool(cpu_count()) as pool:
        results = pool.map(_process_chunk_load, chunks)
    
    node_A_set, node_B_set = set(), set()
    for a_set, b_set in results:
        node_A_set.update(a_set)
        node_B_set.update(b_set)
    
    node_A_list = sorted(node_A_set, key=str)
    node_B_list = sorted(node_B_set, key=str)
    
    node_A_to_idx = {node: idx for idx, node in enumerate(node_A_list)}
    node_B_to_idx = {node: idx for idx, node in enumerate(node_B_list)}
    
    # Second pass: build CSR matrix in parallel
    chunks = pd.read_csv(file_path, sep='\t', header=None, chunksize=chunksize)
    shape = (len(node_A_list), len(node_B_list))
    
    _build_csr_chunk_partial = partial(_build_csr_chunk, 
                                       node_A_to_idx=node_A_to_idx, 
                                       node_B_to_idx=node_B_to_idx, 
                                       shape=shape)
    
    with Pool(cpu_count()) as pool:
        coo_matrices = pool.map(_build_csr_chunk_partial, chunks)
    
    bipartite_matrix = sum(coo_matrices).tocsr()
    
    print(f"Network loaded in: {time.time() - start_time:.2f} seconds")
    print(f"Matrix shape: {bipartite_matrix.shape}, NNZ: {bipartite_matrix.nnz}")
    
    return bipartite_matrix, node_A_list, node_B_list

def project_bipartite_optimized(bipartite_matrix, node_list, projection_side='A'):
    """Projects the bipartite network to a one-mode network on the specified side."""
    start_time = time.time()
    
    if projection_side == 'B':
        # Projection on side B: B^T * B
        projected = bipartite_matrix.T.dot(bipartite_matrix)
    else:
        # Projection on side A: B * B^T
        projected = bipartite_matrix.dot(bipartite_matrix.T)
    
    # Convert to COO for easy filtering and removal of self-loops and duplicate edges
    projected_coo = projected.tocoo()
    upper_mask = projected_coo.row < projected_coo.col
    projected_coo = coo_matrix(
        (projected_coo.data[upper_mask],
         (projected_coo.row[upper_mask], projected_coo.col[upper_mask])),
        shape=projected_coo.shape
    )
    
    # Create a DataFrame with edges and weights
    edges = list(zip(
        [node_list[i] for i in projected_coo.row],
        [node_list[i] for i in projected_coo.col],
        projected_coo.data
    ))
    
    df_one_mode = pd.DataFrame(edges, columns=['Source', 'Target', 'weight'])
    
    print(f"Projection completed in: {time.time() - start_time:.2f} seconds")
    print(f"Projected edges: {len(df_one_mode)}")
    
    return df_one_mode, projected_coo

# --- 2. BiCM and Z-Score Computation (Parallelized) ---

def apply_bicm_parallel(bipartite_matrix):
    """
    Applies the Bipartite Configuration Model (BiCM) to calculate the parameters
    for each node in the network.
    """
    start_time = time.time()
    
    coo = bipartite_matrix.tocoo()
    edgelist = np.column_stack((coo.row, coo.col))
    
    B = BipartiteGraph()
    B.set_edgelist(edgelist.astype(np.int64))
    B.solve_tool(model="BiCM")
    
    # Get the parameters for each side
    lambdas = B.x  # For nodes on side A
    etas = B.y     # For nodes on side B
    
    # Free up memory
    del B, edgelist, coo
    gc.collect()
    
    print(f"BiCM solved in: {time.time() - start_time:.2f} seconds")
    return lambdas, etas

# ΔΙΟΡΘΩΣΗ: Προσθήκη projection_side για επίλυση του TypeError
def compute_z_scores_parallel(projected_coo, lambdas, etas, z_threshold, node_list, projection_side):
    """
    Performs a parallel computation of Z-scores for each edge in the projected network.
    """
    start_time = time.time()
    
    # Επιλέγουμε τις σωστές παραμέτρους BiCM ανάλογα με την πλευρά προβολής
    if projection_side == 'A':
        # Προβολή σε A. Projected nodes: A (lambdas). Other side: B (etas).
        node_params = lambdas
        other_side_params = etas
    else: # projection_side == 'B'
        # Προβολή σε B. Projected nodes: B (etas). Other side: A (lambdas).
        node_params = etas
        other_side_params = lambdas
        
    # Create chunks for parallel processing
    chunk_size = max(100000, projected_coo.nnz // (cpu_count() * 10))
    chunks = []
    for i in range(0, projected_coo.nnz, chunk_size):
        chunk = (
            projected_coo.row[i:i+chunk_size],
            projected_coo.col[i:i+chunk_size],
            projected_coo.data[i:i+chunk_size]
        )
        chunks.append(chunk)
    
    # Χρήση των ορισμένων μεταβλητών στην partial
    _process_z_score_chunk_partial = partial(_process_z_score_chunk, 
                                             node_params=node_params, 
                                             other_side_params=other_side_params, 
                                             z_threshold=z_threshold)
    
    with Pool(cpu_count()) as pool:
        results = pool.map(_process_z_score_chunk_partial, chunks)
    
    # Concatenate the results from all processes
    filtered_rows = np.concatenate([r[0] for r in results])
    filtered_cols = np.concatenate([r[1] for r in results])
    filtered_data = np.concatenate([r[2] for r in results])
    filtered_z_scores = np.concatenate([r[3] for r in results])
    
    # Create a sparse matrix from the filtered edges
    filtered_coo = coo_matrix(
        (filtered_data, (filtered_rows, filtered_cols)),
        shape=projected_coo.shape
    )
    
    print(f"Z-scores computed in: {time.time() - start_time:.2f} seconds")
    print(f"Edges after filtering: {filtered_coo.nnz}")
    
    return filtered_coo, filtered_z_scores

# --- 3. Hierarchical Strength Computation (Optimized) ---

def compute_degrees_parallel(filtered_coo, num_nodes):
    """Computes the degrees of the nodes from the filtered sparse matrix in parallel."""
    degrees = np.zeros(num_nodes)
    
    chunk_size = max(100000, filtered_coo.nnz // cpu_count())
    chunks = []
    for i in range(0, filtered_coo.nnz, chunk_size):
        chunks.append((
            filtered_coo.row[i:i+chunk_size],
            filtered_coo.data[i:i+chunk_size]
        ))
    
    _process_degree_chunk_partial = partial(_process_degree_chunk, num_nodes=num_nodes)
    
    with Pool(cpu_count()) as pool:
        results = pool.map(_process_degree_chunk_partial, chunks)
    
    degrees = sum(results)
    return degrees

def compute_hierarchical_strength_parallel(filtered_coo, degrees):
    """
    Computes the hierarchical strength (alpha) for each edge in parallel.
    """
    # Πρέπει να επιστρέψουμε έναν κενό πίνακα αν δεν υπάρχουν ακμές
    if filtered_coo.nnz == 0:
        return np.array([])
    
    k_max = degrees.max()
    num_nodes = len(degrees)
    
    chunk_size = max(100000, filtered_coo.nnz // cpu_count())
    chunks = []
    for i in range(0, filtered_coo.nnz, chunk_size):
        chunks.append((
            filtered_coo.row[i:i+chunk_size],
            filtered_coo.col[i:i+chunk_size],
            filtered_coo.data[i:i+chunk_size]
        ))
    
    _process_alpha_chunk_partial = partial(_process_alpha_chunk, degrees=degrees, k_max=k_max)
    
    with Pool(cpu_count()) as pool:
        alpha_chunks = pool.map(_process_alpha_chunk_partial, chunks)
    
    alpha = np.concatenate(alpha_chunks)
    return alpha

# --- 4. Hierarchical Network Construction ---

def build_hierarchical_network_optimized(filtered_coo, alpha, alpha_threshold, node_names):
    """
    Constructs the final hierarchical network by filtering edges based on the alpha threshold.
    """
    mask = alpha > alpha_threshold
    edges_rows, edges_cols, edges_alphas = filtered_coo.row[mask], filtered_coo.col[mask], alpha[mask]
    
    chunk_size = max(100000, len(edges_rows) // cpu_count())
    chunks = []
    for i in range(0, len(edges_rows), chunk_size):
        chunks.append((
            edges_rows[i:i+chunk_size],
            edges_cols[i:i+chunk_size],
            edges_alphas[i:i+chunk_size]
        ))
    
    _process_edge_chunk_partial = partial(_process_edge_chunk, node_names=node_names)
    
    with Pool(cpu_count()) as pool:
        edge_chunks = pool.map(_process_edge_chunk_partial, chunks)
    
    edges_flat = [edge for chunk in edge_chunks for edge in chunk]
    df_hierarchical = pd.DataFrame(edges_flat, columns=['Source', 'Target', 'weight'])
    
    print(f"Hierarchical network built with {len(df_hierarchical)} edges")
    return df_hierarchical

def remove_redundant_edges_optimized(df_hierarchical):
    """
    Applies transitive reduction to the hierarchical network to remove redundant
    edges and expose the underlying backbone structure.
    """
    if df_hierarchical.empty:
        return pd.DataFrame(columns=['Source', 'Target', 'weight'])
    
    start_time = time.time()
    
    G = nx.DiGraph()
    edge_weights = {}
    
    # Add edges in chunks
    chunk_size = 100000
    for i in range(0, len(df_hierarchical), chunk_size):
        chunk = df_hierarchical.iloc[i:i+chunk_size]
        for _, row in chunk.iterrows():
            G.add_edge(row.Source, row.Target)
            edge_weights[(row.Source, row.Target)] = row.weight
    
    # Perform transitive reduction
    G_reduced = nx.transitive_reduction(G)
    
    # Rebuild DataFrame with edge weights
    cleaned_edges = []
    for u, v in G_reduced.edges():
        cleaned_edges.append((u, v, edge_weights.get((u, v), 0)))
    
    df_cleaned = pd.DataFrame(cleaned_edges, columns=['Source', 'Target', 'weight'])
    
    print(f"Redundant edges removed in: {time.time() - start_time:.2f} seconds")
    print(f"Final edges: {len(df_cleaned)}")
    return df_cleaned

# --- Main Execution ---
def main():
    """
    The main function that orchestrates the entire process, from loading the data
    to saving the final backbone network.
    """
    print("=== Optimized Hierarchical Backbone Extraction ===")
    
    # Configuration parameters
    config = {
        'input_file': input("Enter the path of the bipartite file: ").strip(),
        'projection_side': 'B',  # 'A' or 'B'. Nodes from this side will form the one-mode network.
        'z_threshold_percentile': 95, # NOTE: This value is ignored below, but kept for printing consistency.
        'alpha_threshold_percentile': 95, # Alpha threshold is set using the 95th percentile of alpha values.
        'chunk_size': 1000000,
        'output_file': 'hierarchical_backbone_results.tsv'
    }
    
    try:
        # Step 1: Load the bipartite network
        print("\n[1/7] Loading bipartite network (optimized)...")
        bipartite_matrix, node_A_list, node_B_list = load_bipartite_network_optimized(
            config['input_file'], 
            chunksize=config['chunk_size']
        )
        
        # Step 2: Project to a one-mode network
        print(f"\n[2/7] Projecting to one-mode ({config['projection_side']}-nodes)...")
        node_list = node_A_list if config['projection_side'] == 'A' else node_B_list
        df_one_mode, projected_coo = project_bipartite_optimized(
            bipartite_matrix, 
            node_list, 
            config['projection_side']
        )
        
        # Step 3: Apply BiCM to find parameters
        print("\n[3/7] Applying BiCM (parallel)...")
        lambdas, etas = apply_bicm_parallel(bipartite_matrix)
        
        del bipartite_matrix
        gc.collect()
        
        #Statisticall Z-score threshold
        print("\n[4/7] Computing Z-scores for all projected edges...")
        
        _, all_z_scores = compute_z_scores_parallel(
            projected_coo, 
            lambdas, 
            etas, 
            z_threshold=-100.0,
            node_list=node_list,
            projection_side=config['projection_side']
        )

        z_threshold_correct = 1.64
        print(f"  Z-score threshold (Canonical Z > 1.64 for p < 0.05): {z_threshold_correct:.2f}")

        del all_z_scores # Δεν χρειάζεται πλέον
        gc.collect()

        print("\n[5/7] Filtering edges based on Z-score threshold...")
        filtered_coo, z_scores = compute_z_scores_parallel(
            projected_coo, 
            lambdas, 
            etas, 
            z_threshold_correct,
            node_list=node_list,
            projection_side=config['projection_side']
        )

        del projected_coo, df_one_mode
        gc.collect()
               
        if filtered_coo.nnz == 0:
            print("\n!!! Warning: No statistically significant edges found after Z-score filtering (Z > %.2f). !!!" % z_threshold_correct)
            df_final = pd.DataFrame(columns=['Source', 'Target', 'weight'])
            df_final.to_csv(config['output_file'], sep='\t', index=False)
            print(f"Results saved to: {config['output_file']} (empty)")
            print("\n=== Process completed successfully ===")
            return 

        # Step 6: Compute hierarchical strength (alpha)
        print("\n[6/7] Computing hierarchical strength (parallel)...")
        degrees = compute_degrees_parallel(filtered_coo, len(node_list))
        alpha = compute_hierarchical_strength_parallel(filtered_coo, degrees)
        
        # Step 7: Build the hierarchical network
        print("\n[7/7] Building hierarchical network...")
        alpha_threshold = np.percentile(alpha, config['alpha_threshold_percentile'])
        print(f"  Alpha threshold ({config['alpha_threshold_percentile']}% percentile): {alpha_threshold:.4f}")
        
        df_hierarchical = build_hierarchical_network_optimized(
            filtered_coo, 
            alpha, 
            alpha_threshold, 
            node_list
        )
        
        # Step 8: Remove redundant edges and save the final output
        print("\n[8/8] Removing redundant edges...")
        df_final = remove_redundant_edges_optimized(df_hierarchical)
        
        df_final.to_csv(config['output_file'], sep='\t', index=False)
        print(f"\n=== Process completed successfully ===")
        print(f"Results saved to: {config['output_file']}")
        print(f"Final edge count: {len(df_final)}")
        
    except Exception as e:
        print(f"\n!!! Fatal Error: {str(e)}")
        # If the error is not handled, print the full traceback for the user to debug
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
