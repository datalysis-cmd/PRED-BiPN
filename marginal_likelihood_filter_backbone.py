"""
This code implements the Marginal Likelihood Filter (MLF) for pruning edges in weighted complex networks,
as described in the article:
"Unwinding the hairball graph: Pruning algorithms for weighted complex networks"
by Navid Dianati (PHYSICAL REVIEW E 93, 012304 (2016)).

Purpose:
The primary goal is to identify the statistically most significant edges in a weighted network
(where edge weights are integers and count the occurrences of an "event" relating the nodes)
and extract a subgraph consisting of those edges. This helps in "unwinding" "hairball" graphs,
removing "noise" and revealing the most important structures.

How it works:
1.  **Null Model:** It assumes that unit edges are randomly assigned between nodes
    with probabilities proportional to the product of their weighted degrees (strengths). In other words,
    nodes with higher degrees are more likely to be connected to one another purely by chance.
2.  **p-value Calculation:** For each edge, a p-value is calculated using the binomial distribution.
    This p-value expresses the probability of observing the specific edge weight (or a larger one)
    by chance, given the weighted degrees of the two connected nodes and the total weight of the network.
    A low p-value indicates that the edge is statistically significant and less likely to have occurred randomly.
3.  **Filtering:** All edges whose p-value is higher than a predefined threshold `alpha` (e.g., 0.05) are
    removed. The remaining edges are considered the most significant.
4.  **Chunk Processing:** For efficient handling of large datasets, the code reads and processes data
    in chunks, thus avoiding memory overload.
5.  **Parallel Computation:** The p-value calculation is performed in parallel using all available CPU cores,
    significantly speeding up the process for large networks.
6.  **Sparse Matrix Output:** The filtered graph is optionally saved as a sparse matrix for memory efficiency.
7.  **Test Mode:** Includes a test mode to process only a subset of the data for quick debugging and testing.
"""


from scipy.stats import binom, poisson
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.sparse import dok_matrix, save_npz
from multiprocessing import Pool, cpu_count
import warnings
import os
from math import ceil

# Global settings
warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "0"

class MLFProcessor:
    def __init__(self, alpha=0.05, chunk_size=50000, test_mode=False, test_size=10000):
        """
        Initializes the MLFProcessor with parameters for significance, chunking, and testing.

        Args:
            alpha (float): The significance threshold for filtering edges (p-value < alpha).
            chunk_size (int): The number of rows (edges) to process at once from the input file.
            test_mode (bool): If True, only processes a limited number of edges defined by test_size.
            test_size (int): The maximum number of edges to process when test_mode is True.
        """
        
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.test_mode = test_mode
        self.test_size = test_size
    
    def load_data(self, file_path):
        """Load data with test mode support"""
        if self.test_mode:
            print(f"TEST MODE: Loading first {self.test_size} rows...")
            return pd.read_csv(file_path, sep="\t", nrows=self.test_size, chunksize=self.chunk_size)
        else:
            print("Loading full dataset in chunks...")
            return pd.read_csv(file_path, sep="\t", chunksize=self.chunk_size)
    
    @staticmethod
    def calculate_p_value(args):
        """
        Calculates the p-value using the Poisson approximation for large networks.
        """
        w_ij, s_i, s_j, T = args
        
        # Handle edge cases
        if T <= 0 or s_i <= 0 or s_j <= 0 or w_ij <= 0:
            return 1.0

        # Calculate probability according to MLF formula
        p_ij = (s_i * s_j) / (2 * T ** 2)
        
        # Calculate the mean (lambda) for the Poisson approximation
        lambda_val = T * p_ij
        
        # Use Poisson approximation for large T or small p
        # Poisson is a good approximation of Binomial when T is large and p is small
        # A common rule of thumb is T*p < 10, but it works well even for larger values.
        # Given your large T, this is the most robust approach.
        try:
            # Use survival function for P(X >= w_ij)
            p_val = poisson.sf(w_ij - 1, lambda_val)
            return p_val
        except Exception as e:
            print(f"Error in p-value calculation: {e}, args: {args}")
            return np.nan

    def process_graph(self, file_path, output_file):
        """
        Main method to process the graph
        """
        
        # First, check the column names in the file
        print("Reading file header to detect column names...")
        with open(file_path, 'r') as f:
            header = f.readline().strip().split('\t')
        print(f"Columns detected: {header}")
        
        # Assume first two columns are nodes, third is weight
        node1_col, node2_col, weight_col = header[0], header[1], header[2]
        print(f"Using columns: {node1_col}, {node2_col}, {weight_col}")

        # FIRST PASS: Calculate strengths
        print("\n=== FIRST PASS ===")
        node_index = {}
        current_idx = 0
        strength = []
        total_weight = 0
        edge_count = 0
        
        for chunk in self.load_data(file_path):
            # Use the detected weight column name
            chunk[weight_col] = pd.to_numeric(chunk[weight_col], errors="coerce").fillna(1).astype(int)
            
            for _, row in chunk.iterrows():
                # Use the detected node and weight column names
                g1 = row[node1_col]
                g2 = row[node2_col]
                w = row[weight_col]
                
                # Skip zero or negative weights
                if w <= 0:
                    continue
                
                # Update node index
                if g1 not in node_index:
                    node_index[g1] = current_idx
                    strength.append(0)
                    current_idx += 1
                if g2 not in node_index:
                    node_index[g2] = current_idx
                    strength.append(0)
                    current_idx += 1
                
                # Update strengths
                idx1 = node_index[g1]
                idx2 = node_index[g2]
                strength[idx1] += w
                strength[idx2] += w
                total_weight += w
                edge_count += 1
                
                # Early stop in test mode
                if self.test_mode and edge_count >= self.test_size:
                    break
            
            if self.test_mode and edge_count >= self.test_size:
                break
        
        # Total weight should be sum of all edge weights
        total_weight /= 2
        strength = np.array(strength, dtype=np.float64)
        print(f"Processed {len(node_index)} nodes and {edge_count} edges")
        print(f"Total weight: {total_weight:.2f}")

        # SECOND PASS: Filter edges
        print("\n=== SECOND PASS ===")
        sparse_graph = dok_matrix((len(node_index), len(node_index)), dtype=np.float32)
        results = []
        processed_count = 0
        
        for chunk in self.load_data(file_path):
            # Use the detected weight column name
            chunk[weight_col] = pd.to_numeric(chunk[weight_col], errors="coerce").fillna(1).astype(int)
            
            args = []
            valid_rows = []
            for _, row in chunk.iterrows():
                g1 = row[node1_col]
                g2 = row[node2_col]
                w = row[weight_col]
                
                # Skip zero or negative weights
                if w <= 0:
                    continue
                
                # Skip nodes not found in first pass
                if g1 not in node_index or g2 not in node_index:
                    continue
                
                s_i = strength[node_index[g1]]
                s_j = strength[node_index[g2]]
                
                args.append((w, s_i, s_j, total_weight))
                valid_rows.append(row)
                
                if self.test_mode and processed_count >= self.test_size:
                    break
                
                processed_count += 1
            
            # Parallel processing
            if args:
                with Pool(cpu_count()) as pool:
                    p_values = pool.map(self.calculate_p_value, args)
                
                # Build sparse matrix and results
                for row, p_val in zip(valid_rows, p_values):
                    g1 = row[node1_col]
                    g2 = row[node2_col]
                    w = row[weight_col]
                    
                    if p_val < self.alpha:
                        i, j = node_index[g1], node_index[g2]
                        sparse_graph[i, j] = w
                        sparse_graph[j, i] = w  # Undirected
                        results.append((g1, g2, w, p_val))
                    
                    # Debug output for first few edges
                    if len(results) < 10:
                        print(f"Edge: {g1}-{g2}, weight: {w}, strength_i: {strength[node_index[g1]]:.2f}, strength_j: {strength[node_index[g2]]:.2f}, p-value: {p_val:.6f}")
            
            if self.test_mode and processed_count >= self.test_size:
                break

        # Save results
        print("\n=== SAVING RESULTS ===")
        print(f"Found {len(results)} significant edges out of {processed_count} processed")
        
        if len(results) > 0:
            if not self.test_mode:
                save_npz("filtered_graph_sparse.npz", sparse_graph.tocsr())
                print("Saved sparse matrix: filtered_graph_sparse.npz")
            
            df_results = pd.DataFrame(results, columns=[node1_col, node2_col, "Weight", "p_value"])
            df_results.to_csv(output_file, sep="\t", index=False)
            print(f"Saved {len(results)} filtered edges to {output_file}")
        else:
            print("No significant edges found! Try adjusting the alpha threshold.")
        
        print(f"Sparse matrix stats: {sparse_graph.shape} shape, {sparse_graph.nnz} non-zero elements")

if __name__ == "__main__":
    # Configuration
    input_file = "weighted_disease_edges_one_mode.tsv"
    output_file = "mlf_diseases_filtered.tsv" if True else "filtered_edges_FULL.tsv"
    
    # Initialize processor in test mode (first 10,000 edges)
    processor = MLFProcessor(
        alpha=0.05,
        chunk_size=5000,  # Smaller chunks for testing
        test_mode=False,   # Enable test mode
        test_size=1000000   # Process only first 10,000 edges
    )
    
    # Run processing
    processor.process_graph(
        file_path=input_file,
        output_file=output_file
    )
