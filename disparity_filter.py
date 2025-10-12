"""
Description:
    This script applies the Disparity Filter (Serrano et al., 2009) to a 
    weighted network to extract statistically significant edges. 
    It can handle very large networks by processing the data in chunks.

Input file format:
    Tab-separated file with three columns:
        Node1    Node2    Weight

Output file:
    CSV file with filtered edges and alpha values:
        Source,Target,Weight,Alpha

Required libraries:
    pip install pandas scipy tqdm psutil

How to run:
    1. Save this script as disparity_filter.py
    2. Run it from the command line:
        python disparity_filter.py
    3. Follow the prompts:
        - Enter input filename (e.g., network.tsv)
        - Enter output filename (e.g., filtered_network.csv)
        - Enter chunksize (recommended: 100000)
    4. The script will:
        a) Calculate node strength and degree
        b) Apply disparity filter to edges
        c) Save significant edges to output CSV

Notes:
    - Memory usage is monitored; reduce chunksize if memory usage is high.
    - Alpha threshold can be changed by passing alpha to DisparityFilter.
    
References
----------
Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009).
Extracting the multiscale backbone of complex weighted networks.
Proceedings of the National Academy of Sciences, 106(16), 6483–6488.
https://doi.org/10.1073/pnas.0808904106
"""

import pandas as pd
from scipy.stats import beta
import warnings
import os
from tqdm import tqdm
import psutil

warnings.filterwarnings("ignore")


class DisparityFilter:
    """
    DisparityFilter class applies the disparity filter to network edges.
    """
    def __init__(self, alpha=0.05):
        # significance level for filtering edges
        self.alpha = alpha
        # cache for beta survival function to speed up repeated calculations
        self.beta_cache = {}

    def get_beta_sf(self, p, k):
        """
        Calculate the beta survival function for normalized weight p
        and node degree k. Uses caching to speed up repeated calls.
        """
        if k <= 1:
            return 1.0
        if (p, k) not in self.beta_cache:
            self.beta_cache[(p, k)] = beta.sf(p, 1, k - 1)
        return self.beta_cache[(p, k)]

    def process_chunk(self, chunk, strength, degree):
        """
        Process a chunk of edges and return significant edges
        based on the disparity filter.
        
        Args:
            chunk: pd.DataFrame with columns Node1, Node2, Weight
            strength: dict with node strengths
            degree: dict with node degrees
        
        Returns:
            List of tuples (Node1, Node2, Weight, Alpha)
        """
        results = []
        for _, row in chunk.iterrows():
            u, v, weight = row['Node1'], row['Node2'], row['Weight']

            try:
                # Skip nodes with zero strength
                if strength.get(u, 0) == 0 or strength.get(v, 0) == 0:
                    continue

                # normalized weight and alpha for node u
                pij_u = weight / strength[u]
                alpha_ij_u = self.get_beta_sf(pij_u, degree[u])

                # normalized weight and alpha for node v
                pij_v = weight / strength[v]
                alpha_ij_v = self.get_beta_sf(pij_v, degree[v])

                # take minimum alpha as significance
                alpha_ij = min(alpha_ij_u, alpha_ij_v)

                if alpha_ij < self.alpha:
                    results.append((u, v, weight, alpha_ij))
            except Exception:
                continue
        return results


def calculate_network_stats(filename, chunksize=100000):
    """
    Calculate network statistics (strength and degree) for all nodes.
    
    Args:
        filename: path to input file
        chunksize: number of rows per chunk
    
    Returns:
        strength: dict of node total weights
        degree: dict of node degrees
    """
    print("Calculating network statistics...")

    strength = {}
    degree = {}

    # Count total rows for progress bar
    with open(filename) as f:
        total_rows = sum(1 for line in f) - 1  # subtract header if exists

    # Read file in chunks
    chunk_reader = pd.read_csv(filename, sep="\t", header=None,
                               names=["Node1", "Node2", "Weight"],
                               chunksize=chunksize)

    for chunk in tqdm(chunk_reader, total=total_rows/chunksize, desc="Processing network"):
        chunk['Weight'] = pd.to_numeric(chunk['Weight'], errors='coerce')
        chunk = chunk.dropna()

        for _, row in chunk.iterrows():
            u, v, weight = row['Node1'], row['Node2'], row['Weight']

            # Update node strength
            strength[u] = strength.get(u, 0) + weight
            strength[v] = strength.get(v, 0) + weight

            # Update node degree
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1

    return strength, degree


def filter_large_file(filename, output_file, strength, degree, chunksize=100000):
    """
    Apply disparity filter to the network in chunks and save results.
    
    Args:
        filename: path to input file
        output_file: path to output CSV
        strength: dict of node strengths
        degree: dict of node degrees
        chunksize: number of rows per chunk
    """
    print("Applying disparity filter...")

    disparity = DisparityFilter()

    # Count total rows for progress bar
    with open(filename) as f:
        total_rows = sum(1 for line in f) - 1

    with open(output_file, 'w') as f_out:
        f_out.write("Source,Target,Weight,Alpha\n")

        chunk_reader = pd.read_csv(filename, sep="\t", header=None,
                                   names=["Node1", "Node2", "Weight"],
                                   chunksize=chunksize)

        for chunk in tqdm(chunk_reader, total=total_rows/chunksize, desc="Filtering edges"):
            chunk['Weight'] = pd.to_numeric(chunk['Weight'], errors='coerce')
            chunk = chunk.dropna()

            results = disparity.process_chunk(chunk, strength, degree)

            for u, v, weight, alpha in results:
                f_out.write(f"{u},{v},{weight},{alpha:.6f}\n")

            # Check memory usage
            if psutil.virtual_memory().percent > 90:
                warnings.warn("High memory usage! Consider reducing chunksize.")


def main():
    print("="*50)
    print("LARGE NETWORK DISPARITY FILTER".center(50))
    print("="*50)

    filename = input("Enter input filename: ").strip()
    output_file = input("Enter output filename: ").strip()
    chunksize = int(input(f"Enter chunksize (recommended 100000) [100000]: ") or 100000)

    try:
        # Step 1: Calculate network statistics
        strength, degree = calculate_network_stats(filename, chunksize)

        print(f"\nFound {len(degree):,} nodes in the network")
        print(f"Total edges to process: {sum(degree.values())//2:,}")

        # Step 2: Filter and save results
        filter_large_file(filename, output_file, strength, degree, chunksize)

        print(f"\nResults saved to: {output_file}")
        print(f"Output file size: {os.path.getsize(output_file)/1024/1024:.2f} MB")

    except Exception as e:
        print(f"\nERROR: {str(e)}")


if __name__ == "__main__":
    main()
