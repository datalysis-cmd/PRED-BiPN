import pandas as pd
from scipy.stats import beta
import warnings
import os
from tqdm import tqdm
import psutil

warnings.filterwarnings("ignore")

class DisparityFilter:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.beta_cache = {}


    # p: normalized weight, k: degree    
    def get_beta_sf(self, p, k):
        if k <= 1:
            return 1.0
        if (p, k) not in self.beta_cache:
            self.beta_cache[(p, k)] = beta.sf(p, 1, k - 1)
        return self.beta_cache[(p, k)]
    
    def process_chunk(self, chunk, strength, degree):
        results = []
        for _, row in chunk.iterrows():
            u, v, weight = row['Node1'], row['Node2'], row['Weight']
            
            try:
                # Skip nodes with zero strength
                if strength.get(u, 0) == 0 or strength.get(v, 0) == 0:
                    continue
                    
                pij_u = weight / strength[u]
                alpha_ij_u = self.get_beta_sf(pij_u, degree[u])
                
                pij_v = weight / strength[v]
                alpha_ij_v = self.get_beta_sf(pij_v, degree[v])
                
                alpha_ij = min(alpha_ij_u, alpha_ij_v)
                
                if alpha_ij < self.alpha:
                    results.append((u, v, weight, alpha_ij))
            except Exception as e:
                continue
        return results

def calculate_network_stats(filename, chunksize=100000):
    print("Calculating network statistics...")
    
    strength = {}
    degree = {}
    
    # Count total lines for progress bar
    with open(filename) as f:
        total_rows = sum(1 for line in f) - 1  # Subtract header if exists
    
    chunk_reader = pd.read_csv(filename, sep="\t", header=None,
                             names=["Node1", "Node2", "Weight"],
                             chunksize=chunksize)
    
    for chunk in tqdm(chunk_reader, total=total_rows/chunksize, desc="Processing network"):
        chunk['Weight'] = pd.to_numeric(chunk['Weight'], errors='coerce')
        chunk = chunk.dropna()
        
        for _, row in chunk.iterrows():
            u, v, weight = row['Node1'], row['Node2'], row['Weight']
            
            # Update strength
            strength[u] = strength.get(u, 0) + weight
            strength[v] = strength.get(v, 0) + weight
            
            # Update degree
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            
    return strength, degree

def filter_large_file(filename, output_file, strength, degree, chunksize=100000):
    print("Applying disparity filter...")
    
    disparity = DisparityFilter()
    
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
            
            # Memory check
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
