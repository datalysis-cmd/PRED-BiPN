"""
This Python script implements the Stochastic Degree-Sequence Model (SDSM) for extracting the statistical backbone of a bipartite network.
The method identifies statistically significant edges by comparing the observed network to a randomized null model that preserves the degree sequence of the original network. 
Edges with a significantly higher count than expected by chance are considered part of the "backbone."
To run this code, you'll need the following Python libraries:

-pandas: For data manipulation and file I/O.
-numpy: For numerical operations and statistical simulations.
-scipy: Specifically for the normal distribution function (norm).
You can install these dependencies using pip, the Python package installer. Open your terminal or command prompt and run the following command:

Bash: pip install pandas numpy scipy
"""
import pandas as pd
import numpy as np
import time
from collections import Counter
from scipy.stats import norm

def main():
    """
    Main function to orchestrate the entire SDSM backbone extraction process.
    """
    start_time = time.time()

    # --- 1. Load Data ---
    print("Loading bipartite network...")
    bipartite_edges = load_bipartite_network("weighted_disease_edges_one_mode.tsv")
    print(f"Loaded {len(bipartite_edges)} gene-disease edges")

    # --- 2. Compute Node Degrees ---
    print("Computing node degrees...")
    gene_degrees, disease_degrees, total_edges = compute_node_degrees(bipartite_edges)
    print("Degree computation completed.")

    # --- 3. Generate Random Network ---
    generate_random_network(gene_degrees, disease_degrees, total_edges)

    # --- 4. Extract Backbone ---
    extract_backbone(bipartite_edges, gene_degrees, disease_degrees, total_edges, num_simulations=500)
    
    # --- 5. Finalize ---
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

def load_bipartite_network(filename, chunk_size=10000):
    """
    Loads a bipartite network from a tab-separated file.
    
    This function reads the file in chunks to handle large datasets efficiently.
    It stops after reading the first chunk (10,000 edges) to serve as an example.
    
    Args:
        filename (str): The path to the input file.
        chunk_size (int): The number of rows to read in each chunk.
    
    Returns:
        pd.DataFrame: A DataFrame containing the bipartite edges.
    """
    edges = []
    for chunk in pd.read_csv(filename, sep='\t', header=None, names=['Gene', 'Disease'], chunksize=chunk_size, low_memory=False):
        edges.append(chunk)
         # The original code only processes the first chunk
    return pd.concat(edges, ignore_index=True)

def compute_node_degrees(bipartite_edges):
    """
    Computes the degree of each node (gene and disease) and the total number of edges.
    
    Args:
        bipartite_edges (pd.DataFrame): The DataFrame of the bipartite network.
    
    Returns:
        tuple: A tuple containing gene degrees (dict), disease degrees (dict), and total edges (int).
    """
    degree_start = time.time()
    gene_degrees = bipartite_edges.groupby('Gene').size().to_dict()
    disease_degrees = bipartite_edges.groupby('Disease').size().to_dict()
    total_edges = len(bipartite_edges)
    print(f"Degree computation completed in {time.time() - degree_start:.2f} seconds")
    return gene_degrees, disease_degrees, total_edges

def compute_p_ik(gene, disease, gene_degrees, disease_degrees, total_edges):
    """
    Computes the probability of an edge between a gene and a disease under the SDSM null model.
    
    Args:
        gene (str): The gene identifier.
        disease (str): The disease identifier.
        gene_degrees (dict): A dictionary of gene degrees.
        disease_degrees (dict): A dictionary of disease degrees.
        total_edges (int): The total number of edges in the network.
    
    Returns:
        float: The probability of an edge p_ik.
    """
    p_ik = (gene_degrees.get(gene, 0) * disease_degrees.get(disease, 0)) / total_edges if total_edges > 0 else 0
    return min(max(p_ik, 0), 1)

def generate_random_network(gene_degrees, disease_degrees, total_edges):
    """
    Generates and saves a single random bipartite network based on the SDSM probabilities.
    
    This function serves as a demonstration of the null model's output.
    
    Args:
        gene_degrees (dict): A dictionary of gene degrees.
        disease_degrees (dict): A dictionary of disease degrees.
        total_edges (int): The total number of edges in the network.
    """
    np.random.seed(42)
    random_network = []
    genes = list(gene_degrees.keys())
    diseases = list(disease_degrees.keys())
    
    print("Generating random bipartite network using Bernoulli trials...")
    random_network_start = time.time()
    
    for gene in genes:
        for disease in diseases:
            p_ik = compute_p_ik(gene, disease, gene_degrees, disease_degrees, total_edges)
            if np.random.rand() < p_ik:
                random_network.append((gene, disease))
    
    print(f"Random network generation completed in {time.time() - random_network_start:.2f} seconds")
    print(f"Random network has {len(random_network)} edges")
    
    pd.DataFrame(random_network, columns=['Gene', 'Disease']).to_csv(
        "random_network.tsv", sep='\t', index=False
    )
    print("Random network saved as 'random_network.tsv'")

def extract_backbone(bipartite_edges, gene_degrees, disease_degrees, total_edges, num_simulations):
    """
    Extracts the statistical backbone of the network using Z-scores and p-values.
    
    Args:
        bipartite_edges (pd.DataFrame): The DataFrame of the original bipartite network.
        gene_degrees (dict): A dictionary of gene degrees.
        disease_degrees (dict): A dictionary of disease degrees.
        total_edges (int): The total number of edges in the network.
        num_simulations (int): The number of simulations to run for the statistical test.
    """
    observed_edges = set(zip(bipartite_edges['Gene'], bipartite_edges['Disease']))
    observed_counts = Counter(observed_edges)
    
    chunk_size = 10000
    backbone_results = []
    print("\nStarting backbone extraction...")
    process_start = time.time()
    
    for i, (gene, disease) in enumerate(observed_counts.keys()):
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} pairs in {time.time() - process_start:.2f} seconds")
        
        p_ik = compute_p_ik(gene, disease, gene_degrees, disease_degrees, total_edges)
        
        # Simulate the distribution of edge counts under the null model
        simulated_counts = np.random.binomial(num_simulations, p_ik, size=num_simulations)
        std_dev = np.std(simulated_counts)
        mean_simulated = np.mean(simulated_counts)
        observed_count = observed_counts.get((gene, disease), 0)
        
        # Calculate Z-score and p-value
        z_score = 0 if std_dev == 0 else (observed_count - mean_simulated) / std_dev
        p_value = norm.sf(z_score)
        
        if p_value < 0.05:  # Filter for statistically significant edges
            backbone_results.append((gene, disease, z_score, p_value))
        
        # Write results to file in chunks to manage memory
        if len(backbone_results) >= chunk_size:
            pd.DataFrame(backbone_results, columns=['Gene', 'Disease', 'Z-Score', 'P-Value']).to_csv(
                "sdsm_backbone.tsv", sep='\t', mode='a', index=False, header=False
            )
            backbone_results = []
    
    # Write any remaining results
    if backbone_results:
        pd.DataFrame(backbone_results, columns=['Gene', 'Disease', 'Z-Score', 'P-Value']).to_csv(
            "sdsm_backbone.tsv", sep='\t', mode='a', index=False, header=False
        )
    
    print(f"Backbone extraction completed in {time.time() - process_start:.2f} seconds")
    print(f"Backbone results saved as 'sdsm_backbone.tsv'")

if __name__ == "__main__":
    main()
