"""
Dependencies
This script requires the following Python libraries:
-pandas : For efficient data manipulation, especially when handling large datasets and reading files in chunks.
-psutil: For monitoring system resource usage, such as memory.
You can install these dependencies using `pip`:
-bash : pip install pandas psutil
"""

import pandas as pd
import time
import psutil
from collections import defaultdict
import time


input_file = "data_file.txt"
gene_output_file = "gene_to_all_diseases.tsv"
disease_output_file = "disease_to_all_genes.tsv"
gene_edges_file = "gene_edges_one_mode.tsv"
disease_edges_file = "disease_edges_one_mode.tsv"


CHUNK_SIZE = 50000

def timer(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} finished in {end_time - start_time:.4f} sec")
        return result
    return wrapper


start_time = time.time()
data_load_time = time.time()

disease_to_genes = defaultdict(set)
gene_to_diseases = defaultdict(set)

print("Reading file in chunks...")

try:
    total_rows = 0
    for chunk in pd.read_csv(input_file, sep='\t', header=0, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        for phenotype, gene in zip(chunk.iloc[:, 0], chunk.iloc[:, 1]):
            if pd.notna(phenotype) and pd.notna(gene):
                disease_to_genes[phenotype].add(gene)
                gene_to_diseases[gene].add(phenotype)

        mem_usage = psutil.virtual_memory().percent
        print(f"Processed {total_rows} rows... (Memory Usage: {mem_usage}%)")
except Exception as e:
    print(f"Error while reading file: {e}")

print(f"Data loaded in {time.time() - data_load_time:.2f} sec")


save_mappings_time = time.time()
print("Saving disease-gene and gene-disease mappings...")

with open(gene_output_file, "w", encoding="utf-8") as g_file:
    for gene, diseases in gene_to_diseases.items():
        g_file.write(f"{gene}\t{','.join(diseases)}\n")

with open(disease_output_file, "w", encoding="utf-8") as d_file:
    for disease, genes in disease_to_genes.items():
        d_file.write(f"{disease}\t{','.join(genes)}\n")

print(f"Mappings saved in {time.time() - save_mappings_time:.2f} sec")

gene_edges_time = time.time()
print("Generating gene-gene edges without the third column...")

genes = list(gene_to_diseases.keys())

with open(gene_edges_file, "w", encoding="utf-8") as edge_file:
    edge_file.write("Gene1\tGene2\n") # The header is now only two columns
    for i in range(len(genes)):
        g1 = genes[i]
        for j in range(i + 1, len(genes)):
            g2 = genes[j]
            # Check if there is at least one shared disease
            if gene_to_diseases[g1] & gene_to_diseases[g2]:
                edge_file.write(f"{g1}\t{g2}\n") # Writing only the two gene names

print(f"Gene edges computed in {time.time() - gene_edges_time:.2f} sec")


disease_edges_time = time.time()
print("Generating disease-disease edges without the third column...")

diseases = list(disease_to_genes.keys())

with open(disease_edges_file, "w", encoding="utf-8") as edge_file:
    edge_file.write("Disease1\tDisease2\n") # The header is now only two columns
    for i in range(len(diseases)):
        d1 = diseases[i]
        for j in range(i + 1, len(diseases)):
            d2 = diseases[j]
            # Check if there is at least one shared gene
            if disease_to_genes[d1] & disease_to_genes[d2]:
                edge_file.write(f"{d1}\t{d2}\n") # Writing only the two disease names

print(f"Disease edges computed in {time.time() - disease_edges_time:.2f} sec")


print(f"Total execution time: {time.time() - start_time:.2f} sec")
