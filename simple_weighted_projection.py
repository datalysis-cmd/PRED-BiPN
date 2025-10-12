"""
Dependencies
This script requires the following Python libraries:
-pandas : For efficient data manipulation, especially when handling large datasets and reading files in chunks.
-psutil: For monitoring system resource usage, such as memory.
You can install these dependencies using `pip`:
-bash : pip install pandas psutil

How to run: python3 simple_weighted_projection.py


References
----------
Newman, M. E. J. (2001).
Scientific collaboration networks. II. Shortest paths, weighted networks, and centrality.
Physical Review E, 64, 016132.
https://doi.org/10.1103/PhysRevE.64.016132

"""

import pandas as pd
import time
import psutil
from collections import defaultdict
import time


input_file = "data.txt"
gene_output_file = "gene_to_all_diseases.tsv"
disease_output_file = "disease_to_all_genes.tsv"
gene_edges_file = "weighted_gene_edges_one_mode.tsv"
disease_edges_file = "weighted_disease_edges_one_mode.tsv"



CHUNK_SIZE = 50000  

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} finished in {end_time - start_time:.4f} sec")
        return result
    return wrapper


def count_unique_diseases_and_genes(filename):
    """Calculates the number of unique diseases and genes from the file"""
    df = pd.read_csv(filename, sep='\t', header=None, names=['Disease', 'Gene'], dtype=str, low_memory=False)

    unique_diseases = df['Disease'].nunique()
    unique_genes = df['Gene'].nunique()

    return unique_diseases, unique_genes


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
print("Generating gene-gene edges...")

genes = list(gene_to_diseases.keys())

with open(gene_edges_file, "w", encoding="utf-8") as edge_file:
    edge_file.write("Gene1\tGene2\tShared_Diseases\n")
    for i in range(len(genes)):
        g1 = genes[i]
        for j in range(i + 1, len(genes)):  # Αποφυγή διπλών συγκρίσεων
            g2 = genes[j]
            shared_diseases = gene_to_diseases[g1] & gene_to_diseases[g2]  # Ταχύτερη set σύγκριση
            if shared_diseases:
                edge_file.write(f"{g1}\t{g2}\t{len(shared_diseases)}\n")

print(f"Gene edges computed in {time.time() - gene_edges_time:.2f} sec")

disease_edges_time = time.time()
print("Generating disease-disease edges...")

diseases = list(disease_to_genes.keys())

with open(disease_edges_file, "w", encoding="utf-8") as edge_file:
    edge_file.write("Disease1\tDisease2\tShared_Genes\n")
    for i in range(len(diseases)):
        d1 = diseases[i]
        for j in range(i + 1, len(diseases)):  # Αποφυγή διπλών συγκρίσεων
            d2 = diseases[j]
            shared_genes = disease_to_genes[d1] & disease_to_genes[d2]  # Ταχύτερη set σύγκριση
            if shared_genes:
                edge_file.write(f"{d1}\t{d2}\t{len(shared_genes)}\n")

print(f"Disease edges computed in {time.time() - disease_edges_time:.2f} sec")


print(f"Total execution time: {time.time() - start_time:.2f} sec")
