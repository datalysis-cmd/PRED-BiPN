"""
This script calculates the odds ratios, confidence intervals, and p-values
for pairs of genes based on their shared and unique disease associations.

The main steps are:
1. Count the total number of diseases from the 'disease_to_all_genes.tsv' file.
2. Load the number of diseases associated with each individual gene from
   'gene_diseases_count.tsv'.
3. Iterate through pairs of genes from 'gene_edges_one_mode.tsv', which
   specifies the number of shared diseases for each pair.
4. For each gene pair, construct a 2x2 contingency table and calculate
   the odds ratio, confidence interval, and p-value using a Chi-squared test.
5. Write the results to 'gene_odds_ratios.tsv'.

To run this script, you'll need to install the following Python libraries.
You can do this using pip in your bash terminal:

pip install pandas
pip install numpy
pip install scipy

References
----------

Hattori, S., & Yanagawa, T. (200X).
Mantel-Haenszel estimators for irregular sparse K2 × J tables.
Biostatistics Center, Kurume University, Japan.
[Include DOI or URL if available]

Chung, Y., Lee, S. Y., Elston, R. C., & Park, T. (2006).
Odds ratio based multifactor-dimensionality reduction method for detecting gene–gene interactions.
Genetic Epidemiology, 30(8), 655–665.
https://doi.org/10.1002/gepi.20134
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2


gene_disease_counts = "disease_counts.csv"
gene_edges_file = "weighted_disease_edges_one_mode.tsv"
disease_to_all_genes_file = "disease_to_all_genes.tsv"
output_file = "disease_odds_ratios.tsv"

CHUNK_SIZE = 50000  


def count_total_diseases(filename):
    """
    Counts the total number of lines in a given file. This is used to determine
    the total number of diseases, as each line in the 'disease_to_all_genes_file'
    corresponds to a unique disease.
    """
    count = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count



def compute_odds_ratios_with_stats(filename, gene_disease_counts, total_diseases):
    """
    Computes the odds ratio, confidence interval, and p-value for each gene pair.
    This is the main processing function of the script. It reads gene pair data
    in chunks from the 'gene_edges_file'. For each pair (Gene1, Gene2) and their
    shared diseases 'a', it retrieves the individual disease counts for each gene
    (b and c) and calculates 'd' (diseases not associated with either gene).

    These values (a, b, c, d) form a 2x2 contingency table, which is then used
    to calculate:
    - The odds ratio (OR = (a*d)/(b*c)).
    - The 95% confidence interval (CI) for the odds ratio using a log-transformed
    standard error.
    - The p-value using a Chi-squared test with 1 degree of freedom.

    The results for each valid pair are written to the specified output file.
    Pairs with missing data or zero values that would cause division by zero
    are skipped to prevent errors.
    """
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("Source\tTarget\tOdds_Ratio\tCI_Lower\tCI_Upper\tP_value\n")  

        count_processed = 0
        count_skipped = 0
        count_written = 0

        for chunk in pd.read_csv(filename, sep="\t", header=0, names=["Gene1", "Gene2", "SharedDiseases"], chunksize=CHUNK_SIZE):
            for _, row in chunk.iterrows():
                g1, g2, a = row["Gene1"].strip(), row["Gene2"].strip(), int(row["SharedDiseases"])

                if g1 not in gene_disease_counts or g2 not in gene_disease_counts:
                    count_skipped += 1
                    continue  

                
                b = gene_disease_counts[g1] - a
                c = gene_disease_counts[g2] - a
                d = total_diseases - (a + b + c)

                if b * c == 0 or d <= 0:
                    count_skipped += 1
                    continue  

                
                or_value = (a * d) / (b * c)

                
                SE = np.sqrt(1/a + 1/b + 1/c + 1/d) if a > 0 and b > 0 and c > 0 and d > 0 else np.inf
                log_or = np.log(or_value)
                ci_lower = np.exp(log_or - 1.96 * SE)
                ci_upper = np.exp(log_or + 1.96 * SE)

                
                chi2_stat = ((a * d - b * c) ** 2) * (a + b + c + d) / ((a + b) * (a + c) * (b + d) * (c + d))
                p_value = 1 - chi2.cdf(chi2_stat, df=1)

                # for Gephi input
                f_out.write(f"{g1}\t{g2}\t{or_value:.4f}\t{ci_lower:.4f}\t{ci_upper:.4f}\t{p_value:.4e}\n")
                count_written += 1
                count_processed += 1

        print(f"Total pairs processed: {count_processed}")
        print(f"Pairs skipped (missing genes or zero values): {count_skipped}")
        print(f"Pairs written to file: {count_written}")


print("Counting total diseases...")
total_diseases = count_total_diseases(disease_to_all_genes_file)
print(f"Total Diseases: {total_diseases}")

print("Loading gene-disease counts...")


print("Computing odds ratios with confidence intervals and p-values...")
compute_odds_ratios_with_stats(gene_edges_file, gene_disease_counts, total_diseases)

print(f"Results saved to {output_file}")
