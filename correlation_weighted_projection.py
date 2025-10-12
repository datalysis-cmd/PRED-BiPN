"""
This script defines the GeneCorrelationAnalyzer class, designed to analyze relationships
between genes or diseases from gene-disease interaction data.

It performs the following main functions:
1. Reads and cleans gene-disease interaction data from a tab-separated file.
2. Builds mappings for gene and disease identifiers to numerical indices.
3. Constructs a sparse bipartite (gene-disease) matrix representing the interactions.
4. Computes one-mode projections (gene-gene or disease-disease) to identify shared connections.
5. Calculates various correlation coefficients (Jaccard, Cosine, Pearson, Spearman, Kendall)
   between genes or diseases using parallel processing for efficiency.
6. Saves the computed correlations to a tab-separated output file.

Dependencies:
To run this script, you need the following Python libraries installed:
- pandas: For data manipulation and reading large files in chunks.
- numpy: For numerical operations, especially on arrays.
- scipy: For sparse matrix operations (lil_matrix, csr_matrix, save_npz) and statistical correlations (kendalltau, spearmanr, pearsonr).
- scikit-learn: Specifically for efficient cosine similarity calculation (cosine_similarity).
- tqdm: For displaying progress bars during long operations.

You can install these dependencies using pip:
pip install pandas numpy scipy scikit-learn tqdm
"""

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')


class GeneCorrelationAnalyzer:
    """
        Initializes the GeneCorrelationAnalyzer class.

        Args:
            filepath (str): The path to the input file with interaction data.
            chunksize (int): The size of chunks for reading the file.
    """
    def __init__(self, filepath, chunksize=10000):
        self.filepath = filepath
        self.chunksize = chunksize
        self.gene_dict = {}
        self.disease_dict = {}
        self.matrix = None          # bipartite gene-disease sparse matrix
        self.one_mode_matrix = None # one-mode projection sparse matrix
        self.correlation_matrix = None

    def _clean_data(self, chunk):
        """
        Internal method for cleaning a chunk of data.
        Removes rows with NaN values and rows containing 'nan' or 'None' (case-insensitive).

        Args:
            chunk (pd.DataFrame): A chunk of data from the input file.

        Returns:
            pd.DataFrame: The cleaned data chunk.
        """
        chunk = chunk.dropna()
        chunk = chunk[~chunk[0].str.contains('nan|None', case=False, na=False)]
        chunk = chunk[~chunk[1].str.contains('nan|None', case=False, na=False)]
        return chunk

    def _process_chunk(self, chunk):
        """
        Internal method for processing a data chunk and creating
        local gene and disease mappings. Used in parallel processing.

        Args:
            chunk (pd.DataFrame): A chunk of data from the input file.

        Returns:
            tuple: A tuple containing two dictionaries: local gene dictionary and local disease dictionary.
        """
        chunk = self._clean_data(chunk)
        local_gene_dict = {}
        local_disease_dict = {}

        for _, row in chunk.iterrows():
            gene, disease = str(row[0]).strip(), str(row[1]).strip()
            if not gene or not disease:
                continue
            if gene not in local_gene_dict:
                local_gene_dict[gene] = len(local_gene_dict)
            if disease not in local_disease_dict:
                local_disease_dict[disease] = len(local_disease_dict)

        return local_gene_dict, local_disease_dict

    def build_mappings(self):
        """
        Builds the final mappings for genes and diseases
        to numerical indices, by reading the file in chunks and using
        parallel processing.
        """

        print("Building gene and disease mappings...")
        chunks = pd.read_csv(self.filepath, sep='\t', header=None, chunksize=self.chunksize, dtype=str)

        with Pool(cpu_count()) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(self._process_chunk, chunks),
                               total=os.path.getsize(self.filepath) // self.chunksize):
                results.append(result)

        gene_counter, disease_counter = 0, 0
        for g_dict, d_dict in results:
            for gene in g_dict:
                if gene not in self.gene_dict:
                    self.gene_dict[gene] = gene_counter
                    gene_counter += 1
            for disease in d_dict:
                if disease not in self.disease_dict:
                    self.disease_dict[disease] = disease_counter
                    disease_counter += 1

    def _build_sparse_chunk(self, chunk):
        """
        Internal method for constructing parts of the sparse matrix from a data chunk.
        Used in parallel processing.

        Args:
            chunk (pd.DataFrame): A chunk of data from the input file.

        Returns:
            tuple: A tuple with lists for rows, columns, and data (1 for interaction).
        """
        chunk = self._clean_data(chunk)
        rows, cols, data = [], [], []

        for _, row in chunk.iterrows():
            gene = str(row[0]).strip()
            disease = str(row[1]).strip()

            if not gene or not disease:
                continue

            try:
                rows.append(self.gene_dict[gene])
                cols.append(self.disease_dict[disease])
                data.append(1)
            except KeyError:
                continue

        return rows, cols, data

    def construct_sparse_matrix(self):
        """
        Constructs the bipartite sparse matrix (genes x diseases) representing
        the interactions. Saves the matrix to a .npz file.
        """
        print("Constructing bipartite sparse matrix (genes x diseases)...")
        chunks = pd.read_csv(self.filepath, sep='\t', header=None, chunksize=self.chunksize, dtype=str)

        rows, cols, data = [], [], []
        with Pool(cpu_count()) as pool:
            results = pool.imap_unordered(self._build_sparse_chunk, chunks)
            for r, c, d in tqdm(results, total=os.path.getsize(self.filepath) // self.chunksize):
                if r and c and d:
                    rows.extend(r)
                    cols.extend(c)
                    data.extend(d)

        if rows and cols and data:
            self.matrix = csr_matrix((data, (rows, cols)),
                                     shape=(len(self.gene_dict), len(self.disease_dict)))
            save_npz("sparse_matrix.npz", self.matrix)
            print(f"Sparse matrix shape: {self.matrix.shape}")
        else:
            raise ValueError("No valid data found to construct matrix")

    def compute_one_mode_projection(self, mode='gene'):
        """
        Computes one-mode projection (gene-gene or disease-disease).
        This projection shows connections between entities of the same type
        through common neighbors in the bipartite matrix.

        Args:
            mode (str): 'gene' for gene-gene projection or 'disease' for disease-disease projection.

        Returns:
            csr_matrix: Square one-mode sparse matrix with weights (e.g., number of common connections).
        """
        if self.matrix is None:
            raise ValueError("Bipartite sparse matrix not constructed yet.")

        print(f"Computing one-mode projection for mode: {mode}")

        if mode == 'gene':
            one_mode = self.matrix.dot(self.matrix.T)
        elif mode == 'disease':
            one_mode = self.matrix.T.dot(self.matrix)
        else:
            raise ValueError("Mode must be 'gene' or 'disease'.")

        # Μηδενίζουμε διαγώνιο (αν θέλουμε μόνο ακμές μεταξύ διαφορετικών κόμβων)
        one_mode.setdiag(0)
        one_mode.eliminate_zeros()

        print(f"One-mode projection shape: {one_mode.shape}, nnz: {one_mode.nnz}")

        self.one_mode_matrix = one_mode
        return one_mode

    def _calculate_correlations(self, args):
        """
        Internal method for calculating correlations for a set of pairs.
        Used in parallel processing.

        Args:
            args (tuple): A tuple containing:
                          - gene_pairs (list): List of index pairs (i, j) for correlation calculation.
                          - correlation_type (str): The type of correlation ('jaccard', 'cosine', 'pearson', 'spearman', 'kendall').

        Returns:
            list: List of tuples (i, j, similarity) for the calculated pairs.
        """
      
        gene_pairs, correlation_type = args
        results = []

        # Επιλέγουμε τη matrix που θα χρησιμοποιήσουμε για correlations:
        # αν υπάρχει one_mode_matrix, δουλεύουμε με αυτήν,
        # αλλιώς με την bipartite (πιθανώς για jaccard)
        matrix = self.one_mode_matrix if self.one_mode_matrix is not None else self.matrix

        for i, j in gene_pairs:
            vec_i = matrix.getrow(i).toarray()[0]
            vec_j = matrix.getrow(j).toarray()[0]

            if correlation_type == 'jaccard':
                intersection = np.sum((vec_i > 0) & (vec_j > 0))
                union = np.sum((vec_i > 0) | (vec_j > 0))
                if union == 0:
                    continue
                similarity = intersection / union

            elif correlation_type == 'cosine':
                # Αν θέλουμε, μπορούμε να υπολογίσουμε cosine εδώ, αλλιώς ξεχωριστή μέθοδος
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i == 0 or norm_j == 0:
                    continue
                similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)

            else:
                common = (vec_i > 0) | (vec_j > 0)
                if np.sum(common) < 2:
                    continue

                try:
                    if correlation_type == 'kendall':
                        similarity, _ = kendalltau(vec_i[common], vec_j[common])
                    elif correlation_type == 'spearman':
                        similarity, _ = spearmanr(vec_i[common], vec_j[common])
                    elif correlation_type == 'pearson':
                        similarity, _ = pearsonr(vec_i[common], vec_j[common])
                    else:
                        raise ValueError(f"Unknown correlation type: {correlation_type}")

                    if np.isnan(similarity):
                        continue
                except Exception:
                    continue

            results.append((i, j, similarity))

        return results

    def compute_correlations_parallel(self, correlation_type='kendall', threshold=0.0):
        """
        Computes correlations (Jaccard, Pearson, Spearman, Kendall) between genes/diseases
        in parallel.

        Args:
            correlation_type (str): The type of correlation to compute.
            threshold (float): Minimum correlation value to store.

        Returns:
            lil_matrix: The sparse correlation matrix.
        """
        print(f"Computing {correlation_type} correlations...")
        n = self.one_mode_matrix.shape[0] if self.one_mode_matrix is not None else self.matrix.shape[0]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        batch_size = min(1000, len(pairs) // (cpu_count() * 10)) or 1
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]

        self.correlation_matrix = lil_matrix((n, n))

        with Pool(cpu_count()) as pool:
            args = [(batch, correlation_type) for batch in batches]
            for batch_result in tqdm(pool.imap_unordered(self._calculate_correlations, args), total=len(args)):
                for i, j, val in batch_result:
                    if val >= threshold:
                        self.correlation_matrix[i, j] = val
                        self.correlation_matrix[j, i] = val

        print(f"Non-zero correlations: {self.correlation_matrix.nnz}")
        return self.correlation_matrix

    def compute_cosine_similarity(self, threshold=0.0):
        """
        Computes cosine similarity between genes/diseases. This method uses
        scikit-learn's optimized cosine_similarity function.

        Args:
            threshold (float): Minimum similarity value to store.

        Returns:
            lil_matrix: The sparse cosine similarity matrix.
        """
        print("Computing cosine similarity matrix...")
        matrix = self.one_mode_matrix if self.one_mode_matrix is not None else self.matrix
        cos_sim = cosine_similarity(matrix, dense_output=False)
        cos_sim = cos_sim.tolil()
        self.correlation_matrix = lil_matrix(cos_sim.shape)

        for i in range(cos_sim.shape[0]):
            for j in cos_sim.rows[i]:
                if i < j and cos_sim[i, j] >= threshold:
                    self.correlation_matrix[i, j] = cos_sim[i, j]
                    self.correlation_matrix[j, i] = cos_sim[i, j]

        print(f"Non-zero cosine similarities: {self.correlation_matrix.nnz}")
        return self.correlation_matrix

    def save_correlations(self, output_file, mode='gene'):
        keys = list(self.gene_dict.keys()) if mode == 'gene' else list(self.disease_dict.keys())
        with open(output_file, 'w') as f:
            f.write(f"{mode.capitalize()}1\t{mode.capitalize()}2\tCorrelation\n")
            for i in range(self.correlation_matrix.shape[0]):
                for j in range(i + 1, self.correlation_matrix.shape[1]):
                    corr = self.correlation_matrix[i, j]
                    if corr != 0:
                        f.write(f"{keys[i]}\t{keys[j]}\t{corr:.4f}\n")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    try:
        correlation_type = input("Enter correlation type (jaccard/cosine/pearson/spearman/kendall): ").strip().lower()
        if correlation_type not in ['jaccard', 'cosine', 'pearson', 'spearman', 'kendall']:
            print("Invalid type! Defaulting to jaccard.")
            correlation_type = 'jaccard'

        mode = input("Enter projection mode (gene/disease/none): ").strip().lower()
        if mode not in ['gene', 'disease', 'none']:
            print("Invalid mode! Defaulting to gene.")
            mode = 'gene'

        analyzer = GeneCorrelationAnalyzer("data_file.txt")
        analyzer.build_mappings()
        analyzer.construct_sparse_matrix()

        if mode != 'none':
            analyzer.compute_one_mode_projection(mode=mode)

        if correlation_type == 'cosine':
            analyzer.compute_cosine_similarity(threshold=0.0)
        else:
            analyzer.compute_correlations_parallel(correlation_type=correlation_type, threshold=0.0)

        output_file = f"{mode}_{correlation_type}_correlation_results.tsv" if mode != 'none' else f"{correlation_type}_correlation_results.tsv"
        analyzer.save_correlations(output_file, mode=mode if mode != 'none' else 'gene')

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your input file and correlation type.")
