"""
TrueMultiscaleBackbone: Multiscale backbone extraction from weighted networks.

Requirements:
-------------
- Python >= 3.8
- Libraries: numpy, pandas, networkx, scipy, tqdm

Install dependencies via pip:
-----------------------------
pip install numpy pandas networkx scipy tqdm

Usage:
------
1. Prepare a weighted edge list in TSV format with columns:
   source<TAB>target<TAB>weight
   (the first line can be a header; it will be skipped automatically)

2. Import and run the backbone extraction:
   
   from true_multiscale_backbone import TrueMultiscaleBackbone

   extractor = TrueMultiscaleBackbone()
   
   # Strict extraction (default)
   backbone = extractor.extract_backbone(
       "weighted_disease_edges_one_mode.tsv",
       "strict_ds_backbone.edgelist",
       mode='strict',
       nrows=200000   # optional: limit to first 200k rows
   )

3. Other modes:
   - 'balanced': keep more edges while maintaining structure
   - 'full': keep all edges passing significance

4. The result is a NetworkX DiGraph and saved weighted edge list.

References
----------
Slater, P. B. (2009).
A two-stage algorithm for extracting the multiscale backbone of complex weighted networks.
Proceedings of the National Academy of Sciences, 106, E66–E66.
https://doi.org/10.1073/pnas.0904725106
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict
from tqdm import tqdm

class TrueMultiscaleBackbone:
    """
    Flexible extraction of the multiscale backbone from weighted networks.

    Parameters:
    - max_iter: Maximum iterations for doubly stochastic normalization.
    - tol: Tolerance for convergence in normalization.
    - smoothing_power: Power to apply after normalization.
    - significance_level: Minimum edge weight considered significant.
    - sparsity_adjustment: Adjust for sparsity (not currently used directly).
    - early_stop_percentile: Stop adding edges when weight falls below this percentile.
    - min_edges: Minimum number of edges to keep before applying stopping criteria.
    """

    def __init__(self, max_iter=200, tol=1e-6, smoothing_power=2,
                 significance_level=0.05, sparsity_adjustment=True,
                 early_stop_percentile=5, min_edges=None):
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing_power = smoothing_power
        self.significance_level = significance_level
        self.sparsity_adjustment = sparsity_adjustment
        self.early_stop_percentile = early_stop_percentile
        self.min_edges = min_edges

    def _make_doubly_stochastic(self, matrix):
        """
        Performs exact doubly stochastic normalization of the input matrix.

        Returns:
        - normalized matrix raised to the smoothing power.
        """
        matrix = matrix.tocsr()
        for _ in range(self.max_iter):
            # Row normalization
            row_sums = matrix.sum(axis=1).A.ravel()
            row_sums[row_sums == 0] = 1
            matrix = matrix.multiply(1 / row_sums[:, None])

            # Column normalization
            col_sums = matrix.sum(axis=0).A.ravel()
            col_sums[col_sums == 0] = 1
            matrix = matrix.multiply(1 / col_sums)

            # Check convergence
            if np.max(np.abs(1 - matrix.sum(axis=1).A.ravel())) < self.tol and \
               np.max(np.abs(1 - matrix.sum(axis=0).A.ravel())) < self.tol:
                break

        return matrix.power(self.smoothing_power)

    def _tarjan_strongly_connected(self, adj_list):
        """
        Tarjan's algorithm to identify strongly connected components.
        """
        index = 0
        indices = {}
        lowlinks = {}
        on_stack = set()
        stack = []
        components = []

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in adj_list.get(node, []):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                components.append(component)

        for node in adj_list:
            if node not in indices:
                strongconnect(node)

        return components

    def _build_backbone(self, matrix, node_names):
        """
        Constructs the backbone graph using hierarchical extraction.

        Steps:
        1. Sort edges by weight descending.
        2. Apply significance threshold and early stopping percentile.
        3. Check strong connectivity periodically.
        """
        coo = matrix.tocoo()
        edges = sorted(zip(coo.row, coo.col, coo.data), key=lambda x: -x[2])

        # Pre-calculate thresholds
        weights = [w for _, _, w in edges]
        percentile_threshold = np.percentile(weights, self.early_stop_percentile)
        significance_threshold = self.significance_level if self.significance_level else 0

        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        adj_list = defaultdict(list)
        backbone_edges = []

        for i, j, w in tqdm(edges, desc="Building backbone"):
            if w < significance_threshold:
                continue

            G.add_edge(node_names[i], node_names[j], weight=w)
            adj_list[i].append(j)
            backbone_edges.append((i, j, w))

            # Minimum edges requirement
            if self.min_edges and len(backbone_edges) < self.min_edges:
                continue

            # Percentile stopping
            if w < percentile_threshold:
                print(f"Stopping: edge weight {w:.4f} below {percentile_threshold:.4f} percentile")
                break

            # Periodic strong connectivity check
            if len(backbone_edges) % 1000 == 0:
                components = self._tarjan_strongly_connected(adj_list)
                if len(components) == 1:
                    print("Stopping: achieved strong connectivity")
                    break

        return G

    def extract_backbone(self, edge_list_path, output_path, mode='strict', nrows=None):
        """
        Extracts the multiscale backbone from a weighted edge list.

        Parameters:
        - edge_list_path: Path to the input TSV file (columns: source, target, weight).
        - output_path: Path to save the weighted edge list of the backbone.
        - mode: 'strict', 'balanced', or 'full' extraction mode.
        - nrows: Optional. Number of rows to read from the input file.

        Returns:
        - NetworkX DiGraph of the extracted backbone.
        """
        # Set extraction mode parameters
        if mode == 'strict':
            self.early_stop_percentile = 20
            self.significance_level = 0.05
            self.min_edges = None
        elif mode == 'balanced':
            self.early_stop_percentile = 1
            self.significance_level = 0.001
            self.min_edges = 100
        elif mode == 'full':
            self.early_stop_percentile = 0
            self.significance_level = 0
            self.min_edges = None

        print("Loading network...")
        try:
            df = pd.read_csv(edge_list_path, sep="\t",
                             names=["source", "target", "weight"], skiprows=1, nrows=nrows)

            print(f"Rows loaded: {len(df)}")
            if df.empty:
                print(f"Warning: File '{edge_list_path}' is empty or could not be parsed. Returning empty graph.")
                return nx.DiGraph()

            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df = df.dropna()
            df = df.query("source != target")

            if df.empty:
                print("Warning: No valid edges after cleaning. Returning empty graph.")
                return nx.DiGraph()

        except FileNotFoundError:
            print(f"Error: File '{edge_list_path}' not found. Check path and filename.")
            return nx.DiGraph()
        except Exception as e:
            print(f"Unexpected error while reading file: {e}")
            return nx.DiGraph()

        nodes = sorted(set(df['source']) | set(df['target']))
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        if not nodes:
            print("Warning: No nodes found after processing. Returning empty graph.")
            return nx.DiGraph()

        sparse_matrix = coo_matrix(
            (df['weight'],
             ([node_to_idx[s] for s in df['source']],
              [node_to_idx[t] for t in df['target']])),
            shape=(len(nodes), len(nodes))
        ).tocsr()

        print(f"Network: {len(nodes)} nodes, {sparse_matrix.nnz} edges")

        if sparse_matrix.nnz == 0:
            print("Warning: Sparse matrix has no edges. Cannot normalize. Returning empty graph.")
            return nx.DiGraph()

        print("Normalizing matrix...")
        normalized = self._make_doubly_stochastic(sparse_matrix)

        print("Extracting backbone...")
        backbone = self._build_backbone(normalized, nodes)

        print("\nResults:")
        print(f"Nodes: {backbone.number_of_nodes()}")
        print(f"Edges: {backbone.number_of_edges()}")
        print(f"Strongly connected: {nx.is_strongly_connected(backbone)}")

        nx.write_weighted_edgelist(backbone, output_path)
        print(f"Saved backbone to {output_path}")

        return backbone


if __name__ == "__main__":
    extractor = TrueMultiscaleBackbone()

    strict_backbone = extractor.extract_backbone(
        "weighted_disease_edges_one_mode.tsv",
        "strict_ds_backbone_200k.edgelist",
        mode='strict',
        nrows=200000  # Only read first 200k rows
    )

    """
    # Balanced extraction example
    balanced_backbone = extractor.extract_backbone(
        "weighted_gene_edges_one_mode.tsv",
        "balanced_ds_backbone.edgelist",
        mode='balanced'
    )

    # Full extraction example
    full_backbone = extractor.extract_backbone(
        "gene_edges_one_mode.tsv",
        "full_ds_backbone.edgelist",
        mode='full'
    )
    """
