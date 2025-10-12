#!/usr/bin/env python3
"""
Pólya Filter - Numerically stable implementation for backbone extraction

Description
-----------
This script implements a numerically stable Pólya filter for extracting
the backbone of a weighted network. It reads a tab-separated edge list
(file with three columns: Node1, Node2, Weight), computes node properties
(degree and strength), computes per-edge Pólya p-values from both node
perspectives, applies multiple-testing correction (Bonferroni or FDR),
and writes the backbone edges that pass the corrected significance test.

Features
--------
- Numerically stable log-PMF computation for Pólya (Beta-Binomial like) model.
- Exact p-value calculation using log-sum-exp to avoid underflow/overflow.
- Handles files with headers and gracefully skips invalid lines.
- Supports Bonferroni and Benjamini-Hochberg FDR correction.
- Saves backbone edges with p-values and a computed ratio (r).

Input
-----
- TSV file with at least 3 columns per line: node1, node2, weight
  (weight should be numeric and > 0). Header lines are tolerated.

Output
------
- TSV file with columns:
    Node1  Node2  Weight  P_value_1  P_value_2  Ratio_r

Dependencies
------------
- numpy
- scipy
- statsmodels
- typing (standard)
- os, time (standard)

Install dependencies:
    pip install numpy scipy statsmodels

Usage
-----
- Update INPUT_FILE and OUTPUT_FILE variables (or adapt into CLI).
- Run the script:
    python polya_filter.py

Notes
-----
- The 'a' parameter controls the Pólya prior (a = 1.0 corresponds to
  the behavior similar to certain disparity-like filters).
- For undirected networks the number of tests is treated as 2 * M (L = 2M).
- The implementation aims to be robust to numeric instabilities.
"""

import numpy as np
import scipy.special as sp
from typing import Tuple, Dict, Union, List
import os
import time
from statsmodels.stats.multitest import multipletests

# -------------------------
# PolyaFilter implementation
# -------------------------
class PolyaFilter:
    """
    Pólya Filter: numerically stable implementation for backbone extraction.
    """

    def __init__(self, significance_level: float = 0.05, method: str = 'fdr'):
        """
        Initialize the Polya Filter.

        Args:
            significance_level (float): global significance level (alpha_u)
            method (str): multiple-testing correction method: 'bonferroni' or 'fdr'
        """
        self.alpha_u = significance_level
        self.method = method.lower()
        self.adj_dict: Dict[Tuple[str, str], float] = {}
        self.node_props: Dict[str, Tuple[int, float]] = {}  # node -> (degree, strength)

    def _log_polya_pmf(self, w: int, k: int, s: int, a: float) -> float:
        """
        Compute the log of the Pólya PMF P(w | k, s, a) in a numerically stable way.

        Args:
            w (int): observed integer weight (number of shared items)
            k (int): degree of the node
            s (int): strength (sum of incident weights) for the node
            a (float): Pólya parameter (a > 0)

        Returns:
            float: log-probability (natural log), or -inf if invalid
        """
        if w < 0 or w > s or s <= 0 or k <= 1 or a <= 0:
            return -np.inf

        alpha = 1.0 / a
        beta = (k - 1.0) / a

        try:
            # log(binomial(s, w)) = gammaln(s+1) - gammaln(w+1) - gammaln(s-w+1)
            log_binom = sp.gammaln(s + 1) - sp.gammaln(w + 1) - sp.gammaln(s - w + 1)

            # log Beta ratio: betaln(w + alpha, s - w + beta) - betaln(alpha, beta)
            log_beta_ratio = sp.betaln(w + alpha, s - w + beta) - sp.betaln(alpha, beta)

            return float(log_binom + log_beta_ratio)
        except Exception:
            return -np.inf

    def polya_pvalue(self, w: float, k: int, s: float, a: float) -> float:
        """
        Compute the right-tail p-value P(X >= w) under the Pólya model
        using log-sum-exp for numerical stability.

        Args:
            w (float): observed weight (will be rounded to nearest int)
            k (int): node degree
            s (float): node strength (rounded to int)
            a (float): Pólya parameter

        Returns:
            float: p-value in [0, 1]
        """
        w_int = int(round(w))
        s_int = int(round(s))

        # Edge cases
        if k == 1:
            return 0.0 if w_int > 0 else 1.0
        if w_int <= 0:
            return 1.0
        if w_int > s_int:
            return 0.0

        log_probs = []
        for x in range(w_int, s_int + 1):
            log_prob = self._log_polya_pmf(x, k, s_int, a)
            if log_prob > -np.inf:
                log_probs.append(log_prob)

        if not log_probs:
            return 0.0

        log_max = max(log_probs)
        s_exp = sum(np.exp(lp - log_max) for lp in log_probs)
        log_cumulative_prob = log_max + np.log(s_exp)

        # Convert back to probability
        p_value = np.exp(log_cumulative_prob)
        # Guard numerical boundaries
        return float(min(max(p_value, 0.0), 1.0))

    def compute_link_significance(self, w: float, node1: Tuple[int, float],
                                  node2: Tuple[int, float], a: float) -> Tuple[float, float]:
        """
        Compute the p-values for an edge (w) from both node perspectives.

        Args:
            w (float): observed weight for the edge
            node1, node2: tuples (degree, strength) for the two endpoints
            a (float): Pólya parameter

        Returns:
            (p1, p2): p-values from node1 and node2 point of view
        """
        k1, s1 = node1
        k2, s2 = node2
        p1 = self.polya_pvalue(w, k1, s1, a)
        p2 = self.polya_pvalue(w, k2, s2, a)
        return p1, p2

    def load_network_data(self, file_path: str, directed: bool = False) -> int:
        """
        Load network data from a TSV file and compute node properties (degree, strength).
        The function tolerates a header line and skips invalid lines.

        Args:
            file_path (str): path to input TSV file
            directed (bool): whether the network is directed (affects duplicate handling)

        Returns:
            int: approx. number of processed link entries (for reporting)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        adj_dict: Dict[Tuple[str, str], float] = {}
        node_properties: Dict[str, List[Union[int, float]]] = {}
        edge_list: List[Tuple[str, str, float]] = []
        all_nodes = set()

        print(f"[INFO] Reading data from '{file_path}'...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        node1 = parts[0]
                        node2 = parts[1]
                        weight = float(parts[2])
                        if weight <= 0:
                            continue
                        edge_list.append((node1, node2, weight))
                        all_nodes.add(node1)
                        all_nodes.add(node2)
                    except ValueError:
                        # Interpret non-numeric third column in first line as a header
                        if i == 0:
                            print(f"[WARN] First line appears to be a header; skipping it.")
                            continue
                        else:
                            print(f"[WARN] Skipping line {i+1} due to invalid weight: '{parts[2]}'")
                            continue
        except Exception as e:
            raise IOError(f"Error reading file: {e}")

        # Second pass: compute degree (k) and strength (s) per node, and build adjacency
        processed_edges = set()
        for u, v, w in edge_list:
            # Normalize edge ordering for undirected graphs to avoid duplicates
            u_norm, v_norm = (u, v) if u < v else (v, u)
            if not directed and (u_norm, v_norm) in processed_edges:
                continue

            adj_dict[(u, v)] = w
            processed_edges.add((u_norm, v_norm))

            # Initialize node properties if missing
            if u not in node_properties:
                node_properties[u] = [0, 0.0]  # [degree, strength]
            if not directed and v not in node_properties:
                node_properties[v] = [0, 0.0]

            # Update u
            node_properties[u][0] += 1
            node_properties[u][1] += w

            # Update v (for undirected, avoid counting self-loops twice)
            if not directed:
                if u != v:
                    node_properties[v][0] += 1
                    node_properties[v][1] += w

        # Convert properties to tuple (degree, strength) and store
        self.node_props = {
            node: (int(props[0]), float(props[1]))
            for node, props in node_properties.items() if props[0] > 0
        }
        self.adj_dict = adj_dict

        print(f"[INFO] Loaded {len(self.adj_dict)} unique edges and {len(self.node_props)} nodes.")
        return len(self.adj_dict) * (2 if not directed else 1)

    def extract_backbone(self, a: float = 1.0, directed: bool = False) -> List[Tuple]:
        """
        Extract backbone edges applying multiple-testing correction.

        Args:
            a (float): Pólya parameter
            directed (bool): whether to treat the network as directed

        Returns:
            List of tuples: (node1, node2, weight, p1, p2, r_ratio)
        """
        if not self.adj_dict:
            raise ValueError("Network data not loaded. Call load_network_data first.")

        num_links = len(self.adj_dict)
        num_tests = num_links if directed else 2 * num_links  # L

        p_values: List[float] = []
        edges_data: List[Tuple[Tuple[str, str], float, float, float]] = []

        print(f"[INFO] Computing p-values for {num_tests} tests...")

        # Compute p-values for all edges
        for (i, j), weight in self.adj_dict.items():
            if i not in self.node_props or j not in self.node_props:
                continue
            node_i = self.node_props[i]
            node_j = self.node_props[j]
            p1, p2 = self.compute_link_significance(weight, node_i, node_j, a)
            # Use the minimum of the two perspectives as the edge-level p-value for correction
            p_values.append(min(p1, p2))
            edges_data.append(((i, j), weight, p1, p2))

        # If no p-values (empty), return early
        if not p_values:
            print("[INFO] No p-values were computed (empty network or no valid edges).")
            return []

        # Apply multiple-testing correction
        if self.method == 'bonferroni':
            thresh = self.alpha_u / max(1, num_tests)
            rejected = np.array(p_values) < thresh
            print(f"[INFO] Bonferroni correction applied. Threshold = alpha_u / L = {thresh:.6e}")
        elif self.method == 'fdr':
            rejected, _, _, _ = multipletests(p_values, alpha=self.alpha_u, method='fdr_bh')
            print("[INFO] FDR (Benjamini-Hochberg) correction applied.")
        else:
            raise ValueError("Correction method must be 'bonferroni' or 'fdr'.")

        # Collect edges that passed the correction
        backbone_edges: List[Tuple[str, str, float, float, float, float]] = []
        for idx, (edge, weight, p1, p2) in enumerate(edges_data):
            if rejected[idx]:
                i, j = edge
                r_ratio = self.compute_r_ratio(weight, self.node_props[i], self.node_props[j])
                backbone_edges.append((i, j, weight, p1, p2, r_ratio))

        print(f"[INFO] Kept {len(backbone_edges)} edges in backbone out of {num_links} original edges.")
        return backbone_edges

    def compute_r_ratio(self, w: float, node1_props: Tuple[int, float],
                        node2_props: Tuple[int, float]) -> float:
        """
        Compute the ratio r = max(r_i, r_j) as in Eq. 4:
            r_i = (w / s_i) * k_i
        where k is degree and s is strength.

        Args:
            w (float): edge weight
            node1_props, node2_props: (degree, strength)

        Returns:
            float: ratio r
        """
        r_values: List[float] = []
        k1, s1 = node1_props
        if s1 > 0:
            r_values.append((w / s1) * k1)
        k2, s2 = node2_props
        if s2 > 0:
            r_values.append((w / s2) * k2)
        return float(max(r_values)) if r_values else 0.0

    def save_results(self, backbone_edges: List[Tuple], output_file: str):
        """
        Save backbone edges to TSV file with header.

        Args:
            backbone_edges: list of tuples (node1, node2, weight, p1, p2, r)
            output_file: path to output TSV
        """
        if not backbone_edges:
            print("[INFO] No backbone edges to save.")
            return

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Node1\tNode2\tWeight\tP_value_1\tP_value_2\tRatio_r\n")
                for node1, node2, w, p1, p2, r in backbone_edges:
                    f.write(f"{node1}\t{node2}\t{w}\t{p1:.10e}\t{p2:.10e}\t{r:.6f}\n")
            print(f"[INFO] Results saved to: {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")

# -------------------------
# Script entry point
# -------------------------
if __name__ == '__main__':
    # Default parameters (customize as needed)
    INPUT_FILE = "weighted_disease_edges_one_mode.tsv"
    OUTPUT_FILE = "polya_backbone_fdr_results.tsv"
    POLYA_PARAM_A = 1.0        # Pólya parameter a (a=1.0 is a common choice)
    ALPHA_LEVEL = 0.05        # global significance level
    CORRECTION_METHOD = 'fdr'  # 'fdr' or 'bonferroni'

    print("=" * 60)
    print("Pólya Filter - Numerically Stable Implementation")
    print(f"Input file: {INPUT_FILE}")
    print(f"Correction method: {CORRECTION_METHOD.upper()}")
    print(f"Alpha (global): {ALPHA_LEVEL}")
    print(f"Pólya parameter a: {POLYA_PARAM_A}")
    print("=" * 60)

    polya_filter = PolyaFilter(significance_level=ALPHA_LEVEL, method=CORRECTION_METHOD)

    start_time = time.time()
    try:
        # Load network data and compute node properties
        polya_filter.load_network_data(INPUT_FILE)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[ACTION] Please ensure the input file exists and retry.")
        raise SystemExit(1)

    # Extract backbone
    backbone_edges = polya_filter.extract_backbone(a=POLYA_PARAM_A, directed=False)

    # Save results
    polya_filter.save_results(backbone_edges, OUTPUT_FILE)

    total_time = time.time() - start_time
    print(f"[INFO] Total execution time: {total_time:.2f} seconds.")
