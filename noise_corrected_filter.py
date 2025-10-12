"""
===========================================================
Noise-Corrected (NC) Backbone Extraction for Weighted Networks
===========================================================

Description:
------------
This script extracts a statistically significant backbone from 
weighted bipartite or gene-disease networks using the Noise-Corrected 
(NC) method based on the Delta method and Beta-Binomial modeling. 
It calculates per-edge statistics including expected weight, 
normalized weight (L_ij, L_tilde), posterior probability, variance, 
z-scores, and significance flags.

The script supports both standard and strict filtering, parallel 
processing for large datasets, and visualization/export of the 
backbone network.

Input:
------
- Tab-separated file with at least three columns:
    Gene1    Gene2    Shared_Diseases
- Gene1 and Gene2: node identifiers
- Shared_Diseases: weight of the edge between the nodes (numeric)

Output:
-------
1. filtered_edges.tsv
    - Contains filtered edges with detailed statistics
2. backbone_network.gexf
    - Gephi-compatible network including node/edge attributes
3. network_preview.png
    - Optional visualization preview for small networks

Features:
---------
- Numerically stable per-edge calculations (E_ij, L_ij, L_tilde, z_score)
- Delta-method variance estimation
- Beta-Binomial modeling of edge probabilities
- Standard and strict backbone filtering (z-score, node degree)
- Parallel processing using multiprocessing
- Automatic memory management for large datasets
- Safe visualization for small and medium networks

Required Dependencies:
----------------------
- pandas      : Data handling
- numpy       : Numerical operations
- scipy       : Statistical functions (beta, norm)
- networkx    : Network analysis and visualization
- matplotlib  : Optional network visualization
- tqdm        : Progress bars
- multiprocessing, gc, warnings, argparse, datetime : Standard libraries

Install dependencies using pip:
--------------------------------
pip install pandas numpy scipy networkx matplotlib tqdm

Usage:
------
1. Save this script as 'noise_corrected_backbone.py'
2. Place your input TSV file in the working directory
3. Run from the command line with optional parameters:
   python noise_corrected_backbone.py --input weighted_disease_edges_one_mode.tsv --output filtered_edges.tsv

Optional Arguments:
------------------
--strict           : Apply strict backbone filtering
--delta            : Delta threshold (z-score)
--zthresh          : Z-score threshold for strict filtering
--percentile       : Node degree percentile for strict filtering
--output           : Output TSV filename
--limit            : Maximum number of input rows processed
--alpha            : Alpha prior for Beta distribution
--one-sided        : One-sided z-test (positive L_tilde only)
--workers          : Number of parallel worker processes

Notes:
------
- Ensure the input file contains positive weights; zero-weight edges are automatically removed
- Large networks may require adjusting the --limit or number of workers
- Visualization generates a GEXF file and a PNG preview (if network is small)
"""


import argparse
import datetime
import gc
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from tqdm import tqdm

warnings.filterwarnings('ignore')


def calculate_edge_stats(row, N_total, alpha_prior, delta, one_sided=False, eps=1e-12):
    """
    Numerically-stable calculation of per-edge statistics.

    Returns a dict with extra fields: E_ij, L_ij, L_tilde, P_ij, V_Nij,
    V_L_tilde, significant, z_score
    """
    try:
        # Ensure floats (use float64 precision)
        Nij = float(row['Shared_Diseases'])
        Ni = float(row['N_i'])
        Nj = float(row['N_j'])
        N = float(N_total)

        # Early exit for degenerate cases
        if Ni <= 0 or Nj <= 0 or N <= 0:
            out = row.to_dict()
            out.update({
                'E_ij': 0.0, 'L_ij': 0.0, 'L_tilde': 0.0,
                'P_ij': 0.0, 'V_Nij': 0.0, 'V_L_tilde': 0.0,
                'significant': False, 'z_score': 0.0
            })
            return out

        # Expected value under null (avoid zero division)
        E_ij = (Ni * Nj) / (N + eps)

        # Posterior Beta parameters (conjugate)
        alpha_post = Nij + alpha_prior
        beta_post = (N - Nij) + alpha_prior
        S = alpha_post + beta_post + eps

        # Posterior mean (predictive p)
        P_ij = alpha_post / S

        # Beta-Binomial marginal (predictive) variance for Nij
        # Var(Nij) = N * p * (1-p) * (S + N) / (S + 1)
        V_Nij = N * P_ij * (1.0 - P_ij) * (S + N) / (S + 1.0 + eps)
        if not np.isfinite(V_Nij) or V_Nij < 0:
            V_Nij = 0.0

        # kappa and derivative d_kappa w.r.t Nij using chain rule approximation
        # kappa = 1/E_ij
        E_safe = E_ij + eps
        kappa = 1.0 / E_safe

        # approximate derivative of E_ij wrt Nij assuming Ni and Nj and N depend on Nij by +1
        # A more exact dependency may exist in alternative formulations, but this derivative
        # follows the user's earlier attempt and keeps numerical stability.
        # dE_dnij = (Nj + Ni) / N - (Ni*Nj) / N^2  (approx)
        dE_dnij = ((Nj + Ni) / (N + eps)) - ((Ni * Nj) / ((N + eps) ** 2))

        # d_kappa = - dE / E^2
        d_kappa = - dE_dnij / (E_safe * E_safe + eps)

        # L and transformation
        L_ij = Nij / E_safe
        L_tilde = (L_ij - 1.0) / (L_ij + 1.0 + eps)

        # delta-method variance for L_tilde
        numerator = 2.0 * (kappa + Nij * d_kappa)
        denominator = (kappa * Nij + 1.0) ** 2 + eps
        V_L_tilde = V_Nij * ( (numerator / denominator) ** 2 )

        if not np.isfinite(V_L_tilde) or V_L_tilde <= 0:
            # fallback: small positive variance to avoid NaNs
            V_L_tilde = eps

        # robust z-score
        z_score = float(L_tilde / np.sqrt(V_L_tilde + eps))

        # decide significance: default two-sided unless one_sided=True
        if one_sided:
            significant = bool(np.isfinite(z_score) and (z_score > delta))
        else:
            significant = bool(np.isfinite(z_score) and (abs(z_score) > delta))

        result = row.to_dict()
        result.update({
            'E_ij': float(E_ij),
            'L_ij': float(L_ij),
            'L_tilde': float(L_tilde),
            'P_ij': float(P_ij),
            'V_Nij': float(V_Nij),
            'V_L_tilde': float(V_L_tilde),
            'significant': significant,
            'z_score': float(z_score)
        })
        return result

    except Exception as e:
        # In case of unexpected error, return conservative values
        result = row.to_dict()
        result.update({'significant': False, 'z_score': 0.0,
                       'E_ij': 0.0, 'L_ij': 0.0, 'L_tilde': 0.0,
                       'P_ij': 0.0, 'V_Nij': 0.0, 'V_L_tilde': 0.0})
        return result


def process_chunk(chunk, N_total, alpha_prior, delta, one_sided=False):
    results = []
    # iterate rows without creating huge lists of Series objects
    for _, row in chunk.iterrows():
        results.append(calculate_edge_stats(row, N_total, alpha_prior, delta, one_sided=one_sided))
    return pd.DataFrame(results)


def noise_corrected_backbone(df, source_col, target_col, weight_col,
                             delta=1.64, alpha_prior=1.0, one_sided=False, max_workers=None):
    """
    Main backbone extraction using parallel processing. Returns DataFrame with statistics.
    """
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Use higher precision for numeric columns
    df[weight_col] = df[weight_col].astype('float64')
    df[source_col] = df[source_col].astype('category')
    df[target_col] = df[target_col].astype('category')

    print("Calculating node strengths...")
    node_strength_out = df.groupby(source_col)[weight_col].sum()
    node_strength_in = df.groupby(target_col)[weight_col].sum()

    df['N_i'] = df[source_col].map(node_strength_out).astype('float64')
    df['N_j'] = df[target_col].map(node_strength_in).astype('float64')
    N_total = float(df[weight_col].sum())

    # choose chunk size that balances memory and overhead
    chunk_size = min(50000, max(1000, len(df) // max(1, (cpu_count()))))
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

    workers = max_workers if max_workers is not None else min(cpu_count(), 8)
    process_func = partial(process_chunk, N_total=N_total, alpha_prior=alpha_prior, delta=delta, one_sided=one_sided)

    results = []
    print(f"Processing {len(df):,} edges in {len(chunks)} chunk(s) using {workers} worker(s)")
    with Pool(processes=workers) as pool:
        for result in tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Processing edges"):
            results.append(result)

    final_df = pd.concat(results, ignore_index=True)

    # debug info
    if 'V_L_tilde' in final_df.columns:
        bad_var_count = (final_df['V_L_tilde'] <= 0).sum()
        nan_z = final_df['z_score'].isna().sum()
        print(f"Edges with non-positive V_L_tilde: {bad_var_count} / {len(final_df)}")
        print(f"NaN z_scores: {nan_z}")

    # Ensure required columns exist
    required_columns = ['significant', 'E_ij', 'L_ij', 'L_tilde', 'z_score']
    for col in required_columns:
        if col not in final_df.columns:
            final_df[col] = False if col == 'significant' else 0.0

    # free memory
    del results, chunks
    gc.collect()

    return final_df


def strict_backbone_filtering(df, source_col, target_col, weight_col,
                              delta=3.09, z_threshold=4.0, degree_percentile=95,
                              alpha_prior=1.0, one_sided=False):
    nc_df = noise_corrected_backbone(df, source_col, target_col, weight_col,
                                     delta=delta, alpha_prior=alpha_prior, one_sided=one_sided)

    backbone = nc_df[(nc_df['significant']) & (abs(nc_df['z_score']) > z_threshold)].copy()

    if backbone.empty:
        print("No edges passed z-score and significance filters in strict filtering.")
        return backbone

    # Node degree in backbone (undirected combination)
    node_degree = pd.concat([
        backbone[source_col].value_counts(),
        backbone[target_col].value_counts()
    ]).groupby(level=0).sum()

    threshold = np.percentile(node_degree, degree_percentile)
    top_nodes = node_degree[node_degree > threshold].index

    return backbone[backbone[source_col].isin(top_nodes) & backbone[target_col].isin(top_nodes)].copy()


def save_filtered_edges(backbone, output_file, params=None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = [
            f"# Filtered gene-disease associations network",
            f"# Generated on: {timestamp}",
            f"# Total edges: {len(backbone)}",
            f"# Parameters: {params}" if params else "",
            f"# Columns: Gene1, Gene2, Shared_Diseases, N_i, N_j, E_ij, L_ij, L_tilde, P_ij, V_Nij, V_L_tilde, significant, z_score"
        ]

        with open(output_file, 'w') as f:
            f.write("\n".join(metadata) + "\n")
            backbone.to_csv(f, sep="\t", index=False, float_format='%.6f')

        print(f"\nSuccessfully saved filtered edges to: {output_file}")
        print(f"File contains {len(backbone)} significant edges")

    except Exception as e:
        print(f"\nError saving filtered edges: {str(e)}")
        backbone.to_csv(output_file, sep="\t", index=False)
        print(f"Saved basic version without metadata to: {output_file}")


def visualize_network_safely(backbone, source_col='Gene1', target_col='Gene2', weight_col='Shared_Diseases'):
    try:
        G = nx.Graph()  # undirected backbone is often easier to visualize

        for _, row in backbone.iterrows():
            u = row[source_col]
            v = row[target_col]
            w = float(row[weight_col])
            G.add_edge(u, v, weight=w)
            G.nodes[u]['label'] = u
            G.nodes[u]['size'] = np.log1p(row.get('N_i', 0.0))
            G.nodes[v]['label'] = v
            G.nodes[v]['size'] = np.log1p(row.get('N_j', 0.0))

        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        print(f"\nNetwork contains {num_nodes:,} nodes and {num_edges:,} edges")

        nx.write_gexf(G, "backbone_network.gexf", encoding='utf-8', prettyprint=True, version='1.2draft')
        print("GEXF file saved with complete node/edge attributes")

        if num_nodes < 5000 and num_edges > 0:
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)

            if num_nodes < 200:
                nx.draw_networkx(G, pos, with_labels=True, node_size=50, font_size=8, alpha=0.8, edge_color='gray')
            else:
                nx.draw_networkx_nodes(G, pos, node_size=5, alpha=0.6)
                nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.2)

            plt.title(f"Network Preview: {num_nodes} nodes, {num_edges} edges")
            plt.axis('off')
            plt.savefig("network_preview.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Created preview image (network_preview.png)")

    except Exception as e:
        print(f"Visualization error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Gene-disease network backbone extraction (fixed)')
    parser.add_argument('--strict', action='store_true', help='Use strict filtering')
    parser.add_argument('--delta', type=float, default=3.09, help='Delta threshold (z)')
    parser.add_argument('--zthresh', type=float, default=4.0, help='Z-score threshold for strict filtering')
    parser.add_argument('--percentile', type=float, default=95, help='Node degree percentile for strict filtering')
    parser.add_argument('--output', type=str, default='filtered_edges.tsv', help='Output TSV filename')
    parser.add_argument('--limit', type=int, default=200000, help='Limit number of input rows processed')
    parser.add_argument('--input', type=str, default='weighted_disease_edges_one_mode.tsv', help='Input TSV file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha prior for Beta')
    parser.add_argument('--one-sided', action='store_true', help='Use one-sided z test (positive L_tilde only)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default=min(cpu,8))')

    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(args.input, sep="\t", nrows=args.limit)

    required_columns = ['Gene1', 'Gene2', 'Shared_Diseases']
    if not all(col in df.columns for col in required_columns):
        print("Error: Missing required columns: expected Gene1, Gene2, Shared_Diseases")
        return

    df = df[df['Shared_Diseases'] > 0].copy()
    print(f"Working with {len(df):,} edges (after filtering zero weights)")

    if args.strict:
        print("\nApplying STRICT backbone filtering with:")
        print(f"Delta (z): {args.delta} (two-sided p ~ {norm.sf(abs(args.delta)) * 2:.4g})")
        print(f"Z-score threshold: {args.zthresh}")
        print(f"Node degree percentile: {args.percentile}%")

        backbone = strict_backbone_filtering(
            df,
            source_col="Gene1",
            target_col="Gene2",
            weight_col="Shared_Diseases",
            delta=args.delta,
            z_threshold=args.zthresh,
            degree_percentile=args.percentile,
            alpha_prior=args.alpha,
            one_sided=args.one_sided
        )

        params = {
            'filter_type': 'strict',
            'delta': args.delta,
            'z_threshold': args.zthresh,
            'degree_percentile': args.percentile,
            'row_limit': args.limit,
            'alpha': args.alpha,
            'one_sided': args.one_sided
        }
    else:
        print("\nApplying standard backbone filtering")
        nc_df = noise_corrected_backbone(df, "Gene1", "Gene2", "Shared_Diseases",
                                        delta=args.delta, alpha_prior=args.alpha,
                                        one_sided=args.one_sided, max_workers=args.workers)
        backbone = nc_df[nc_df['significant']].copy()
        params = {'filter_type': 'standard', 'delta': args.delta, 'row_limit': args.limit, 'alpha': args.alpha}

    save_filtered_edges(backbone, args.output, params)

    print("\nResults:")
    print(f"Original network: {len(df):,} edges")
    print(f"Backbone network: {len(backbone):,} edges ({(len(backbone)/len(df) if len(df)>0 else 0):.2%})")

    if len(backbone) > 0:
        print("\nStarting visualization...")
        visualize_network_safely(backbone)
    else:
        print("No edges passed the backbone filter")


if __name__ == '__main__':
    main()
