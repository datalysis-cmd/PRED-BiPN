#!/usr/bin/env python3
"""
===========================================================
H-Backbone Network Analysis for Gene Interaction Networks
===========================================================

Implementation of the method from:
Zhang et al. (2018) "Extracting h-Backbone as a Core Structure in Weighted Networks"
Scientific Reports 8:14356

Description:
------------
This script extracts the h-backbone from weighted gene interaction
networks, identifying core edges that are significant either via
h-strength (weight-based) or h-bridge (betweenness-based) metrics.
It calculates network-wide metrics, selects edges satisfying
h-backbone criteria, and outputs a filtered network.

Input:
------
- Tab-separated file (TSV) without headers or with 3 columns:
    gene1    gene2    weight
- gene1, gene2: Node identifiers
- weight: Numeric positive weight of the edge

Output:
-------
1. gene_backbone_results.tsv
    - Contains filtered backbone edges and analysis metadata

Features:
---------
- Computes h-strength: h-index of edge weights
- Computes h-bridge: h-index of edge betweenness centrality
- Edge sampling option for large networks
- Safe handling of missing/invalid weights
- Output includes metadata with original and backbone network stats
- Can limit number of processed rows to reduce memory usage

Required Dependencies:
----------------------
- pandas      : Data handling
- networkx    : Graph construction and analysis
- numpy       : Numerical operations
- warnings    : Suppress runtime warnings
- heapq       : Priority queue for Dijkstra's algorithm
- collections : defaultdict for counting
- typing      : Type hints

Install dependencies using pip:
--------------------------------
pip install pandas networkx numpy

Usage:
------
1. Save this script as 'h_backbone.py'
2. Place your input TSV file in the working directory
3. Run from the command line with required input file:
   python h_backbone.py input_file.tsv

Optional Arguments:
------------------
-m, --max_rows       : Maximum number of rows to process (default=200000)
-s, --sample_size    : Number of nodes to sample for betweenness calculation (default=None)
-o, --output         : Output TSV file path (default="gene_backbone_results.tsv")

Notes:
------
- Only positive edge weights are retained
- Large networks may benefit from setting sample_size < total nodes
- Output TSV contains metadata and edge list suitable for downstream analysis
"""
