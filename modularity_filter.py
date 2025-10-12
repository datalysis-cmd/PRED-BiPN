"""
### Code's Functionallity

This Python code implements a method to extract a **backbone** from a large, undirected, weighted graph. A backbone is a simplified version of the original graph that retains its most essential structural properties, especially its community structure, while removing less important nodes and edges.

The central concept of the method is **modularity vitality**. A node's vitality measures how much its removal would impact the overall **modularity** of the graph. Modularity is a metric that quantifies the strength of a graph's community partition. Nodes with a low vitality score are considered less critical to the community structure and are good candidates for removal.

The process follows these steps:
1.  **Graph Loading**: The code reads an edge list from a tab-separated file to construct the graph.
2.  **Initial Community Detection**: It uses the **Louvain algorithm** to find an initial community partition.
3.  **Iterative Pruning**: It repeatedly calculates the vitality of all remaining nodes and removes the one with the lowest score.
4.  **Backbone Extraction**: This iterative removal continues until a target percentage of nodes remains. The largest connected component of the resulting subgraph is then output as the final backbone.

To handle large graphs efficiently, the code uses **sparse matrices** from the `scipy` library. This approach is memory-efficient and speeds up computations, which is crucial for big data.

---

### How to Run the Code and Install Dependencies

**1. Install Dependencies**

The script requires the following Python libraries: `networkx`, `python-louvain`, `numpy`, `scipy`, and `tqdm`. You can install them all at once using `pip`:

`pip install networkx python-louvain numpy scipy tqdm`

**2. Run from the Command Line**

To run the script, you execute it from the command line, providing the path to your input file and a name for the output file as arguments.

`python your_script_name.py input_file.tsv output_backbone.tsv`

**Optional Arguments:**
-   `--percentage <float>`: The percentage of nodes to keep in the backbone. The default value is `0.3` (30%).
-   `--max_lines <int>`: The maximum number of lines to read from the input file, which is useful for testing with very large datasets.

Example with optional arguments:
`python your_script_name.py my_graph.tsv my_backbone.tsv --percentage 0.1 --max_lines 100000`

References
----------
Rajeh, S., Savonnet, M., Leclercq, E., & Cherif, H. (2022).
Modularity-Based Backbone Extraction in Weighted Complex Networks.
In Network Science: 7th International Winter Conference, NetSci-X 2022, Porto, Portugal, February 8-11, 2022, Proceedings, 67–79. Springer.
"""
import networkx as nx
import time
from community import community_louvain
from collections import defaultdict
import numpy as np
from scipy import sparse
from tqdm import tqdm


class SparseModularityBackbone:
    """
    Class for extracting a modularity-based backbone from large undirected weighted graphs.
    Optimized for memory-efficient processing using sparse matrices.
    """

    def __init__(self, G):
        """
        Initialize the class with a given NetworkX graph.

        Parameters:
        G (networkx.Graph): An undirected, weighted graph.
        """
        self.original_G = G.copy()
        self.G = G.copy()
        
        self.nodes = list(self.G.nodes())
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)
        
        # Construct sparse adjacency matrix in CSR format
        rows, cols, data = [], [], []
        for u, v, d in self.G.edges(data=True):
            weight = d.get('weight', 1.0)
            rows.append(self.node_index[u])
            cols.append(self.node_index[v])
            data.append(weight)
            if u != v:  # For undirected graphs, ensure symmetry
                rows.append(self.node_index[v])
                cols.append(self.node_index[u])
                data.append(weight)
        
        self.adj_matrix = sparse.csr_matrix((data, (rows, cols)), 
                                            shape=(self.n_nodes, self.n_nodes))
        
        self.partition = None
        self.communities = None
        self.total_weight = None
        self.community_metrics = None

    def compute_initial_partition(self):
        """
        Compute the initial community partition using the Louvain method
        and precompute community-level metrics.
        """
        print("Computing initial partition using Louvain method...")
        start = time.time()
        
        self.partition = community_louvain.best_partition(self.G, weight='weight')
        self._update_community_structures()
        
        self.total_weight = self.adj_matrix.sum() / 2
        self._precompute_community_metrics()
        
        print(f"Initial partition completed in {time.time()-start:.2f} seconds")

    def _update_community_structures(self):
        """
        Build internal structures mapping each community to its member nodes.
        """
        self.communities = defaultdict(set)
        for node, comm in self.partition.items():
            self.communities[comm].add(node)

    def _precompute_community_metrics(self):
        """
        Precompute modularity-related metrics for each community.
        """
        self.community_metrics = {}
        
        for comm, node_ids in self.communities.items():
            node_indices = [self.node_index[node_id] for node_id in node_ids]
            submatrix = self.adj_matrix[node_indices, :][:, node_indices]
            sum_in = submatrix.sum() / 2
            sum_tot = self.adj_matrix[node_indices, :].sum()
            
            self.community_metrics[comm] = {
                'sum_in': sum_in,
                'sum_tot': sum_tot,
                'node_ids': node_ids
            }

    def compute_modularity(self):
        """
        Calculate the modularity value from precomputed community metrics.

        Returns:
        float: Modularity score.
        """
        if self.total_weight == 0:
            return 0.0

        Q = 0.0
        for metrics in self.community_metrics.values():
            sum_in = metrics['sum_in']
            sum_tot = metrics['sum_tot']
            Q += (sum_in / self.total_weight) - (sum_tot / (2 * self.total_weight)) ** 2
        return Q

    def compute_vitality(self, node_id):
        """
        Compute the vitality of a node (impact of its removal on modularity).

        Parameters:
        node_id (str): Node identifier.

        Returns:
        float: Vitality score.
        """
        if node_id not in self.partition:
            return 0.0

        node_idx = self.node_index[node_id]
        comm = self.partition[node_id]
        original_modularity = self.compute_modularity()

        node_row = self.adj_matrix[node_idx, :]
        node_sum_tot = node_row.sum()

        comm_node_indices = [self.node_index[n] for n in self.communities[comm]]
        node_sum_in = node_row[:, comm_node_indices].sum()

        temp_metrics = {c: dict(metrics) for c, metrics in self.community_metrics.items()}
        temp_metrics[comm]['sum_in'] -= node_sum_in
        temp_metrics[comm]['sum_tot'] -= node_sum_tot

        m = self.total_weight - node_sum_tot
        if m <= 0:
            return 0.0

        new_modularity = 0.0
        for metrics in temp_metrics.values():
            sum_in = metrics['sum_in']
            sum_tot = metrics['sum_tot']
            new_modularity += (sum_in / m) - (sum_tot / (2 * m)) ** 2

        return original_modularity - new_modularity

    def remove_node_and_update(self, node_id):
        """
        Remove a node from the graph and update internal structures.

        Parameters:
        node_id (str): Node identifier.
        """
        if node_id not in self.partition:
            return

        node_idx = self.node_index[node_id]
        comm = self.partition[node_id]

        node_row = self.adj_matrix[node_idx, :]
        node_sum_tot = node_row.sum()
        comm_node_indices = [self.node_index[n] for n in self.communities[comm]]
        node_sum_in = node_row[:, comm_node_indices].sum()

        self.community_metrics[comm]['sum_in'] -= node_sum_in
        self.community_metrics[comm]['sum_tot'] -= node_sum_tot
        self.total_weight -= node_sum_tot

        self.adj_matrix[node_idx, :] = 0
        self.adj_matrix[:, node_idx] = 0

        self.communities[comm].remove(node_id)
        del self.partition[node_id]

        if not self.communities[comm]:
            del self.communities[comm]
            del self.community_metrics[comm]

    def extract_backbone(self, percentage=0.3):
        """
        Extract the backbone network by iteratively removing low-vitality nodes.

        Parameters:
        percentage (float): Fraction of nodes to retain.

        Returns:
        networkx.Graph: The extracted backbone subgraph.
        """
        self.compute_initial_partition()
        target_size = max(1, int(self.n_nodes * percentage))

        print(f"Extracting backbone with target size {target_size} nodes...")
        pbar = tqdm(total=self.n_nodes - target_size)

        while self.adj_matrix.nnz > 0 and len(self.partition) > target_size:
            vitals = {node_id: self.compute_vitality(node_id) for node_id in list(self.partition.keys())}
            if not vitals:
                break

            min_node = min(vitals.keys(), key=lambda x: abs(vitals[x]))
            self.remove_node_and_update(min_node)
            pbar.update(1)

        pbar.close()

        remaining_nodes = list(self.partition.keys())
        backbone = self.G.subgraph(remaining_nodes).copy()

        if len(backbone.nodes()) > 0:
            largest_cc = max(nx.connected_components(backbone), key=len) if len(backbone.nodes()) > 1 else set(backbone.nodes())
            backbone = backbone.subgraph(largest_cc).copy()
        else:
            backbone = nx.Graph()

        return backbone


def load_graph_sparse(file_path, max_lines=None):
    """
    Load a graph from a TSV file using sparse structure.

    Parameters:
    file_path (str): Path to the edge list file.
    max_lines (int, optional): Maximum number of lines to read.

    Returns:
    networkx.Graph: The constructed graph.
    """
    G = nx.Graph()
    nodes = set()
    edges = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    u = parts[0]
                    v = parts[1]
                    w = float(parts[2]) if len(parts) > 2 else 1.0
                    nodes.add(u)
                    nodes.add(v)
                    edges.append((u, v, w))
                except ValueError:
                    continue

    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)

    return G


def save_backbone_sparse(backbone, output_file):
    """
    Save the resulting backbone to a TSV file.

    Parameters:
    backbone (networkx.Graph): The backbone graph.
    output_file (str): File path to write output.
    """
    with open(output_file, 'w') as f:
        for u, v, d in backbone.edges(data=True):
            f.write(f"{u}\t{v}\t{d.get('weight',1.0)}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sparse Modularity Vitality Backbone Extraction')
    parser.add_argument('input', help='Input edge list file')
    parser.add_argument('output', help='Output backbone file')
    parser.add_argument('--percentage', type=float, default=0.3, help='Percentage of nodes to keep')
    parser.add_argument('--max_lines', type=int, help='Maximum number of lines to read')

    args = parser.parse_args()

    print("Loading graph using sparse representation...")
    start_load = time.time()
    G = load_graph_sparse(args.input, args.max_lines)
    print(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges in {time.time()-start_load:.2f} seconds")

    print("Starting backbone extraction...")
    start_compute = time.time()
    backbone = SparseModularityBackbone(G).extract_backbone(args.percentage)
    print(f"Backbone extraction completed in {time.time()-start_compute:.2f} seconds")

    print("\nSummary:")
    print(f"Original graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Backbone: {len(backbone.nodes())} nodes, {len(backbone.edges())} edges")

    if len(backbone.nodes()) > 0:
        print("Saving results...")
        save_backbone_sparse(backbone, args.output)
        print(f"Backbone saved to {args.output}")
