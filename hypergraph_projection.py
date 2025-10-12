"""
=========================================
GENE COMMUNITY HYPERGRAPH VISUALIZATION
=========================================

Description:
    This script reads a hypergraph file representing gene communities
    and visualizes the largest communities in a circular layout. 
    Each community is plotted as a colored circle, nodes are placed 
    around the community center, and only some labels are shown to 
    reduce clutter. Statistics of communities are printed as well.

Input file format:
    Tab-separated file with two columns:
        Column 1: Source node (gene)
        Column 2: Comma-separated list of target nodes (genes in the same hyperedge)
    Example:
        GeneA    GeneB,GeneC,GeneD
        GeneE    GeneF,GeneG

Output:
    - A PNG image 'circular_communities.png' with the circular layout visualization
    - Printed statistics about the number of communities and their sizes

Required libraries:
    pip install networkx matplotlib numpy

    - networkx: for graph creation and community detection
    - matplotlib: for plotting nodes, edges, and community circles
    - numpy: for numeric operations, layout calculations

How to run:
    1. Save this script as 'visualize_gene_communities.py'
    2. Make sure you have the input file (e.g., 'hypergraph_gene_edges.txt')
    3. Run the script from the command line:
        python visualize_gene_communities.py
    4. The script will:
        - Read the hypergraph data and construct a network
        - Detect communities and select the largest ones
        - Create a circular layout for visualization
        - Draw nodes, edges, and community circles
        - Print statistics about community sizes
        - Save the figure as 'circular_communities.png'
        - Display the figure interactively

Notes:
    - Only the 15 largest communities are visualized by default
    - Node labels are randomly shown (~30%) to reduce clutter
    - The visualization uses a circular layout for community centers
    - You can adjust parameters like `max_communities` and label probability

Example input:
    Gene1    Gene2,Gene3,Gene4
    Gene5    Gene6,Gene7
    ...
    
References
----------
Young, J.-G., Petri, G., & Peixoto, T. P. (2023).
Hypergraph reconstruction from network data.
Communications Physics, 6, 226.
https://doi.org/10.1038/s42005-023-01164-7
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

# --- Step 1: Read data and find communities ---
def read_hypergraph_data(file_path):
    """Reads the file and returns unique hyperedges"""
    G_components = nx.Graph()
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            source = parts[0]
            targets = parts[1].split(',')
            
            for target in targets:
                G_components.add_edge(source, target)
    
    return list(nx.connected_components(G_components))

# --- Step 2: Create graph and layout based on communities ---
def create_community_layout(hyperedges, max_communities=15):
    """Creates a layout where each community is placed on a circle"""
    
    # Select the largest communities
    communities_sorted = sorted(hyperedges, key=len, reverse=True)[:max_communities]
    
    # Create graph
    G = nx.Graph()
    community_colors = {}
    community_centers = {}
    
    # Colors for each community
    colors = cm.tab20(np.linspace(0, 1, len(communities_sorted)))
    
    # Positions of community centers on a circle
    radius = 5
    angles = np.linspace(0, 2*np.pi, len(communities_sorted), endpoint=False)
    
    for i, (community, color) in enumerate(zip(communities_sorted, colors)):
        community_id = i
        community_colors[community_id] = color
        
        # Center of the community on the circle
        center_x = radius * np.cos(angles[i])
        center_y = radius * np.sin(angles[i])
        community_centers[community_id] = (center_x, center_y)
        
        # Add genes and edges
        for gene in community:
            G.add_node(gene, community=community_id)
        
        # Add edges between all genes in the community
        genes_list = list(community)
        for j in range(len(genes_list)):
            for k in range(j+1, len(genes_list)):
                G.add_edge(genes_list[j], genes_list[k], weight=2.0)
    
    # Compute positions for each node
    pos = {}
    for community_id, center in community_centers.items():
        community_genes = [node for node, attr in G.nodes(data=True) 
                          if attr['community'] == community_id]
        
        # Random position around the community center
        for gene in community_genes:
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(0, 1.5)  # Dispersion radius
            pos[gene] = (
                center[0] + distance * np.cos(angle),
                center[1] + distance * np.sin(angle)
            )
    
    return G, pos, community_colors, community_centers, communities_sorted

# --- Step 3: Visualization ---
def visualize_communities_circular(G, pos, community_colors, community_centers, communities):
    """Visualize communities in a circular layout"""
    
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    
    # Draw circles for each community
    patches = []
    for community_id, center in community_centers.items():
        circle = Circle(center, 2.0, alpha=0.2, 
                       color=community_colors[community_id])
        patches.append(circle)
        
        # Add label for the community
        plt.text(center[0], center[1] + 2.2, f'C{community_id}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=community_colors[community_id], alpha=0.7))
    
    # Add the circles to the plot
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    
    # Draw nodes colored by community
    node_colors = [community_colors[G.nodes[node]['community']] 
                  for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.8, ax=ax)
    
    # Draw only the most important edges (internal to the community)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', ax=ax)
    
    # Draw labels for some genes only (to avoid clutter)
    labels = {}
    for node in G.nodes():
        if np.random.random() < 0.3:  # 30% chance to show label
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    
    plt.title('Gene Community Visualization in Circular Layout\n(Each color = Different Community)', fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    
    # Legend with community info
    plt.figtext(0.02, 0.02, "Community Information:", fontweight='bold')
    y_pos = 0.01
    for i, community in enumerate(communities[:10]):  # First 10 communities
        if len(community) > 1:
            info = f"C{i}: {len(community)} genes"
            if len(community) <= 8:
                info += f" ({', '.join(sorted(community))})"
            plt.figtext(0.02, y_pos, info, fontsize=8)
            y_pos -= 0.015
    
    return ax

# --- Step 4: Detailed analysis ---
def print_community_stats(communities):
    """Print statistics for communities"""
    print(f"\n=== COMMUNITY STATISTICS ===")
    print(f"Total communities: {len(communities)}")
    print(f"Total genes: {sum(len(he) for he in communities)}")
    
    sizes = [len(he) for he in communities]
    print(f"\nSize distribution:")
    print(f"  - Max: {max(sizes)} genes")
    print(f"  - Min: {min(sizes)} genes")
    print(f"  - Mean: {np.mean(sizes):.2f} genes")
    
    print(f"\n10 LARGEST COMMUNITIES:")
    communities_sorted = sorted(communities, key=len, reverse=True)
    for i, community in enumerate(communities_sorted[:10]):
        print(f"{i+1}. Community {i}: {len(community)} genes")
        if len(community) <= 12:
            print(f"   Genes: {', '.join(sorted(community))}")
        print()

# --- Step 5: Main program ---
if __name__ == "__main__":
    file_path = "hypergraph_gene_edges.txt"
    
    print("Reading file...")
    hyperedges = read_hypergraph_data(file_path)
    
    print(f"Found {len(hyperedges)} communities")
    print(f"Total genes: {sum(len(he) for he in hyperedges)}")
    
    # Print community statistics
    print_community_stats(hyperedges)
    
    # Create visualization for the 15 largest communities
    print("Creating visualization...")
    G, pos, community_colors, community_centers, selected_communities = create_community_layout(
        hyperedges, max_communities=15
    )
    
    # Visualization
    visualize_communities_circular(G, pos, community_colors, community_centers, selected_communities)
    
    # Save figure
    plt.savefig('circular_communities.png', dpi=300, bbox_inches='tight')
    print("Visualization completed! Saved as 'circular_communities.png'")
    
    plt.show()
