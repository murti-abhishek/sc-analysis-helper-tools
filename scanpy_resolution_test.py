import spatialdata as sd
from spatialdata_io import xenium

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import squidpy as sq

import numpy as np
import pandas as pd

import networkx as nx
import os

def plot_umap_grid_by_resolution(adata, resolutions=np.arange(0.2, 1.8, 0.2), method='leiden', n_cols=3):
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc

    n_res = len(resolutions)
    n_rows = int(np.ceil(n_res / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))

    axs = axs.flatten()

    for i, res in enumerate(resolutions):
        key = f'{method}_{res:.1f}'
        if key not in adata.obs:
            if method == 'leiden':
                sc.tl.leiden(adata, resolution=res, key_added=key)
            elif method == 'louvain':
                sc.tl.louvain(adata, resolution=res, key_added=key)
            else:
                raise ValueError("Unsupported clustering method")

        sc.pl.umap(
            adata, color=key, ax=axs[i],
            title=f'{method} res={res:.1f}',
            show=False, legend_loc='on data', frameon=False
        )

    # Turn off extra axes if needed
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

# plot_umap_grid_by_resolution(adata, resolutions=np.arange(0.2, 1.9, 0.2), method='leiden')

def plot_clustertree_like_r(adata, resolutions=np.arange(0.2, 1.8, 0.2), method='leiden'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import scanpy as sc

    # Run clustering at each resolution if not already done
    for res in resolutions:
        key = f'{method}_{res:.1f}'
        if key not in adata.obs:
            if method == 'leiden':
                sc.tl.leiden(adata, resolution=res, key_added=key)
            elif method == 'louvain':
                sc.tl.louvain(adata, resolution=res, key_added=key)
            else:
                raise ValueError("Unsupported clustering method")

    # Build DataFrame of cluster assignments
    cluster_df = pd.DataFrame({
        f'{method}_{res:.1f}': adata.obs[f'{method}_{res:.1f}'].astype(str)
        for res in resolutions
    }, index=adata.obs_names)

    G = nx.DiGraph()
    node_sizes = {}
    edge_fractions = {}

    # Build graph edges across adjacent resolution levels
    for i in range(len(resolutions) - 1):
        r1 = f'{method}_{resolutions[i]:.1f}'
        r2 = f'{method}_{resolutions[i+1]:.1f}'

        ct = pd.crosstab(cluster_df[r1], cluster_df[r2])
        row_sums = ct.sum(axis=1)

        for c1 in ct.index:
            for c2 in ct.columns:
                weight = ct.loc[c1, c2]
                if weight > 0:
                    node1 = f'{r1}_{c1}'
                    node2 = f'{r2}_{c2}'
                    G.add_edge(node1, node2, weight=weight)

                    # Fraction of c1 that transitions to c2
                    frac = weight / row_sums[c1]
                    edge_fractions[(node1, node2)] = frac

    # Add node metadata
    for res in resolutions:
        r_key = f'{method}_{res:.1f}'
        counts = cluster_df[r_key].value_counts()
        for cluster, size in counts.items():
            node = f'{r_key}_{cluster}'
            G.add_node(node, resolution=res, size=size)
            node_sizes[node] = size

    # Layout: center nodes by resolution level, sorted by size
    pos = {}
    y_levels = sorted(list(set([G.nodes[n]['resolution'] for n in G.nodes])))
    y_map = {res: -i for i, res in enumerate(y_levels)}  # top-down layout
    clusters_by_level = {res: [] for res in y_levels}
    for n in G.nodes:
        res = G.nodes[n]['resolution']
        clusters_by_level[res].append(n)

    for res in y_levels:
        nodes = clusters_by_level[res]
        nodes = sorted(nodes, key=lambda n: -node_sizes[n])  # sort by size
        x_start = -0.5 * (len(nodes) - 1)
        for i, node in enumerate(nodes):
            pos[node] = (x_start + i, y_map[res])

    # Assign node colors using adata.uns
    node_colors = []
    for node in G.nodes:
        res = G.nodes[node]['resolution']
        res_key = f'{method}_{res:.1f}'
        cluster_id = node.split('_')[-1]
        try:
            cats = adata.obs[res_key].cat.categories
            idx = list(cats).index(cluster_id)
            color = adata.uns[f'{res_key}_colors'][idx]
        except Exception:
            color = 'lightgray'
        node_colors.append(color)

    # Scale node sizes
    scaled_node_sizes = [np.sqrt(node_sizes[n]) * 6 for n in G.nodes]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    nx.draw_networkx_nodes(
        G, pos, node_size=scaled_node_sizes,
        node_color=node_colors, edgecolors='black',
        linewidths=1.0, ax=ax
    )

    nx.draw_networkx_labels(
        G, pos, labels={n: n.split('_')[-1] for n in G.nodes},
        font_size=12, ax=ax
    )

    # Draw arrows with thickness scaled by fraction of transition
    for (u, v), frac in edge_fractions.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=frac * 5,  # scale for visibility
            alpha=0.4, edge_color='gray', arrows=True, ax=ax
        )

    # Resolution labels on far left
    for res in y_levels:
        y = y_map[res]
        ax.text(-12, y, f"res={res:.1f}", va='center', ha='right', fontsize=15, fontweight='bold')

    ax.set_title('Clustertree-style Plot', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# plot_clustertree_like_r(adata, resolutions=np.arange(0.2, 1.9, 0.2))
