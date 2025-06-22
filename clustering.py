# %%
from matplotlib import pyplot as plt
import torch
import numpy as np
import json
import seaborn as sns
from jaxtyping import Float, Int
import circuitsvis as cv
from tqdm import tqdm
import pandas as pd
import networkx as nx
import leidenalg
import igraph as ig
from transformer_lens import HookedTransformer

from src.ablation import get_vocab_attn1_input
from src.utils import clean_mem

# %load_ext autoreload
# %autoreload 2

LAYER = 1
HEAD_IDX= 5
model = HookedTransformer.from_pretrained("gpt2-small")

# load token frequencies and subset
import pandas as pd
token_freq_df = pd.read_csv("out/token_frequencies.csv")
LIM = 3000
IDX = token_freq_df.id[:LIM]
VOCAB = token_freq_df.token[:LIM]

# get residuals for that subset
E : Float[torch.Tensor, "batch pos d_model"] = get_vocab_attn1_input(model).clone()[IDX]

# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()

attn = E @ W_K @ W_Q.T @ E.T

del E, W_Q, W_K, token_freq_df
clean_mem()

# %%
sns.heatmap(attn[:32,:32].numpy(force=True),
    xticklabels=VOCAB[:32],
    yticklabels = VOCAB[:32],
    vmin=0, 
    # vmax=180    
)
# %%
def normalize_similarity_matrix(sim_matrix: torch.Tensor) -> torch.Tensor:
    norms = sim_matrix.norm(dim=1, keepdim=True)
    return sim_matrix / (norms @ norms.T + 1e-6)
# normalized_attn = normalize_similarity_matrix(attn)
normalized_attn = attn - attn.min(dim=1, keepdim=True)[0] / (attn.max(dim=1, keepdim=True)[0] - attn.min(dim=1, keepdim=True)[0])


# %%
from sklearn.decomposition import PCA

def plot_token_pca(sim_matrix, tokens, num_points=100):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(sim_matrix[:num_points].cpu().numpy())
    
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)

    for i, token in enumerate(tokens[:num_points]):
        plt.text(coords[i, 0], coords[i, 1], token, fontsize=9)
    plt.title("PCA of token similarities")
    plt.show()

plot_token_pca(normalized_attn, VOCAB, num_points=128)
# %%
from sklearn.cluster import SpectralClustering

def cluster_tokens(sim_matrix, tokens, n_clusters=20):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = clustering.fit_predict(sim_matrix.cpu().numpy())

    cluster_dict = {}
    for token, label in zip(tokens, labels):
        cluster_dict.setdefault(label, []).append(token)
    return cluster_dict

clusters = cluster_tokens(normalized_attn.clamp(0), VOCAB, n_clusters=10)

# Display some clusters
for cluster_id, tokens in list(clusters.items())[:5]:
    print(f"Cluster {cluster_id}: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

# %%
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import leidenalg  # pip install leidenalg
import igraph as ig
import orjson

def detect_token_communities(attn: torch.Tensor, vocab: list[str], threshold_quantile: float = 0.9, plot=True):
    """
    Clusters tokens based on similarity matrix (dot-product) using Leiden community detection.

    Parameters:
    - attn: torch.Tensor of shape [N, N], similarity matrix from dot products
    - vocab: list of N token strings
    - threshold_quantile: percentile (0-1) below which similarities are zeroed

    Returns:
    - partition: dict of {token_idx: cluster_id}
    - G: NetworkX DiGraph
    """
    # Step 1: Symmetrize
    attn_sym = attn # (attn + attn.T) / 2

    # Step 2: Normalize to [0, 1]
    # attn_min, attn_max = attn_sym.min(), attn_sym.max()
    # attn_norm = (attn_sym - attn_min) / (attn_max - attn_min)

    # Step 3: Threshold by quantile
    threshold = torch.quantile(attn_sym, threshold_quantile, dim=1, keepdim=True)
    attn_thresh = attn_sym.clone()
    attn_thresh[attn_thresh < threshold] = 0.0
    

    # Step 4: Convert to igraph for Leiden
    adj_matrix = attn_thresh.cpu().numpy()
    sources, targets = np.where(adj_matrix > 0)
    print('Total edges:', sources.shape)
    weights = adj_matrix[sources, targets]
    edges = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(edges=edges, directed=True)
    g.es['weight'] = weights

    # Step 5: Leiden community detection
    leiden_partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight', 
    max_comm_size=512
    )
    partition = {i: comm for comm, cluster in enumerate(leiden_partition) for i in cluster}

    # Step 6: Also build NetworkX DiGraph for visualization
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Add softmax and normalized value as edge attributes
    attn_softmax = torch.softmax(attn_sym, dim=1).cpu().numpy()
    attn_norm = (attn_sym - attn_sym.min(dim=1, keepdim=True)[0]) / (attn_sym.max(dim=1, keepdim=True)[0] - attn_sym.min(dim=1, keepdim=True)[0] + 1e-8)
    attn_norm = attn_norm.cpu().numpy()

    for u, v in G.edges:
        G.edges[u, v]['softmax'] = float(attn_softmax[u, v])
        G.edges[u, v]['normalized'] = float(attn_norm[u, v])

    # Optional: Visualize clusters
    if plot:
        draw_token_clusters(G, partition, vocab)

    return partition, G

def draw_token_clusters(G, partition, vocab):
    """Plot graph with tokens colored by cluster ID."""
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=0.3)

    node_colors = [partition[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap='tab20')
    nx.draw_networkx_edges(G, pos, alpha=0.05)

    # Optionally label a few tokens
    for i, (x, y) in enumerate(pos.values()):
        if random.random() < 0.1:  # 2% chance of labeling each token
            plt.text(x, y, vocab[i], fontsize=8, alpha=0.7)

    plt.title("Token Clusters (Louvain Community Detection)")
    plt.axis("off")
    plt.show()



THRESH_Q = 1 - 4 / LIM
partition, G = detect_token_communities(normalized_attn, VOCAB.tolist(), threshold_quantile=THRESH_Q, plot=False)
print(f'Total nodes: {LIM}\nThreshold: {THRESH_Q:.5%}\nClusters: {max(partition.values())}')
# %%
comm_id = 13
tokens_in_comm = [VOCAB[i] for i, c in partition.items() if c == comm_id]
print(f"Community {comm_id}:")
print(tokens_in_comm)

# %%
# Where is token "red"
t = ' red'
t_i = np.where(VOCAB == t)[0][0]
print(f"Token '{t}' in cluster {partition[t_i]}")
# %%
# Save graph to graphology format
def nx_to_graphology_json(G, vocab, partition=None, pos=None, prob_matrix=None, softmax=None):
    """
    Convert a NetworkX DiGraph to Graphology JSON format.
    Each node will have an 'id' (string), a 'label' (token), 
    optional 'cluster' (from partition), and optional 'x','y' (from pos).
    Each edge will have 'source', 'target', and 'weight'.
    """
    nodes = [
        {
            "key": str(i),
            "attributes": {
                "label": vocab[i],
                **({"cluster": partition.get(i, -1)} if partition is not None else {}),
                **({"x": float(pos[i][0]), "y": float(pos[i][1])} if pos is not None and i in pos else {})
            }
        }
        for i in tqdm(G.nodes)
    ]
    edges = [
        {
            "key": f"{u}-{v}",
            "source": str(u),
            "target": str(v),
            "attributes": {
                "weight": data.get("weight", 0),
                'probability': float(prob_matrix[u, v]) if prob_matrix is not None else None,
                'softmax': float(softmax[u, v]) if softmax is not None else None,
            }
        }
        for u, v, data in tqdm(G.edges(data=True)) if data.get("weight", 0)
    ]
    graphology = {
        "type": "directed",
        "nodes": nodes,
        "edges": edges
    }
    print(f"Total edges: {len(edges)}")
    
    return graphology

# Compute layout for coordinates

# Create initial positions based on clusters
def create_cluster_initial_positions(partition, vocab_size):
    """Create initial positions for spring layout based on cluster membership."""
    import math
    
    # Get unique clusters
    clusters = list(set(partition.values()))
    n_clusters = len(clusters)
    
    # Create a circular arrangement of cluster centers
    cluster_centers = {}
    for i, cluster_id in enumerate(clusters):
        angle = 2 * math.pi * i / n_clusters
        radius = 1  # Distance from origin
        cluster_centers[cluster_id] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Initialize positions for all nodes
    pos = {}
    for node_id in range(vocab_size):
        cluster_id = partition.get(node_id, -1)
        if cluster_id in cluster_centers:
            # Add small random offset within cluster
            center_x, center_y = cluster_centers[cluster_id]
            offset_x = np.random.normal(0, 0.3)  # Small random offset
            offset_y = np.random.normal(0, 0.3)
            pos[node_id] = (center_x + offset_x, center_y + offset_y)
        else:
            # Random position for unclustered nodes
            pos[node_id] = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
    
    return pos

# Create cluster-based initial positions
initial_pos = create_cluster_initial_positions(partition, len(VOCAB))

# Use cluster-based initial positions for spring layout
pos = nx.spring_layout(G, pos=initial_pos, seed=42, weight='softmax', iterations=50, k=0.007)

# Save to file, including cluster and coordinates
graphology_json = nx_to_graphology_json(G, VOCAB, partition=partition, pos=pos, prob_matrix=normalized_attn.cpu().numpy(), softmax=torch.softmax(normalized_attn, dim=1).cpu().numpy())
print(f"Graphology JSON created with {len(graphology_json['nodes'])} nodes and {len(graphology_json['edges'])} edges.")
with open("docs/token_similarity_graphology.json", "w") as f:
    # Use orjson for faster JSON dumping if available
    try:
        f.write(orjson.dumps(graphology_json).decode())
    except ImportError:
        json.dump(graphology_json, f, indent=2)
print("Graph saved to docs/token_similarity_graphology.json")
