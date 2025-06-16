# %%
from matplotlib import pyplot as plt
import torch
import numpy as np
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint


from sae_lens import HookedSAETransformer, SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display

import torch.nn.functional as F
import random

import seaborn as sns
from rich.table import Table
from rich.console import Console

import seaborn as sns

def barplot(values, **set_args):
    plt.figure(figsize=(16,8))
    ax = sns.barplot(values.numpy(force=True))
    ax.set_xticks([])  # Remove x-axis ticks
    if set_args:
        ax.set(**set_args)
    for i, v in enumerate(values.numpy(force=True)):
        ax.text(i, v * 1.1,  str(i), ha='center', va='bottom', fontsize=10)
    return ax

def clean_mem():
    import gc
    
    # for var in ['loss', 'E']:
    #     if var in locals():
    #         del locals()[var]
    #     elif var in globals():
    #         del globals()[var]
        
    gc.collect()
    torch.cuda.empty_cache()

from transformers import PreTrainedTokenizerBase
from jaxtyping import Float, Int
import circuitsvis as cv

# %load_ext autoreload
# %autoreload 2

from src.maps import *
from src.utils import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

LAYER = 1
HEAD_IDX= 5
LIM = -1
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"
model = HookedSAETransformer.from_pretrained("gpt2-small")

import pandas as pd
token_freq_df = pd.read_csv("out/token_frequencies.csv")

VOCAB_LIM = 1024
IDX = token_freq_df.id[:VOCAB_LIM]
VOCAB = token_freq_df.token[:VOCAB_LIM]


#%%
VOCAB_SIZE = model.tokenizer.vocab_size


def ablate_component(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
):
    """Mean-ablate a component"""
    return t.mean(dim=(0,1))

def ablate_reduce_component(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
):
    print(hook.name)
    return t[:,IDX,:]

def stop_computation(t, hook):
    raise StopIteration(f"Stopping model mid-execution at {hook.name}")

@torch.inference_mode()
def vocab_attn(
    model: HookedTransformer,
) -> Float[Tensor, "layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    model.reset_hooks()
    vocab = torch.arange(0, VOCAB_SIZE, device=device)
    ablate_components = [
        'hook_pos_embed',
        'blocks.0.hook_attn_out',
        # 'blocks.0.hook_mlp_out',
    ]
    # This dict will store the cached activations
    activation_cache = {}
    
    out_hook = 'blocks.1.ln1.hook_normalized' # 'hook_embed'# 

    # Hook function to cache activations
    def cache_hook(activation, hook):
        activation_cache[hook.name] = activation
        
    def replace_hook(activation, hook):
        return activation_cache['blocks.0.hook_mlp_out']
        
    model.cfg.use_attn_in = False
        
    hooks = [
        # ("blocks.1.attn.hook_q", cache_hook),
        # ("blocks.1.attn.hook_k", cache_hook),
        ('blocks.0.ln1.hook_normalized', ablate_reduce_component),
        # ('blocks.0.hook_mlp_out', cache_hook),
        # ('blocks.0.hook_resid_post', replace_hook ),
        (out_hook, cache_hook),  
        (out_hook, stop_computation),  
    ]
    for component in ablate_components:
        hooks.append((component, ablate_component))
        
    print(hooks)
    try:
        model.run_with_hooks(vocab, fwd_hooks=hooks)
    except StopIteration as e:
        print(e)

    model.reset_hooks()
    return activation_cache[out_hook][0,:]


attn_in = vocab_attn(model).clone()[:LIM]

# %%
# Ground Truth attention pattern
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()
# W_QK_ideal = W_Q @ W_K.T
# W_QK_ideal.fill_diagonal_(2)
# attn_ideal = attn_in @ W_QK_ideal @ attn_in.T

# display_most_attended_tokens(tokens, attn_ideal, k=10)
# %%
# sns.heatmap(attn_ideal[:LIM,:LIM].cpu().numpy(force=True), cmap='viridis', vmin=0)

# %%

group_tokens = [
    ' Monday', # week days
    ' Tuesday',
    ' Wednesday',
    ' Thursday',
    ' Friday', 
    ' Saturday',
    ' red',
    ' blue', 
    ' Blue',
    ' green',
    ' silver', 
    ' White', 
    ' 1918', # years
    ' 1920',
    ' 1930',	
    ' 1943', 
    ' 1998',
    ' 2000',
    ' You', # pronouns
    ' He',
    ' his',
    ' she',
    ' her',
    ' their',
    ' Italy', # countries
    ' Iceland', 
    ' Austria',
    ' Mexico',
    ' Spain', 
    ' France', 
    # ' June', # months
    # ' July',
    # ' August',
    # ' September',
    # ' October',
    # ' November',
    # ' saw', # verbs
    # ' changes',
    # ' have',
    # ' were',
    # ' explained',
    # ' said',
    # ' 43', # numbers
    # ' 69',
    # ' 78',
    # ' 83',
    # ' 32',
    # ' 12',
]

group_toks=model.tokenizer(group_tokens).input_ids
group_toks = list(chain(*group_toks))
# %%
W_QK = W_Q @ W_K.T
W_sym = (W_QK + W_QK.T) / 2
W_skew = (W_QK - W_QK.T) / 2

E = attn_in[group_toks].clone().detach()

attn_base = E @ W_QK @ E.T
attn_sym = E @ W_sym @ E.T
attn_skew = E @ W_skew @ E.T

# optionally normalize attention matrices
# attn_base /= attn_base.diagonal()
# attn_sym /= attn_sym.diagonal()
# attn_skew /= attn_skew.diagonal()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.heatmap(attn_base.cpu().numpy(force=True), cmap='icefire', center=0,ax=axes[0], xticklabels=False, yticklabels=False, vmin=attn_base.min().item(), vmax=attn_base.max().item())
axes[0].set_title("Base Attention")

sns.heatmap(attn_sym.cpu().numpy(force=True), cmap='icefire', center=0,ax=axes[1], xticklabels=False, yticklabels=False, vmin=attn_base.min().item(), vmax=attn_base.max().item())
axes[1].set_title("Symmetric Attention")

sns.heatmap(attn_skew.cpu().numpy(force=True), cmap='icefire', center=0,ax=axes[2], vmax=attn_base.max().item(), vmin=attn_base.min().item(), xticklabels=False, yticklabels=False)
axes[2].set_title("Skew-Symmetric Attention")

plt.tight_layout()
plt.show()
# %%
def largest_sv(W: Float[Tensor, "d d"]) -> Float[Tensor, "d"]:
    """Returns the largest singular value of W"""
    return torch.linalg.svdvals(W).max()

print("Largest singular W_sym:", largest_sv(W_sym))
print("Largest singular W_skew:", largest_sv(W_skew))
# %%
true_eigenvalues = torch.linalg.eigvalsh(W_sym)
true_eigenvalues = true_eigenvalues[torch.argsort(torch.abs(true_eigenvalues), descending=True)]
true_eigenvalues = true_eigenvalues[:64]  # Zero out noise components since W has rank 64
print("Eigenvalues of W_sym:", true_eigenvalues)
print("Number of negative eigenvalues:", torch.sum(true_eigenvalues < 0))
print("Number of positive eigenvalues:", torch.sum(true_eigenvalues > 0))
print("Min eigenvalue:", torch.min(true_eigenvalues))
print("Max eigenvalue:", torch.max(true_eigenvalues))

# %%
# Compute attention and cosine similarity
import torch
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(1)  # For reproducibility

E = attn_in.detach().clone()#[np.random.choice(attn_in.shape[0], 5000, replace=False)]  # Use a subset for visualization

# Assuming E, W_QK are defined and on the correct device
attn_base = (E @ W_QK @ E.T).cpu() / np.sqrt(64)
# get logprobs
attn_base = attn_base.log_softmax(dim=-1)  # Convert to log probabilities
# Normalize attention values to have unit diagonal
attn_base = attn_base - attn_base.diagonal() # Subtract diagonal to center around zero
# attn_base /= attn_base.diagonal()#.reshape(-1, 1)  # Normalize each row by its diagonal value
# attn_base = attn_base.clamp(-10, 10)  # Clip values for better visualization

E = torch.nn.functional.normalize(E)  # Normalize E to unit length
cos_sim = (E @ E.T ).cpu()  # Cosine similarity matrix
cos_sim.fill_diagonal_(1)  # Set diagonal to 1 to avoid machine precision issues

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate statistics based on cosine similarity windows using torch
window_size = 0.08
cos_min, cos_max = cos_sim.min(), cos_sim.max()
bins = torch.linspace(cos_min, 1, 50)  # Create overlapping bins with step 0.01

# Use tensors directly
print(f'Processing {cos_sim.numel()} points')

# Initialize lists to store results
cos_sim_centers = []
mean_attn_values = []
std_attn_values = []
window_counts = []

# For each bin center, calculate statistics for all points in [center, center+window_size]
from tqdm import tqdm
for center in tqdm(bins):
    mask = (cos_sim >= center) & (cos_sim <= center + window_size)  # Create a mask for the current window

    cos_sim_centers.append(center.item())  # Use middle of window as x-coordinate
    mean_attn_values.append(attn_base[mask].mean().item())
    std_attn_values.append(attn_base[mask].std().item())
    window_counts.append(mask.sum().item())

# Create smoothed dataframe
df_smoothed = pd.DataFrame({
    'cos_sim': cos_sim_centers,
    'mean_attn': mean_attn_values,
    'std_attn': std_attn_values,
    'count': window_counts
})

# Filter out windows with too few points for better visualization
# df_smoothed = df_smoothed[df_smoothed['count'] > 50]

print(f'Smoothed {len(df_smoothed)} points with window size {window_size}')

# Calculate confidence interval (95% CI ≈ 1.96 * std / sqrt(n))
df_smoothed['ci_upper'] = df_smoothed['mean_attn'] + 1.96 * df_smoothed['std_attn'] / (df_smoothed['count'] ** 0.25)
df_smoothed['ci_lower'] = df_smoothed['mean_attn'] - 1.96 * df_smoothed['std_attn'] / (df_smoothed['count'] ** 0.25)

# %%
# Plot
plt.figure(figsize=(10, 6))

print(f'Plotting {len(df_smoothed)} smoothed points')
# Confidence interval shading
plt.fill_between(
    df_smoothed['cos_sim'],
    df_smoothed['ci_lower'],
    df_smoothed['ci_upper'],
    color=sns.color_palette("icefire", 10)[-1],
    alpha=0.4,
    label='95% Confidence Interval'
)

# Smoothed line
sns.lineplot(
    x=df_smoothed['cos_sim'],
    y=df_smoothed['mean_attn'],
    label='NormalizeAttention',
    palette='icefire',
)

# Scatter plot of sampled raw points (e.g., 1% of the data)
# Create bins across the cosine similarity range for uniform sampling
# n_bins = 20
# bins = pd.cut(df['cos_sim'], bins=n_bins)
# sampled_df = df.groupby(bins).apply(lambda x: x.sample(min(5, len(x)), random_state=1)).reset_index(drop=True)
# # Filter out extreme outliers
# sampled_df = sampled_df[sampled_df.attn_val.abs() < df_smoothed['ci_upper'].max() * 1.2]
# sns.scatterplot(
#     x=sampled_df['cos_sim'],
#     y=sampled_df['attn_val'],
#     color='gray',
#     alpha=0.3,
#     # s=10,
# )

# Add horizontal baseline at y=1
plt.axhline(y=1, color=sns.color_palette("icefire", 1)[0], linestyle='--', linewidth=1, label='Baseline')

# Labels and title
plt.xlabel('Cosine Similarity')
plt.ylabel('Normalized Attention Value')
plt.title('Attention score on similar tokens')
plt.legend().remove()
# plt.ylim(-10, 10)
plt.grid(True, axis='y',)

# Add arrow in bottom right corner
plt.annotate('more similar', xy=(0.15, -8), xytext=(0, -8),
             arrowprops=dict(facecolor='black', shrink=0.05, headwidth=6, width=1),
             ha='center', va='center')

plt.tight_layout()
plt.show()

# %%
# Get eigenvalues and eigenvectors of symmetric matrix
S, Q = torch.linalg.eigh(W_sym)

# Sort by absolute value in descending order
abs_S = torch.abs(S)
sorted_indices = torch.argsort(abs_S, descending=True)
S = S[sorted_indices]
Q = Q[:, sorted_indices]

S = S[:64]  # Zero out noise components since W has rank 64
Q = Q[:, :64]  # Keep only the first 64 eigenvectors
# Create a figure with subplots for different scaling values
# Create a figure with subplots for different scaling values
scaling_values = [0.5, 0, -1]
titles = [f"Scale negative eigenvalues by {s}" if s != 0 else "Zero out negative eigenvalues" for s in scaling_values]

fig, axes = plt.subplots(1, len(scaling_values), figsize=(20, 6))
if len(scaling_values) == 1:  # Handle case of single subplot
    axes = [axes]

for i, (scale, title) in enumerate(zip(scaling_values, titles)):
    # Clone the eigenvalues to avoid modifying the original
    S_modified = S.clone()
    
    # Apply the scaling to negative eigenvalues
    S_modified[S_modified < 0] *= scale
    
    # Compute attention with modified eigenvalues
    attn_svd = E @ ((Q @ torch.diag(S_modified) @ Q.T) + W_skew) @ E.T
    
    # Plot the heatmap
    sns.heatmap(
        attn_svd[:LIM,:LIM].cpu().numpy(force=True), 
        cmap='icefire', 
        center=0,
        ax=axes[i],
        xticklabels=False,
        yticklabels=False,
    )
    axes[i].set_title(title)

plt.tight_layout()
plt.show()

# %%
S, Q = torch.linalg.eig(W_QK)
S = S[:64]  # Zero out noise components since W has rank 64
# Q = Q[:, :64]  # Keep only the first 64 eigenvectors

S[S.real < 0]*= 0 # Zero out negative eigenvalues


attn_svd = E @ ((Q[:,:64] @ torch.diag(S) @ torch.linalg.inv(Q)[:64]).real) @ E.T
sns.heatmap(attn_svd[:LIM,:LIM].cpu().numpy(force=True), cmap='viridis', vmin=0)
# %%
# W_p = ((Q[:,:64] @ torch.diag(S) @ torch.linalg.inv(Q)[:64]).real)
W_p =  ((Q @ torch.diag(S) @ Q.T) + W_skew)
W_p_sym = (W_p + W_p.T) / 2
W_p_skew = (W_p - W_p.T) / 2


torch.dist(W_p_sym, W_sym).item(), torch.dist(W_p_skew, W_skew).item(), torch.dist(W_p, W_QK).item()

# %%
# Gridsearch on scale value and threshold
# %%
@torch.no_grad()
def compute_rank(attn: Float[Tensor, "seq seq"]) -> tuple[Int[Tensor, "seq"], Int[Tensor, "seq seq"]]:
    """Returns the rank of the diagonal element among all the other elements of the attention matrix using torch only (CPU)."""
    dev = attn.device
    attn = attn.to('cpu')
    # Get the sorted indices for each row (descending order)
    sorted_attn = torch.argsort(attn, dim=-1, descending=True)
    # For each row, find the rank (position) of the diagonal element
    seq_len = attn.size(0)
    idx = torch.arange(seq_len, device=dev)
    sorted_attn = sorted_attn.to(dev)
    # For each row, where is the diagonal index in the sorted list?
    rank = (sorted_attn == idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    return rank, sorted_attn

def mrr(rank: Int[Tensor, "seq"]) -> float:
    """Computes the Mean Reciprocal Rank given the rank of the diagonal elements"""
    return (1 / (rank + 1)).mean().item()

def accuracy_k(rank: Int[Tensor, "seq"], k=32) -> float:
    """Computes the accuracy at k given the rank of the diagonal elements"""
    return (rank < k).float().mean().item()

def precision_k(pred_sorted: Int[Tensor, "seq seq"], gt_sorted: Int[Tensor, "seq seq"], k : int = 128):
    "computes how many relevant items are retrieved within the top-k"
    seq_len = pred_sorted.size(0)
    return (((pred_sorted < k) * (gt_sorted < k)).sum() / k / seq_len).item()

def rank_increase(rank: Int[Tensor, "seq"], base_rank: Int[Tensor, "seq"]) -> float:
    """Computes the rank increase given the rank of the diagonal elements"""
    return (base_rank - rank).float().mean().item()

def spearman(pred_sorted: Int[Tensor, "seq seq"], gt_sorted: Int[Tensor, "seq seq"]) -> float:
    """Computes the row-wise correlation coefficient between the predicted and ground truth ranks"""
    # sequence length
    seq_len = pred_sorted.size(-1)

    # prepare rank matrices
    device = pred_sorted.device
    idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(pred_sorted.size(0), -1)  # (seq, seq)
    pred_ranks = torch.zeros_like(pred_sorted, device=device)
    gt_ranks   = torch.zeros_like(gt_sorted, device=device)
    # scatter the rank positions
    pred_ranks.scatter_(1, pred_sorted, idx)
    gt_ranks.scatter_(1,   gt_sorted,   idx)
    # compute squared differences of ranks
    d2 = (pred_ranks - gt_ranks).pow(2).sum(dim=1)  # sum over each row
    # spearman formula per row: ρ = 1 − (6 * Σd²) / [n(n²−1)]
    n = seq_len
    rho = 1 - (6 * d2) / (n * (n*n - 1))
    # return mean over rows as Python float
    return rho.mean().item()

# %%
from tqdm import tqdm
DEV = device
W_Q = model.blocks[1].attn.W_Q[5].clone().detach().to(DEV)
W_K = model.blocks[1].attn.W_K[5].clone().detach().to(DEV)
S, V = torch.linalg.eig(W_Q @ W_K.T)
U_Q = V[:,:64]
U_K = torch.linalg.inv(V)[:64]
# Zero out noise components
S = S[:64]

LIM = attn_in.shape[0]
E = attn_in[:LIM].to(DEV)

# attn_svd = E @ U_Q @ torch.diag(S) @ U_K @ E.T
gt_rank, ideal_attn_sorted = compute_rank(attn_ideal[:LIM,:LIM].to('cpu'))
components = torch.argsort(S.real, descending=False)[:64]

# %%
records = []

with torch.no_grad():
    for k in tqdm([1,2,3,4,8,16,24,32]):
        for v in np.arange(-5,1.3,0.25):
            ablated = components[:k]
            s = S.clone().detach().to(DEV)
            s[ablated] *= v

            attn_svd_abl = E @ (U_Q @ torch.diag(s) @ U_K).real @ E.T

            rank, attn_sorted = compute_rank(attn_svd_abl.to('cpu'))
            
            records.append({
                'scale': round(v, 2),
                'k': k,
                'rank': rank.float().mean().item(),
                # 'rank_increase': rank_increase(rank, base_rank),
                'spearman': spearman(attn_sorted, ideal_attn_sorted),
                'mrr': mrr(rank),
                'accuracy_1024': accuracy_k(rank, k=1024),
                'accuracy_256': accuracy_k(rank, k=256),
                'accuracy_128': accuracy_k(rank, k=128),
                'precision_32': precision_k(rank, gt_rank, k=32),
                'precision_64': precision_k(rank, gt_rank, k=64),
                'precision_128': precision_k(rank, gt_rank, k=128),
                'precision_256': precision_k(rank, gt_rank, k=256),
                'precision_512': precision_k(rank, gt_rank, k=512),
                'precision_1024': precision_k(rank, gt_rank, k=1024),
            })
            print(records[-1])
            del attn_sorted, rank, attn_svd_abl, s 
            clean_mem()

# Save records to pandas and csv
import pandas as pd
import seaborn as sns
df = pd.DataFrame(records)
df.scale = df.scale.round(2)
df.to_csv("attention_records.csv", index=False)

# pairwise_sim = (U_Q @ U_K)
# sns.histplot(pairwise_sim.diagonal().numpy(force=True), bins=25)
# %% 
import matplotlib.pyplot as plt
df = pd.read_csv('attention_records.csv')
# List of columns to plot (excluding 'scale' and 'k')
value_columns = df.columns.to_list()
value_columns.remove('k')
value_columns.remove("scale")

n_cols = 3
n_rows = (len(value_columns) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

for idx, col in enumerate(value_columns):
    ax = axes[idx // n_cols][idx % n_cols]
    pivot = df.pivot(index='scale', columns='k', values=col)
    sns.heatmap(pivot, ax=ax, cmap='viridis')
    ax.set_title(col)
    ax.set_xlabel('k')
    ax.set_ylabel('scale')

# Hide any unused subplots
for i in range(len(value_columns), n_rows * n_cols):
    axes[i // n_cols][i % n_cols].axis('off')

plt.tight_layout()
plt.show()
# %%
