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
    return t[:,:1024,:]

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
        ('blocks.0.hook_mlp_out', cache_hook),
        ('blocks.0.hook_resid_post', replace_hook ),
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
W_QK_ideal = W_Q @ W_K.T
W_QK_ideal.fill_diagonal_(2)
attn_ideal = attn_in @ W_QK_ideal @ attn_in.T

# display_most_attended_tokens(tokens, attn_ideal, k=10)
# %%
sns.heatmap(attn_ideal[:LIM,:LIM].cpu().numpy(force=True), cmap='viridis', vmin=0)

# %%
W_QK = W_Q @ W_K.T
W_sym = (W_QK + W_QK.T) / 2
W_skew = (W_QK - W_QK.T) / 2

attn_base = attn_in @ W_QK @ attn_in.T
attn_sym = attn_in @ W_sym @ attn_in.T
attn_skew = attn_in @ W_skew @ attn_in.T

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.heatmap(attn_base[:LIM, :LIM].cpu().numpy(force=True), cmap='magma', vmin=0, ax=axes[0])
axes[0].set_title("Base Attention")

sns.heatmap(attn_sym[:LIM, :LIM].cpu().numpy(force=True), cmap='magma', vmin=0, ax=axes[1])
axes[1].set_title("Symmetric Attention")

sns.heatmap(attn_skew[:LIM, :LIM].cpu().numpy(force=True), cmap='magma', vmin=0, ax=axes[2])
axes[2].set_title("Skew-Symmetric Attention")

plt.tight_layout()
plt.show()
# %%
true_eigenvalues = torch.linalg.eigvalsh(W_sym)
print("Eigenvalues of W_sym:", true_eigenvalues)
print("Number of negative eigenvalues:", torch.sum(true_eigenvalues < 0))
print("Number of positive eigenvalues:", torch.sum(true_eigenvalues > 0))
print("Min eigenvalue:", torch.min(true_eigenvalues))
print("Max eigenvalue:", torch.max(true_eigenvalues))
# %%
S, Q = torch.linalg.eigh(W_sym)

# zero negative S components
S[S < 0] *= -0.5
attn_svd = attn_in @ ((Q @ torch.diag(S) @ Q.T) + W_skew) @ attn_in.T
sns.heatmap(attn_svd[:LIM,:LIM].cpu().numpy(force=True), cmap='viridis', vmin=0)

# %%
S, Q = torch.linalg.eig(W_QK)
S = S[:64]  # Zero out noise components since W has rank 64
# Q = Q[:, :64]  # Keep only the first 64 eigenvectors

S[S.real < 0]*= 0 # Zero out negative eigenvalues


attn_svd = attn_in @ ((Q[:,:64] @ torch.diag(S) @ torch.linalg.inv(Q)[:64]).real) @ attn_in.T
sns.heatmap(attn_svd[:LIM,:LIM].cpu().numpy(force=True), cmap='viridis', vmin=0)
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
