# %%
from matplotlib import pyplot as plt
import torch
import numpy as np
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

from sae_lens import HookedSAETransformer, SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display

import torch.nn.functional as F
import random

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

LAYER = 1
HEAD_IDX= 5
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"
model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)


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
    vocab = torch.arange(0, VOCAB_SIZE)
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


attn_in = vocab_attn(model).clone()

# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()

# Resid similarity 
# q = F.layer_norm(attn_in, (768,)) #@ W_Q
# k = F.layer_norm(attn_in, (768,)) #@ W_K

# QK similarity
# q = attn_in @ W_Q
# k = attn_in @ W_K

# %%
attn = FactoredMatrix(k,q.T).AB

attn.shape
# %% [markdown]
# Plot first 32 items of the whole attention matrix. 

# **Observation**: All the digits are clustered together, each digits attend strongly to its successor, normal to other digits, low on itself, very low on all other tokens

# %%
import seaborn as sns
from rich.table import Table
from rich.console import Console
L=32
sns.heatmap(attn[:L, :L].numpy(force=True),
    xticklabels=model.tokenizer.batch_decode(torch.arange(L)),
    yticklabels = model.tokenizer.batch_decode(torch.arange(L))
)
# %%

def get_most_attended_tokens(token:str, attn:Tensor, k=16, tokenizer: PreTrainedTokenizerBase=model.tokenizer):
    tok_idx = tokenizer(token).input_ids
    
    assert len(tok_idx) == 1, f'Can only handle one token at a time. Currently {tok_idx}'
    tok_idx = tok_idx[0]
    
    scores = attn[tok_idx]
    max_val, max_idx = torch.topk(scores, k, largest=True, sorted=True)
    return tokenizer.batch_decode(max_idx), max_val
    
def display_most_attended_tokens(tokens, attn, k=10, tokenizer=model.tokenizer):
    table = Table(title="Most Attended Tokens")
    table.add_column("Input Token", style="bold")
    table.add_column("Top Attended Tokens", style="dim")
    for token in tokens:
        top_tokens, scores = get_most_attended_tokens(token, attn, k=k, tokenizer=tokenizer)
        table.add_row(
            repr(token),
            ", ".join(top_tokens),
        )
    console = Console()
    console.print(table)

# Example usage:
tokens = [
    ' red',
    ' 69',
    'Monday',
    ' John',
    ' +',
    ' if',
    ' Italy',
    # ' </'
]
display_most_attended_tokens(tokens, attn, k=10)

# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()
W_QK_ideal = W_Q @ W_K.T
W_QK_ideal.fill_diagonal_(2)
attn_ideal = attn_in @ W_QK_ideal @ attn_in.T

display_most_attended_tokens(tokens, attn_ideal, k=10)

# %%
pairwise_sim = (W_Q.T @ W_K)

barplot(torch.diagonal(pairwise_sim), ylabel='cos(θ)', title='Similarity between W_K and W_Q rows')
# %%
torch.cuda.empty_cache()

group_tokens = [
    ' red',
    ' Green',
    ' Blue', 
    'blue', 
    'orange', 
    ' brown', 
    ' 49', 
    ' 52', 
    ' 56', 
    ' 68', 
    ' 69',  # numbers
    ' 82', 
    'Monday', # week days
    'Friday', 
    ' weekend', 
    'Week', 
    'Wednesday', 
    ' yesterday', 
    ' John',  # names
    ' Richard',
    ' Liam', 
    ' Anne', 
    ' Sophie',
    ' Sarah',
    ' Italy', # countries
    ' Iceland', 
    ' Austria',
    ' Mexico',
    ' Spain', 
    ' France' 
]

group_toks=model.tokenizer(group_tokens).input_ids
group_toks = list(chain(*group_toks))

sns.heatmap(attn_ideal[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens,
    vmin=0,
)
# %%
def selfloss(scores : Tensor):
    self_score = scores.diagonal()
    total_score = scores.sum(dim=1)
    return - (self_score / total_score).mean()

def softloss(scores : Tensor):
    # computes the softmax
    return - torch.log(F.softmax(scores, dim=0).diagonal() + 1e-12).mean()  
# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach().to(device).requires_grad_(True)
W_K = model.blocks[1].attn.W_K[5].clone().detach().to(device).requires_grad_(True)
E = attn_in.detach().to(device)#    .requires_grad_(True)  # [n_tokens, d_model]

def bilinear_loss(W_QK: Tensor):
    return -W_QK.diagonal().sum()

# loss = selfloss(E @ W_Q @ W_K.T @ E.T)
loss = bilinear_loss(W_Q @ W_K.T)
W_Q.grad = None
W_K.grad = None
loss.backward()
# grad_Q = E.grad @ W_Q       # ∂L/∂Q
# grad_K = E.grad @ W_K       # ∂L/∂K

# importance_Q = grad_Q.mean(dim=0)  # [d_head]
# importance_K = grad_K.mean(dim=0)
# %%
# Barplot of row importance is almost flat
barplot(W_Q.grad.pow(2).mean(dim=0))

# plot histogram of gradient values: is basically a normal distribution
sns.histplot(W_Q.grad.flatten().abs().cpu().numpy(), bins=50).set(yscale='log')

# %%
wk =  W_K.clone().to('cuda:0').detach()
wq =  W_Q.clone().to('cuda:0').detach()
attn_in = attn_in.to('cuda:0')

wk -= W_K.grad.to('cuda:0')
# wq -= W_Q.grad.to('cuda:0')
attn_ablated = attn_in @ wq @ wk.T @ attn_in.T 

sns.heatmap(attn_ablated[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens,
    vmin=0, 
    # vmax=180    
)

print(display_most_attended_tokens(tokens, attn_ablated, k=10))
# %% [markdown]
# **Observation**: I can succesfully boost the self-attention by subtratcting the gradient of the "bilinear" loss. 
# %%
# Now elicit
wk =  W_K.clone().to('cuda:1').detach()
wq =  W_Q.clone().to('cuda:1').detach()
attn_in = attn_in.to('cuda:1')

wk += W_K.grad.to('cuda:1') * 0.1
wq += W_Q.grad.to('cuda:1') * 0.1
attn_ablated = attn_in @ wq @ wk.T @ attn_in.T 

sns.heatmap(attn_ablated[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens, vmin=0,
)

print(display_most_attended_tokens(tokens, attn_ablated, k=10))
del wk, wq, attn_ablated
torch.cuda.empty_cache()

# %% [markdown]
# **Observation**: When adding it instead it elicits the self-suppression, but I have to downscale it to avoid introducing too much noise, otherwise it also stops attending similar tokens as well

# ### SVD decomposition:
# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach().to(device)
W_K = model.blocks[1].attn.W_K[5].clone().detach().to(device)
U_Q, S, U_K = torch.linalg.svd(W_Q @ W_K.T,)

# Zero out noise components
S = S[:64]
U_Q = U_Q[:, :64]
U_K = U_K[:64, :]


print(torch.dist(W_Q @ W_K.T, U_Q @ torch.diag(S) @ U_K))

attn_svd = attn_in @ U_Q @ torch.diag(S) @ U_K @ attn_in.T

sns.heatmap(attn_svd[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens, vmin=0,
)

print(display_most_attended_tokens(tokens, attn_svd, k=10))

# %%
pairwise_sim = (U_Q.T @ U_K.T)
barplot(torch.diagonal(pairwise_sim) * S)
# %%
# %%
s = S.clone().detach()
ablated = torch.diagonal(pairwise_sim) * S < -0.5
s[ablated] *= -5
attn_svd_abl = attn_in @ U_Q @ torch.diag(s) @ U_K @ attn_in.T

sns.heatmap(attn_svd_abl[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens, vmin=0
)

display_most_attended_tokens(tokens, attn_svd_abl, k=10)
# %% [markdown]
# I'm somewhat able to boost the self-attention by manipulating the singular values.
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

def precision_k(rank: Int[Tensor, "seq"], gt_rank: Int[Tensor, "seq"], k : int = 128):
    "computes how many relevant items are retrieved within the top-k"
    return ((rank < k) * (gt_rank < k)).sum() / k

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
DEV = 'cuda:1'
W_Q = model.blocks[1].attn.W_Q[5].clone().detach().to(DEV)
W_K = model.blocks[1].attn.W_K[5].clone().detach().to(DEV)
U_Q, S, U_K = torch.linalg.svd(W_Q @ W_K.T,)

# Zero out noise components
S = S[:64]
U_Q = U_Q[:, :64]
U_K = U_K[:64, :]

LIM = 32768 # attn_in.shape[0]
E = attn_in[:LIM,:LIM].to(DEV)

# attn_svd = E @ U_Q @ torch.diag(S) @ U_K @ E.T
base_rank, ideal_attn_sorted = compute_rank(attn_ideal[:LIM,:LIM].to('cpu'))

pairwise_sim = (U_Q.T @ U_K.T)
components = torch.argsort(torch.diagonal(pairwise_sim) * S).numpy(force=True)

records = []

with torch.no_grad():
    for k in tqdm([1,2,3,4,6,10,16,24,36]):
        for v in np.arange(-5,1.3,0.1):
            ablated = components[:k]
            s = S.clone().detach().to(DEV)
            s[ablated] *= v

            attn_svd_abl = E @ U_Q @ torch.diag(s) @ U_K @ E.T

            rank, attn_sorted = compute_rank(attn_svd_abl.to('cpu'))
            
            records.append({
                'scale': v,
                'k': k,
                'rank': rank.float().mean().item(),
                # 'rank_increase': rank_increase(rank, base_rank),
                'spearman': spearman(attn_sorted, ideal_attn_sorted),
                'mrr': mrr(rank),
                'accuracy_1024': accuracy_k(rank, k=1024),
                'accuracy_256': accuracy_k(rank, k=256),
                'accuracy_128': accuracy_k(rank, k=128),
                'precision_32': precision_k(rank, base_rank, k=32),
                'precision_64': precision_k(rank, base_rank, k=64),
                'precision_128': precision_k(rank, base_rank, k=128),
                'precision_256': precision_k(rank, base_rank, k=256),
                'precision_512': precision_k(rank, base_rank, k=512),
                'precision_1024': precision_k(rank, base_rank, k=1024),
            })
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

# List of columns to plot (excluding 'scale' and 'k')
value_columns = df.columns.to_list()
value_columns.remove('k')
value_columns.remove("score")

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
# uq = U_Q.clone().detach()
# uk = U_K.clone().detach()
# uq[ablated] *= 10
# uk[:,ablated] *= -10
# attn_svd_abl = attn_in @ uq @ torch.diag(s) @ uk @ attn_in.T
# %% [markdown]
# Manually compute the similarity of each key and query item 
# %%
# E = attn_in.to('cuda:0')
# Q = E @ W_Q
# K = E @ W_K

# S = Q*K

# mu = S.mean(dim=0)
# std = S.std(dim=0)

# barplot(mu)
# barplot(std)
# del E, Q, K, S

# %% [markdown]
# test what happens when manually incrementing the diagonal W_QK
# **observation**: After a manual increase of ~0.2 the head starts attending itself

# %%
W_QK = W_Q @ W_K.T

wqk = W_QK.clone()
for i in range(10):
    wqk[torch.arange(768), torch.arange(768)] += 0.1
    a =  attn_in @ wqk @ attn_in.T
    print(get_most_attended_tokens(' red',a)[0])
# %%

# %%


def random_skew_symmetric_W(D, device = device):
    """Generate a full-rank skew-symmetric matrix W ∈ ℝ^{D×D} (D must be even)."""
    assert D % 2 == 0, "D must be even to get full-rank skew-symmetric matrix."
    X = torch.randn(D, D, device = device)
    W = X - X.T
    return W


not_self_a =  attn_in @ random_skew_symmetric_W(768) @ attn_in.T
sns.heatmap(not_self_a[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens,
)
del not_self_a



# %%
def random_skew_factors(D, r, k=None, device=device):
    """
    Generate A, B ∈ ℝ^{D×2r} such that A @ B.T is skew-symmetric of rank ≤ 2r.
    """
    r = r // 2
    if k is None:
        k = r
    X = torch.randn(D, r, device=device)
    Y = torch.randn(D, r, device=device)
    A = torch.cat([X, Y, torch.randn(D,k, device=device)], dim=1)
    B = torch.cat([Y,-X, torch.randn(D,k, device=device)], dim=1)
    # permute columns of both A,B:
    perm = torch.randperm(2 * r + k)
    print(perm)
    A = A[:, perm]
    B = B[:, perm]
    return A, B


A, B = random_skew_factors(768, 64)

W_not_self_approx = A @ B.T
not_self_a =  attn_in @ W_not_self_approx @ attn_in.T
sns.heatmap(not_self_a[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens,
)
# %%
W = random_skew_symmetric_W(64).numpy(force=True)
A, B = random_skew_factors(64, 8)
W = (A @ B.T).numpy(force=True)


sns.heatmap(W @ W)
# %%
W_QK = (W_Q @ W_K.T).cpu()
W_QK_L = W_QK[:32,:32]

sns.heatmap(
    (W_QK_L @ W_QK_L).numpy(force=True)
)
# %%
A, B = random_skew_factors(6, 4)
W = (A @ B.T).numpy(force=True)
s = (W @ W)
# s = (W_QK @ W_QK)
m = np.diagonal(s) <= s.min(axis=0)

m.sum()
# %%
H, D = 64, 20
# A, B = random_skew_factors(H,D, device='cpu')
# print(A.shape)
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()
A, B = W_K, W_Q
A_n = A / A.norm(dim=0, keepdim=True)
B_n = B / B.norm(dim=0, keepdim=True)
A_n, B_n = A_n.numpy(force=True), B_n.numpy(force=True)
# sns.heatmap(A.T @ B)
# plt.show()
M = A_n.T @ B_n
# sns.heatmap(M - M.T)
m = np.argsort(np.abs((M - M.T)).max(axis=0))[::-1]
sns.heatmap(np.abs(M- M.T))
plt.show()
print(m)
W = A @ B.T
maxval, minval = (W@W).max(), (W@W).min()

W = (A @ B.T).numpy(force=True)
sns.heatmap((W@W)[:128,:128], vmin=minval, vmax=maxval)

for i in m[:32]:
    # j = m[i]
    A[:,i] *= 10

plt.show()
# A_n = A / A.norm(dim=0, keepdim=True)
# B_n = B / B.norm(dim=0, keepdim=True)
# A_n, B_n = A_n.numpy(force=True), B_n.numpy(force=True)
W = (A @ B.T).numpy(force=True)
sns.heatmap((W@W)[:128,:128], vmin=minval, vmax=maxval)

plt.show()
torch.cuda.empty_cache()
# wk = W_K.clone()
# wk[:, [21,51,32,39,37,24]] = 0
# wk[:,[2,17,19,20,36,54,4,29]] *= 0
# wk[min_diag] *= 2
# wk[min_diag] = W_Q[min_diag]
# wk[:,diag<0]
k_abl = attn_in @ B
q = attn_in @ A
attn_ablated = q@ k_abl.T

sns.heatmap(attn_ablated[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens,
    vmin=0,)
# %% [markdown]

# # Exp: Column ablation
# %%
D_HEAD = 64
q = attn_in @ W_Q

diff = torch.zeros(D_HEAD)
diag_diff = torch.zeros(D_HEAD)
all_but_diag = torch.zeros(D_HEAD)

for i in range(D_HEAD):
    wk = W_K.clone()
    wk[:, i] = 0
    
    k_abl = attn_in @ wk
    attn_ablated = FactoredMatrix(q, k_abl.T).AB
    attn_ablated /= attn_ablated.sum(dim=-1)
    d = (attn_ablated - attn / attn.sum(dim=-1))#.abs()
    diff[i] = d.mean()
    diag_diff[i] = torch.diagonal(d).mean()
    d.fill_diagonal_(0.)
    all_but_diag[i] = d.mean()
    
    del attn_ablated, d, k_abl, wk
    
# %%
barplot(diff, title="Difference when ablating a specific row", ylabel='Inre ')
# %%
barplot(all_but_diag, title="Difference when ablating a specific row", ylabel='difference in every element but diagonal')
# %%
barplot(diag_diff, title="Increase in self-attention when ablating a specific row", ylabel='diagonal difference')
# %%



