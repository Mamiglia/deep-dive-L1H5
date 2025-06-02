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
    
    for var in ['loss', 'E']:
        if var in locals():
            del locals()[var]
        elif var in globals():
            del globals()[var]
        
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


attn_in = vocab_attn(model)

# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()

# Resid similarity 
# q = F.layer_norm(attn_in, (768,)) #@ W_Q
# k = F.layer_norm(attn_in, (768,)) #@ W_K

# QK similarity
q = attn_in @ W_Q
k = attn_in @ W_K

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

sns.heatmap(attn[group_toks][:,group_toks].numpy(force=True),
    xticklabels=group_tokens,
    yticklabels = group_tokens,
    vmin=0,
)
# %%
W_QK = W_Q @ W_K.T

wqk = W_QK.clone()
for i in range(10):
    wqk[torch.arange(768), torch.arange(768)] += 0.1
    a =  attn_in @ wqk @ attn_in.T
    print(get_most_attended_tokens(' red',a)[0])
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



