# %%
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

from sae_lens import HookedSAETransformer
from src.ablation import get_vocab_attn1_input
from src.utils import clean_mem
from src.constants import GROUP_TOKENS

# %load_ext autoreload
# %autoreload 2

device = 'cuda:0'
torch.cuda.set_device(0)

LAYER = 1
HEAD_IDX = 5
model = HookedSAETransformer.from_pretrained("gpt2-small")

# %%
# Get residual stream vectors for the entire vocabulary
attn_in = get_vocab_attn1_input(model).clone().detach()
E = attn_in.to(device)
group_ids = list(chain(*model.tokenizer(GROUP_TOKENS).input_ids))

# %% [markdown]
# # Skew-Symmetric Component Analysis
# 
# This section explores the hypothesis that a skew-symmetric component in the `W_QK` matrix is responsible for the head's tendency to attend to subsequent tokens in a sequence (e.g., "Monday" -> "Tuesday").

# %%
def random_skew_symmetric_W(D, device = device):
    """Generate a full-rank skew-symmetric matrix W ∈ ℝ^{D×D} (D must be even)."""
    assert D % 2 == 0, "D must be even to get full-rank skew-symmetric matrix."
    X = torch.randn(D, D, device = device)
    W = X - X.T
    return W

W_skew_random = random_skew_symmetric_W(model.cfg.d_model)
attn_skew =  E @ W_skew_random @ E.T

sns.heatmap(attn_skew[group_ids][:,group_ids].numpy(force=True),
    xticklabels=GROUP_TOKENS,
    yticklabels = GROUP_TOKENS,
)
plt.title("Attention from a random skew-symmetric W_QK")
plt.show()
del attn_skew, W_skew_random
clean_mem()

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
    A = A[:, perm]
    B = B[:, perm]
    return A, B

A, B = random_skew_factors(model.cfg.d_model, 64)
W_not_self_approx = A @ B.T
attn_skew_approx =  E @ W_not_self_approx @ E.T
sns.heatmap(attn_skew_approx[group_ids][:,group_ids].numpy(force=True),
    xticklabels=GROUP_TOKENS,
    yticklabels = GROUP_TOKENS,
)
plt.title("Attention from a low-rank random skew-symmetric W_QK")
plt.show()

# %%
def random_symmetric_W(D, device = device, n_negative_eigenvalues=0, rank=64):
    """Generate a full-rank symmetric matrix W ∈ ℝ^{D×D}."""
    # Generate random matrix and orthogonalize it with QR decomposition
    X = torch.randn(D, rank, device=device)
    P, _ = torch.linalg.qr(X)  # P will be orthonormal with shape (D, rank)
    S = torch.rand(rank, device = device)
    S[:n_negative_eigenvalues] *= -1  # Make some eigenvalues negative
    return P @ torch.diag(S) @ P.T

W_sym_random = random_symmetric_W(model.cfg.d_model, n_negative_eigenvalues=33)
attn_sym =  E @ W_sym_random @ E.T
sns.heatmap(attn_sym[group_ids][:,group_ids].numpy(force=True),
    xticklabels=GROUP_TOKENS,
    yticklabels = GROUP_TOKENS,
)
plt.title("Attention from a random symmetric W_QK")
plt.show()
del attn_sym, W_sym_random
# %%
