# %%
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import seaborn as sns

from itertools import chain
from src.ablation import get_vocab_attn1_input
from src.utils import display_most_attended_tokens
from src.constants import GROUP_TOKENS

from transformer_lens import HookedTransformer

device = 'cuda:0'
torch.cuda.set_device(0)

LAYER = 1
HEAD_IDX = 5
model = HookedTransformer.from_pretrained("gpt2-small")

# %% [markdown]
# # Gradient-based Steering of Self-Attention
# 
# This experiment investigates whether we can control the self-attention behavior of head L1H5 by directly manipulating its weights using gradients. We define a loss function that encourages or discourages self-attention and apply gradient ascent/descent to the key and query weights.

# %%
# Get residual stream vectors for the entire vocabulary
attn_in = get_vocab_attn1_input(model).clone().detach()

# %%
def selfloss(scores : torch.Tensor):
    self_score = scores.diagonal()
    total_score = scores.sum(dim=1)
    return - (self_score / total_score).mean()

def softloss(scores : torch.Tensor):
    # computes the softmax
    return - torch.log(F.softmax(scores, dim=0).diagonal() + 1e-12).mean()  

def bilinear_loss(W_QK: torch.Tensor):
    return -W_QK.diagonal().sum()

# %%
# Compute gradients for the bilinear loss
W_Q = model.blocks[LAYER].attn.W_Q[HEAD_IDX].clone().detach().to(device).requires_grad_(True)
W_K = model.blocks[LAYER].attn.W_K[HEAD_IDX].clone().detach().to(device).requires_grad_(True)

loss = bilinear_loss(W_Q @ W_K.T)
loss.backward()

# %% [markdown]
# ### Boosting self-attention
# Subtracting the gradient of the "bilinear" loss from the key matrix `W_K` boosts self-attention.

# %%
wk =  W_K.clone().to(device).detach()
wq =  W_Q.clone().to(device).detach()
E = attn_in.to(device)

wk -= W_K.grad.to(device)
# wq -= W_Q.grad.to(device) # Note: Modifying W_Q has a similar effect
attn_boosted = E @ wq @ wk.T @ E.T 

group_toks_ids = model.tokenizer(GROUP_TOKENS, add_special_tokens=False).input_ids
group_toks_ids = list(chain(*group_toks_ids))

sns.heatmap(attn_boosted[group_toks_ids][:,group_toks_ids].numpy(force=True),
    xticklabels=GROUP_TOKENS,
    yticklabels = GROUP_TOKENS,
    vmin=0, 
)
plt.title("Self-Attention Boosted via Gradient Descent")
plt.show()

display_most_attended_tokens(GROUP_TOKENS[:1], attn_boosted, k=10, tokenizer=model.tokenizer)

# %% [markdown]
# ### Suppressing self-attention
# Adding the gradient (gradient ascent) suppresses self-attention, restoring the original behavior. A scaling factor is used to avoid introducing too much noise.

# %%
wk =  W_K.clone().to(device).detach()
wq =  W_Q.clone().to(device).detach()

wk += W_K.grad.to(device) * 0.1
wq += W_Q.grad.to(device) * 0.1
attn_suppressed = E @ wq @ wk.T @ E.T 

sns.heatmap(attn_suppressed[group_toks_ids][:,group_toks_ids].numpy(force=True),
    xticklabels=GROUP_TOKENS,
    yticklabels = GROUP_TOKENS, vmin=0,
)
plt.title("Self-Attention Suppressed via Gradient Ascent")
plt.show()

display_most_attended_tokens(GROUP_TOKENS[:1], attn_suppressed, k=10, tokenizer=model.tokenizer)