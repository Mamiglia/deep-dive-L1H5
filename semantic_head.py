# %%
from typing import Callable
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

from jaxtyping import Float, Int
import circuitsvis as cv

# %load_ext autoreload
# %autoreload 2

from src.maps import *
from src.utils import *

LAYER = 1
HEAD_IDX= 5
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"

# %%

model, sae, d_sae, d_model = load_model_sae(sae_id=HOOK_POINT)
W_KQ = get_kq(model, layer=LAYER, head_idx=HEAD_IDX)
text, _ = random_toks_with_keywords(model, "colors")
_, cache = model.run_with_cache(text, remove_batch_dim=True)
seq_len = text.shape[-1]

# %%
str_tokens = model.tokenizer.convert_ids_to_tokens(text[0])
for layer in [LAYER]:
    print("Layer:", layer)
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))
# %%
resid = cache[HOOK_POINT]
resid = F.layer_norm(resid, (d_model, ))

W_Q = model.blocks[LAYER].attn.W_Q[HEAD_IDX]

norms = (resid @ W_Q).norm(dim=-1)
# plot
plt.scatter(norms.numpy(force=True), str_tokens, )

# feats = sae.encode(resid)
# # seq features
# feat_idx, _ = topk_feats(feats, 32)

# most_attended_token = torch.argmax(cache["pattern", LAYER][HEAD_IDX], dim=-1)
# assert torch.all(most_attended_token <= torch.arange(0,seq_len, device=device)), "Cannot attend to future tokens"

# gt_feat_idx = feat_idx[most_attended_token] # seq k
# %%
for layer in [LAYER]:
    print("Layer:", layer)
    attention_pattern = resid 
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))
    
# %%
attn = resid @ resid.T

# mask attn
mask = torch.triu(torch.ones_like(attn, dtype=torch.bool))
mask[0,0] = False
attn[mask] = float('-inf')
attn = F.softmax(attn, dim=-1)

display(cv.attention.attention_patterns(tokens=str_tokens, attention=attn.unsqueeze(0)))
# %%
KEYWORDS = {
    "colors": ["blue", "red", "green", "yellow", "purple"],
    "animals": ["dog", "cat", "mouse", "horse", "sheep"],
    "fruits": ["apple", "banana", "grape", "peach", "lemon"],
    "emotions": ["happy", "sad", "angry", "scared", "proud"],
    "weather": ["rain", "snow", "wind", "storm", "sun"],
    "numbers": ["2", "69", "4", "22", "32", "50"]
}

def build_prompt(seq_len=64) -> Tuple[list[str], Int[Tensor, "seq seq"]]:
    """Build a prompt of randomly sampled tokens from categories by shuffling them together.

    The tokens are sampled from the global KEYWORDS dictionary. The prompt
    is a list of strings (tokens). The expected attention pattern indicates
    which tokens in the prompt belong to the same category and are therefore
    expected to attend to each other.

    Args:
        seq_len (int, optional): The desired length of the token sequence in the prompt. Defaults to 64.

    Returns:
        tuple: A tuple containing:
            - constructed_prompt (list[str]): A list of token strings forming the prompt.
            - attention_pattern (Tensor): An N x N matrix (where N is seq_len)
              representing the expected attention. A[i][j] is 1 if tokens at
              position i and j in the prompt belong to the same category, else 0.
              Returns empty lists if seq_len is 0 or KEYWORDS is empty.
    """
    if seq_len == 0 or not KEYWORDS:
        return [], torch.tensor([])

    category_list = list(KEYWORDS.keys())
    constructed_prompt = []
    category_assignments = []

    # Randomly sample tokens to fill the prompt
    while len(constructed_prompt) < seq_len:
        category = random.choice(category_list)
        token = random.choice(KEYWORDS[category])
        constructed_prompt.append(' '+ token)
        category_assignments.append(category)

    # Trim if over
    constructed_prompt = constructed_prompt[:seq_len]
    category_assignments = category_assignments[:seq_len]

    # Build attention pattern
    attention_pattern = torch.zeros(seq_len+1, seq_len+1, dtype=torch.int)

    for i in range(1,seq_len):
        for j in range(1,seq_len):
            if category_assignments[i-1] == category_assignments[j-1]:
                attention_pattern[i, j] = constructed_prompt[i-1] != constructed_prompt[j-1]
                
    # if no attn then attend <bos>
                
    # Mask attention
    mask = torch.triu(attention_pattern>0, 0)
    attention_pattern[mask] = 0
    
    attention_pattern[:,0][attention_pattern.sum(dim=-1) == 0] = 1
    
    return constructed_prompt, attention_pattern

prompt, attn = build_prompt(seq_len=63)
attn.sum(dim=-1)    
# %%
toks = model.tokenizer(prompt).input_ids
toks = list(chain(*toks))
if toks[0] != model.tokenizer.bos_token_id:
    toks.insert(0, model.tokenizer.bos_token_id)
toks = torch.tensor(toks)

_, cache = model.run_with_cache(toks, remove_batch_dim=True)

seq_len = text.shape[-1]

str_tokens = model.tokenizer.convert_ids_to_tokens(toks)
for layer in [LAYER]:
    print("Layer:", layer)
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))


# %%
attn = attn.to(device)
gt_attn = cache["pattern", layer]


num_heads  = 12
explained_attn = np.zeros(num_heads)
unexplained_attn = np.zeros(num_heads)
for head_idx in range(num_heads):
    total_attn = gt_attn[head_idx].sum()
    explained_attn[head_idx] = (gt_attn[head_idx] * attn).sum() / total_attn
    unexplained_attn[head_idx] = (gt_attn[head_idx]*(1 - attn)).sum() / total_attn
    
plt.plot(explained_attn)
plt.plot(unexplained_attn)

def explained_attn_score(
    gt_attn: Float[Tensor, "*batch n_head seq seq"], 
    pred_attn: Float[Tensor, "*batch seq seq"]
) -> Float[Tensor, "*batch n_head"]:
    """
    Computes the negative log cumulative probability of assigning the attention to the expected pattern. 

    Args:
        gt_attn (Tensor): Ground truth attention patterns of shape (..., n_head, seq, seq),
            where ... can be batch dimensions.
        pred_attn (Tensor): Predicted attention patterns of shape (..., seq, seq),
            where ... matches the batch dimensions of gt_attn.

    Returns:
        Tensor: The explained/unexplained attention ratio of shape (..., n_head, seq).
    """
    assert gt_attn.shape[-2:] == pred_attn.shape[-2:], f"{gt_attn.shape=}, {pred_attn.shape=}"
    
    if gt_attn.ndim == 4 and pred_attn.ndim ==3:
        pred_attn = pred_attn.unsqueeze(-3)
        
    prob_q = (gt_attn * pred_attn).sum(dim=-1) # b n s

    surprise = - prob_q.log()
    return surprise.mean(dim=-1)

plt.plot(explained_attn_score(gt_attn.unsqueeze(0), attn)[0].numpy(force=True)
)

# %%
from tqdm import tqdm
from functools import partial

def ablate_component(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
):
    """Mean-ablate a component"""
    return torch.ones_like(t) * t.mean(dim=(0,1))

def explained_attn_score_metric(
    cache: ActivationCache,
    pred_attn
):
    attn = cache["blocks.1.attn.hook_pattern"]
    score = explained_attn_score(attn, pred_attn)
    return score[...,HEAD_IDX]

import torch.nn as nn

@torch.inference_mode()
def ablate_metric(
    model: HookedTransformer,
    batch,
    metric,
) -> Float[Tensor, "layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    components = [
        'hook_embed',
        'hook_pos_embed',
        'blocks.0.hook_attn_out',
        'blocks.0.hook_mlp_out',
        'blocks.0.hook_mlp_resid',
        'blocks.2.hook_mlp_out',
        'None'
    ]
    results = dict()
    out_hook = "blocks.1.attn.hook_pattern"
    
    # Hook function to cache activations
    def cache_hook(activation, hook):
        cache[hook.name] = activation

    for component in tqdm(components):
        model.reset_hooks()
        cache = dict()
        
        hooks : list[tuple[str,Callable]] = [(out_hook, cache_hook)]
        
        match component:
            case 'None':
                pass
            case 'blocks.0.hook_mlp_resid':
                replace_hk = lambda x,hook: cache['blocks.0.hook_mlp_out']
                hooks += [
                    ('blocks.0.hook_mlp_out', cache_hook),
                    ('blocks.0.hook_resid_post', replace_hk)
                ]
            case _:
                hooks.append((component, ablate_component))
            
        model.run_with_hooks(batch,
            fwd_hooks=hooks)
        
        res = metric(cache)
        results[component] = (res.mean().item(), res.std().item())
        del cache

    model.reset_hooks()
    return results

# %% 
BATCH_SIZE = 128 
SEQ_LEN = 64

batch = torch.empty((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
attn_batch = torch.empty((BATCH_SIZE, SEQ_LEN, SEQ_LEN), dtype=torch.long)
for b in range(BATCH_SIZE):
    
    prompt, attn = build_prompt(seq_len=SEQ_LEN-1)

    toks = model.tokenizer(prompt).input_ids
    toks = list(chain(*toks))
    if toks[0] != model.tokenizer.bos_token_id:
        toks.insert(0, model.tokenizer.bos_token_id)
    toks = torch.tensor(toks)
    
    assert toks.shape[0] == SEQ_LEN     
    batch[b] = toks
    attn_batch[b] = attn

batch = batch.to(device)
attn_batch = attn_batch.to(device)
# %%

res = ablate_metric(
    model,
    batch,
    partial(explained_attn_score_metric, pred_attn=attn_batch)
)

# Unpack means and stds for each component
components = list(res.keys())
means = [res[c][0] for c in components]
stds = [res[c][1] for c in components]

plt.figure(figsize=(8, 5))
plt.barh(components, means, xerr=stds, color="skyblue", ecolor="gray")
plt.xlabel("Explained Attention Score (mean ± std)")
plt.title("Component Ablation Effect on Head 1.5")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comment:
# Components important for the head 1.5:
# - embedding matrix
# - mlp0
# What is the circuit responsible for this?
# 
# Why does it attend to the <bos> token when no other token of the same semantic group is present?
#
# Why does it not attend to itself or other occurences of the same token?
# %%
