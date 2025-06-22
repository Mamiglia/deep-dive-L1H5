# %%
from typing import Callable
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import numpy as np

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
import seaborn as sns
from itertools import chain
from jaxtyping import Float, Int
import circuitsvis as cv

from src.utils import load_model_sae
from src.prompt import random_toks_with_keywords, build_prompt
from src.metrics import explained_attn_score
from src.constants import KEYWORDS

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

from sae_lens import HookedSAETransformer, SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display

import torch.nn.functional as F

# %load_ext autoreload
# %autoreload 2

LAYER = 1
HEAD_IDX= 5
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"

# %%

model, sae, d_sae, d_model = load_model_sae(sae_id=HOOK_POINT)
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
# local KEYWORDS & build_prompt/tokenize_prompt replaced by src.prompt imports
# %%
attn = attn.to(device)
gt_attn = cache["pattern", layer].to(device)


num_heads  = 12
explained_attn = np.zeros(num_heads)
unexplained_attn = np.zeros(num_heads)
for head_idx in range(num_heads):
    total_attn = gt_attn[head_idx].sum()
    explained_attn[head_idx] = (gt_attn[head_idx] * attn).sum() / total_attn
    unexplained_attn[head_idx] = (gt_attn[head_idx]*(1 - attn)).sum() / total_attn

fig, ax1 = plt.subplots(figsize=(6, 3))
ax2 = ax1.twinx()
sns.scatterplot(
    y = explained_attn,
    x = np.arange(num_heads),
    label="Explained Attention",
    marker='o',
    color='blue',
    s=100,
    ax=ax1
)
sns.scatterplot(
    y = unexplained_attn,
    x = np.arange(num_heads),
    label="Unexplained Attention",
    marker='x',
    color='red',
    s=100,
    ax=ax1
).set(
    ylim=(0, 1),
)
plt.xticks(np.arange(num_heads), [f"Head {i}" for i in range(num_heads)], rotation=45)


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

sns.scatterplot(
    y = explained_attn_score(gt_attn.unsqueeze(0), attn)[0].numpy(force=True),
    x = np.arange(num_heads),
    label="Unexplained Attention Score",
    marker='*',
    color='green',
    s=100,
    ax=ax2
).set(
    ylim=(0, 3),
)

# %%
def get_explained_attention_scores(model, tokens, gt_attn_pattern):
    """
    Calculate explained attention scores for each head in each layer.
    
    Args:
        model: The transformer model
        tokens: Input token ids
        gt_attn_pattern: Ground truth attention pattern to check against
        
    Returns:
        explained_scores: Dictionary with layer -> head -> score mapping
    """
    # Run the model with caching
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    # Get the number of layers and heads
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # Store results
    explained_scores = {}
    
    # Iterate through all layers
    for layer in range(n_layers):
        explained_scores[layer] = {}
        layer_attn_patterns = cache["pattern", layer]
        
        # Iterate through all heads in the layer
        for head in range(n_heads):
            head_attn = layer_attn_patterns[head]
            
            # Calculate explained attention score
            score = explained_attn_score(
                head_attn.unsqueeze(0).unsqueeze(0),  # Add batch and head dimensions
                gt_attn_pattern
            )[0, 0].item()  # Extract the scalar value
            
            explained_scores[layer][head] = score
            
    return explained_scores

# Generate a test prompt and ground truth attention
test_prompt, test_attn = build_prompt(seq_len=255)

# Convert to tokens
test_tokens = model.tokenizer(test_prompt).input_ids
test_tokens = list(chain(*test_tokens))
if test_tokens[0] != model.tokenizer.bos_token_id:
    test_tokens.insert(0, model.tokenizer.bos_token_id)
test_tokens = torch.tensor(test_tokens, device=device)

# Get scores
scores = get_explained_attention_scores(model, test_tokens, test_attn.to(device))

# Convert to dataframe for visualization
score_data = []
for layer, heads in scores.items():
    for head, score in heads.items():
        score_data.append({
            'layer': layer,
            'head': head,
            'score': score
        })
import pandas as pd
score_df = pd.DataFrame(score_data)
# Plot as heatmap
plt.figure(figsize=(10, 8))
pivot_data = score_df.pivot(index='layer', columns='head', values='score')

# Create a mask to highlight head 5 in layer 1
highlight_mask = np.zeros_like(pivot_data, dtype=bool)
highlight_mask[1, 5] = True

# Create the heatmap with a border around the highlighted cell
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='rocket_r', vmax=5)

# Add a red border around head 5 in layer 1
plt.plot([5, 6, 6, 5, 5], [1, 1, 2, 2, 1], linewidth=3, color='green')

# Add an arrow pointing to the highlighted cell
plt.annotate('Target Head (L1H5)', xy=(5.5, 1.5), xytext=(6.5, 4), color='white',
             arrowprops=dict(facecolor='white', shrink=0.1, width=2, headwidth=8, connectionstyle="arc3,rad=-0.2"),
             fontsize=20, fontweight='bold')

plt.title('Surprisal by Layer and Head', fontsize=18)
plt.xlabel('Head', fontsize=16)
plt.ylabel('Layer', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# # Plot top and bottom heads
# top_heads = score_df.nsmallest(10, 'score')
# plt.figure(figsize=(10, 6))
# sns.barplot(data=top_heads, x='score', y=top_heads.apply(lambda x: f"L{x['layer']}H{x['head']}", axis=1))
# plt.title('Top 10 Heads by Explained Attention Score (lower is better)')
# plt.xlabel('Score')
# plt.ylabel('Layer.Head')
# plt.tight_layout()
# plt.show()


# %%
from tqdm import tqdm
from functools import partial
import seaborn as sns
import pandas as pd

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
        # 'blocks.2.hook_mlp_out',
        'None'
    ]
    results = dict()
    full_res = dict()
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
        full_res[component] = res.numpy(force=True)
        del cache

    model.reset_hooks()
    return results, full_res

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

_, res = ablate_metric(
    model,
    batch,
    partial(explained_attn_score_metric, pred_attn=attn_batch)
)
# %%

# Unpack means and stds for each component
# components = list(res.keys())
# means = np.array([res[c][0] for c in components])
# stds = np.array([res[c][1] for c in components])
data = pd.DataFrame(res)
# This reshapes your data for easier plotting
melted_data = data.melt(var_name='component', value_name='value')
melted_data = melted_data.dropna()
melted_data = melted_data[melted_data['component'] != 'None']

plt.figure(figsize=(12,5))
sns.set_theme(style="white")
# sns.boxplot(data=melted_data, y='component', x='value', hue='component',)
sns.boxplot(data=melted_data, y='component', x='value', hue='component', palette='icefire')

# Add a vertical line for the 'None' baseline
if 'None' in data.columns:
    baseline_value = data['None'].mean()
    # Get current palette's first color for consistency
    current_palette = sns.color_palette('Set2')
    baseline_color = current_palette[-2]  # Use first color from the palette
    plt.axvline(x=baseline_value, color=baseline_color, linestyle='-.', linewidth=2,
                label=f"No Ablation")
    plt.legend(fontsize=14)

plt.title('Influence of components on L1H5', fontsize=18, fontweight='bold')
plt.xlabel('Semantic Category Score', fontsize=16)
plt.ylabel('Ablated Component', fontsize=16)
plt.grid(axis='x', alpha=0.3)  # Changed from axis='y' to axis='x'
plt.yticks(np.arange(0,5), [
    'Embed',
    'Pos Embed',
    'Attn L0',
    'MLP L0',
    'MLP Resid L0',
], fontsize=14)
plt.xticks(fontsize=14)
# Add annotation arrow indicating importance
plt.annotate('more important', xy=(1.25, -0.2), xytext=(0.8, -0.2), 
             arrowprops=dict(facecolor='black', shrink=0.01, width=3, headwidth=7),
             ha='center', va='center', fontsize=14,
            rotation=0)
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
