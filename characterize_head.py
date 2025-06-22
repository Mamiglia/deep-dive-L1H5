# %%
from typing import Callable, Tuple
from matplotlib import pyplot as plt
import torch
import numpy as np
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

from sae_lens import HookedSAETransformer, SAE
from IPython.display import HTML, IFrame, clear_output, display

import torch.nn.functional as F
import random
import seaborn as sns
from itertools import chain

from jaxtyping import Float, Int
import circuitsvis as cv

# %load_ext autoreload
# %autoreload 2

from src.maps import *
from src.utils import load_model_sae
from src.prompt import build_prompt, tokenize_prompt
from src.metrics import explained_attn_score
from semantic_head import get_explained_attention_scores

LAYER = 1
HEAD_IDX= 5
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"

# load model & SAE once
model, _, _, _ = load_model_sae(sae_id=HOOK_POINT)

# %%
# Generate a test prompt and ground truth attention
test_prompt, test_attn = build_prompt(seq_len=255)
test_tokens = tokenize_prompt(model, test_prompt).to(device)
scores = get_explained_attention_scores(model, test_tokens, test_attn.to(device))

# Convert to dataframe for visualization
import pandas as pd
score_data = [{'layer': layer, 'head': head, 'score': score} for layer, heads in scores.items() for head, score in heads.items()]
score_df = pd.DataFrame(score_data)

# Plot as heatmap
plt.figure(figsize=(10, 8))
pivot_data = score_df.pivot(index='layer', columns='head', values='score')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='rocket_r', vmax=5)
plt.plot([5, 6, 6, 5, 5], [1, 1, 2, 2, 1], linewidth=3, color='green')
plt.annotate('Target Head (L1H5)', xy=(5.5, 1.5), xytext=(6.5, 4), color='white', arrowprops=dict(facecolor='white', shrink=0.1, width=2, headwidth=8, connectionstyle="arc3,rad=-0.2"), fontsize=20, fontweight='bold')
plt.title('Surprisal by Layer and Head', fontsize=18)
plt.xlabel('Head', fontsize=16)
plt.ylabel('Layer', fontsize=16)
plt.tight_layout()
plt.show()