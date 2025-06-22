# %%
from typing import Callable, Tuple
from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns

from functools import partial
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer, ActivationCache

from src.utils import load_model_sae
from src.prompt import build_prompt, tokenize_prompt
from src.metrics import explained_attn_score
from src.ablation import ablate_component_mean

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

LAYER = 1
HEAD_IDX= 5
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"
model, _, _, _ = load_model_sae(sae_id=HOOK_POINT)

def explained_attn_score_metric(
    cache: ActivationCache,
    pred_attn
):
    attn = cache["blocks.1.attn.hook_pattern"]
    score = explained_attn_score(attn, pred_attn)
    return score[...,HEAD_IDX]

@torch.inference_mode()
def ablate_metric(
    model: HookedTransformer,
    batch,
    metric,
) -> Tuple[dict, dict]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.
    """
    components = [
        'hook_embed',
        'hook_pos_embed',
        'blocks.0.hook_attn_out',
        'blocks.0.hook_mlp_out',
        'blocks.0.hook_mlp_resid',
        'None'
    ]
    results = dict()
    full_res = dict()
    out_hook = "blocks.1.attn.hook_pattern"
    
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
                hooks.append((component, ablate_component_mean))
            
        model.run_with_hooks(batch, fwd_hooks=hooks)
        
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
    toks = tokenize_prompt(model, prompt)
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
data = pd.DataFrame(res)
melted_data = data.melt(var_name='component', value_name='value').dropna()
melted_data = melted_data[melted_data['component'] != 'None']

plt.figure(figsize=(12,5))
sns.set_theme(style="white")
sns.boxplot(data=melted_data, y='component', x='value', hue='component', palette='icefire')

if 'None' in data.columns:
    baseline_value = data['None'].mean()
    current_palette = sns.color_palette('Set2')
    baseline_color = current_palette[-2]
    plt.axvline(x=baseline_value, color=baseline_color, linestyle='-.', linewidth=2, label=f"No Ablation")
    plt.legend(fontsize=14)

plt.title('Influence of components on L1H5', fontsize=18, fontweight='bold')
plt.xlabel('Semantic Category Score', fontsize=16)
plt.ylabel('Ablated Component', fontsize=16)
plt.grid(axis='x', alpha=0.3)
plt.yticks(np.arange(0,5), ['Embed', 'Pos Embed', 'Attn L0', 'MLP L0', 'MLP Resid L0'], fontsize=14)
plt.xticks(fontsize=14)
plt.annotate('more important', xy=(1.25, -0.2), xytext=(0.8, -0.2), arrowprops=dict(facecolor='black', shrink=0.01, width=3, headwidth=7), ha='center', va='center', fontsize=14, rotation=0)
plt.tight_layout()
plt.show()