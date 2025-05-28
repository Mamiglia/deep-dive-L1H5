# %%
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

%load_ext autoreload
%autoreload 2

from src.maps import *
from src.utils import *

LAYER = 1
HEAD_IDX= 5
HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"

# %%
model, sae, d_sae, d_model = load_model_sae(sae_id=HOOK_POINT)
# %%

display_dashboard(sae_id=HOOK_POINT, latent_idx=1)

# %%
f = sae_decode(sae, feature_idx=0, simple=False)
f.shape

# %%

W_KQ = get_kq(model, layer=LAYER, head_idx=HEAD_IDX)


# %%
import circuitsvis as cv
import random
text = "Mary gave John a book because she was leaving."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

str_tokens = model.to_str_tokens(text)
for layer in [LAYER]:
    print("Layer:", layer)
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))


# %%
text, _ = random_toks_with_keywords(model, "colors")

text

# %%
logits, cache = model.run_with_cache(text, remove_batch_dim=True)

str_tokens = model.to_str_tokens(text)
for layer in [LAYER]:
    print("Layer:", layer)
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))

# Comment:
# out of random text with specific keywords it appears that head 1.5 may be a "semantic head", i.e. a head that attend to tokens that are semantically associated (animals, colors). 
# This is in line with https://www.alignmentforum.org/posts/xmegeW5mqiBsvoaim/we-inspected-every-head-in-gpt-2-small-using-saes-so-you-don which describes 1.5 as "Succession or pairs related behavior. single-token entity (10/10) (men/male/children, human/people/children/girls, he/him/them/his, right, 7, roman numerals, First/Second/Third, third/fourth,  2015-2017, abc)"

# %%
resid = cache[HOOK_POINT]

resid = F.layer_norm(resid, (d_model, ))


feats = sae.encode(resid)

# "she" features:
print(torch.argsort(feats[-2], descending=True)[:8])
print(feats[-4][torch.argsort(feats[-2], descending=True)[:8]])

    

# %%


resid = cache[HOOK_POINT]
feat_attn = token2feat_attn(resid, sae, W_KQ, sae_encoder=False)

dest_feat_idx, _ = topk_feats(feat_attn,12)

# %%
for i in range(1):
    display_dashboard(latent_idx=dest_feat_idx[-1][i], sae_id = HOOK_POINT)

# %%
resid = cache[HOOK_POINT]
resid = F.layer_norm(resid, (d_model, ))
src_feats = sae.encode(resid)
src_feat_idx, _ = topk_feats(src_feats,12)
for i in range(1):
    display_dashboard(latent_idx=src_feat_idx[-1][i], sae_id = HOOK_POINT)
# %%
resid = cache[HOOK_POINT]
resid = F.layer_norm(resid, (d_model, ))
feats = sae.encode(resid)
for t in range(len(str_tokens)):
    token = str_tokens[t]
    src_token = resid[t]
    
    src_feat_idx, _ = topk_feats(feats[t],32)
    
    dest_t = torch.argmax(cache["pattern", layer][HEAD_IDX][t])
    print(f"{token} -> {str_tokens[dest_t]}")
    
    dest_token = resid[dest_t]
    dest_feat_idx, _ = topk_feats(feats[dest_t], 32)
    
    pred_dest_feat = token2feat_attn(resid, sae, W_KQ, sae_encoder=False)[t]
    pred_dest_feat_idx, _ = topk_feats(pred_dest_feat, 32)
    
    # union size: number of indices both in pred_dest_feat_idx and dest_feat_idx
    union = set(dest_feat_idx.tolist()) & set(pred_dest_feat_idx.tolist())
    
    print(len(union), union)
    
# %% [markdown]

# Tested whether tokens attend to expected features. 
# Confirmed that animal tokens attend positively at feature #9270 
# Colours respond positively at features #9975, #22330, #21055

# %% [markdown]
# # Feature imporance
# I want to understand if a feature is considered important for a specific head
# I can do so by checking the norm 
# %%
import matplotlib.pyplot as plt

num_layers = model.cfg.n_layers
num_heads = model.cfg.n_heads

all_norms = []
for layer in range(1,12):
    for head in range(num_heads):
        HOOK_POINT = f"blocks.{layer-1}.hook_resid_post"
        model, sae, d_sae, d_model = load_model_sae(sae_id=HOOK_POINT)
        W_KQ = get_kq(model, layer=layer+1, head_idx=head)
        features = sae.W_dec
        features = features / features.norm(dim=-1, keepdim=True)
        assert torch.allclose(features.norm(dim=-1), torch.ones(features.shape[0], device=device))
        keys = W_KQ @ features.T
        keys = keys.AB
        norms = keys.norm(dim=0).tolist()
        all_norms.append({
            "layer": layer,
            "head": head,
            "norms": norms
        })
        # Optionally plot or print stats per head/layer
        plt.hist(norms, bins=20)
        plt.title(f"Layer {layer+1} Head {head}")
        plt.show()
        # print(f"Layer {layer+1} Head {head}: {torch.sum(keys.norm(dim=0) > 0.6).item()} features with norm > 0.6")
        print(topk_feats(keys.norm(dim=0)))

# %%

def pred_accuracy(model : HookedTransformer, sae: SAE, layer: int, head_idx: int, k=32, kw = None):
    """For each token compute the expected features and the predicted ones and measure the difference."""
    if kw is None:
        kw = random.choice(list(KEYWORDS.keys()))
        print(kw) 
    text, kw_idx = random_toks_with_keywords(model, keywords=kw, seq_len = 20)
        
    _, cache = model.run_with_cache(text, remove_batch_dim=True)
    W_KQ = get_kq(model, layer=layer, head_idx=head_idx)

    seq_len = text.shape[-1]
    
    hook_point = f"blocks.{layer-1}.hook_resid_post"

    
    resid = cache[hook_point]
    resid = F.layer_norm(resid, (d_model, ))
    feats = sae.encode(resid)
    # seq features
    feat_idx, _ = topk_feats(feats, k)
    
    most_attended_token = torch.argmax(cache["pattern", layer][head_idx], dim=-1)
    assert torch.all(most_attended_token <= torch.arange(0,seq_len, device=device)), "Cannot attend to future tokens"
    
    gt_feat_idx = feat_idx[most_attended_token] # seq k

    assert gt_feat_idx.shape == (seq_len, k)


    feat_attn = token2feat_attn(resid, sae, W_KQ)
    pred_feat_idx, _ = topk_feats(feat_attn, k)

    
    return precision(pred_feat_idx, gt_feat_idx, kw_idx[1:])


KEYWORDS = {
    "colors": ["blue", "red", "green","green", " green", "yellow", "purple"],
    "animals": ["dog", "cat", "mouse", "horse", "sheep"],
    "fruits": ["apple", "banana", "grape", "peach", "lemon"],
    "emotions": ["happy", "sad", "angry", "scared", "proud"],
    "weather": ["rain", "snow", "wind", "storm", "sun"],
}

# LAYER = 1
# HEAD_IDX= 5
# HOOK_POINT = f"blocks.{LAYER-1}.hook_resid_post"
# model, sae, d_sae, d_model = load_model_sae(sae_id=HOOK_POINT)
pred_accuracy(model, sae, LAYER, HEAD_IDX)
# %%

num_layers = model.cfg.n_layers
num_heads = model.cfg.n_heads

REPEATS = 64

records = []

for layer in range(1,12):
    hook_point = f"blocks.{layer-1}.hook_resid_post"
    model, sae, d_sae, d_model = load_model_sae(sae_id=hook_point)
    for head in tqdm(range(num_heads)):
        
        for category in KEYWORDS.keys():
            for repeat in range(REPEATS):
                p = pred_accuracy(model, sae, layer, head, kw=category)
                
                records.append({
                    'layer': layer,
                    'head_idx': head,
                    'category': category,
                    'rep': repeat,
                    'precision': p
                })
# %%
from tqdm import tqdm
import pandas as pd
# %%
df = pd.DataFrame.from_records(records)

import seaborn as sns

for category in KEYWORDS.keys():
    p = np.zeros((num_layers, num_heads))
    for l in range(1, num_layers):
        for h in range(num_heads):
            mask = (df.layer == l) & (df.head_idx == h) & (df.category == category)
            p[l, h] = df[mask].precision.mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(p, annot=True)
    plt.title(f"Precision Heatmap for Category: {category}")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.show()
# %%
