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

from jaxtyping import Float, Int

%load_ext autoreload
%autoreload 2

from src.maps import *
from src.utils import *

LAYER = 5
HEAD_IDX= 7
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
text = "Mary gave John a book because she was leaving."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

str_tokens = model.to_str_tokens(text)
for layer in [LAYER]:
    attention_pattern = cache["pattern", layer]
    display(cv.attention.attention_patterns(tokens=str_tokens, attention=attention_pattern))


# %%
resid = cache[HOOK_POINT]

resid = F.layer_norm(resid, (d_model, ))


feats = sae.encode(resid)

# "she" features:
print(torch.argsort(feats[-4], descending=True)[:8])
print(feats[-4][torch.argsort(feats[-4], descending=True)[:8]])

def topk_feats(
    feats: Float[Tensor, "... d_sae"],
    k: int = 8,
):
    values, idx = torch.topk(feats, k, dim=-1)
    # Sort within the top-k
    sorted_values, sorted_indices = torch.sort(values, descending=True, dim=-1)
    sorted_idx = torch.gather(idx, -1, sorted_indices)
    return sorted_idx, sorted_values
    

# %%

@torch.inference_mode()
def token2feat_attn(
    resid : Float[Tensor, "seq d_model"],
    sae : HookedSAETransformer,
    W_KQ: Float[Tensor, "d_model d_model"],
    layer_norm = True,
):
    seq, d_model = resid.shape
    resid = F.layer_norm(resid, (d_model,))
    
    query = W_KQ @ resid.T
    
    W_dec = sae.W_dec # d_sae d
    
    if layer_norm:
        W_dec = F.layer_norm(W_dec, (sae.cfg.d_in,))
    
    return W_dec @ query

resid = cache[HOOK_POINT]
feat_attn = token2feat_attn(resid, sae, W_KQ).AB.T

dest_feat_idx, _ = topk_feats(feat_attn,12)

# %%
for i in range(8,12):
    display_dashboard(latent_idx=dest_feat_idx[3][i], sae_id = HOOK_POINT)

# %%
resid = cache[HOOK_POINT]
resid = F.layer_norm(resid, (d_model, ))
src_feats = sae.encode(resid)
src_feat_idx, _ = topk_feats(src_feats,12)
for i in range(8,12):
    display_dashboard(latent_idx=src_feat_idx[3][i], sae_id = HOOK_POINT)
# %%
resid = cache[HOOK_POINT]
resid = F.layer_norm(resid, (d_model, ))
feats = sae.encode(resid)
for t in range(11):
    token = str_tokens[t]
    src_token = resid[t]
    
    src_feat_idx, _ = topk_feats(feats[t],32)
    
    dest_t = torch.argmax(cache["pattern", layer][HEAD_IDX][t])
    print(f"{token} -> {str_tokens[dest_t]}")
    
    dest_token = resid[dest_t]
    dest_feat_idx, _ = topk_feats(feats[dest_t], 32)
    
    pred_dest_feat = token2feat_attn(resid, sae, W_KQ).AB.T[t]
    pred_dest_feat_idx, _ = topk_feats(pred_dest_feat, 32)
    
    # union size: number of indices both in pred_dest_feat_idx and dest_feat_idx
    union = set(dest_feat_idx.tolist()) & set(pred_dest_feat_idx.tolist())
    
    print(len(union), union)
    
    
# %%
