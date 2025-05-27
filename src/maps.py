import torch
import numpy as np
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint

from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)

from sae_lens import HookedSAETransformer, SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display

from torch import Tensor
import torch.nn.functional as F


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

d_sae = gpt2_sae.W_dec.shape[0]

@torch.inference_mode()
def sae_decode(sae: HookedSAETransformer, feature_idx : int, feature_norm=1, simple=True) -> Tensor:
    if simple:
        return sae.W_dec[feature_idx].unsqueeze(0) * feature_norm
    
    feats = torch.zeros((1, d_sae), device=device)
    feats[:,feature_idx] = feature_norm
    return sae.decode(feats)

feats2resid = sae_decode


@torch.inference_mode()
def feat_attn_scores_q(
    sae: HookedSAETransformer,
    W_KQ: FactoredMatrix, # d d
    src_feature: int,
    layer_norm = False,
    sae_encoder = False,
) -> Tensor:
    src = sae_decode(sae, src_feature, simple=True) # 1 d
    if layer_norm:
        src = F.layer_norm(src, (sae.cfg.d_in, ))
    query = W_KQ @ src.T
    
    if sae_encoder:
        # pass the query through the SAE encoder 
        # rather than forcing to use W_D
        return sae.encode(query)
    
    W_dec = sae.W_dec # d_sae d
    
    if layer_norm:
        W_dec = F.layer_norm(W_dec, (sae.cfg.d_in,))
    
    return W_dec @ query


@torch.inference_mode()
def feat_attn_scores_k(
    sae: HookedSAETransformer,
    W_KQ: FactoredMatrix, # d d
    dest_feature: int,
    layer_norm = False,
) -> Tensor:
    dest = sae_decode(sae, dest_feature, simple=True) # 1 d
    if layer_norm:
        dest = F.layer_norm(dest, (sae.cfg.d_in, ))
    key = dest @ W_KQ
    
    W_dec = sae.W_dec # d_sae d
    
    if layer_norm:
        W_dec = F.layer_norm(W_dec, (sae.cfg.d_in,))
    
    return key @ W_dec.T

    