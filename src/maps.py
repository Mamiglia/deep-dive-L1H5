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

from jaxtyping import Float, Int

from torch import Tensor
import torch.nn.functional as F

from src.utils import device


@torch.inference_mode()
def sae_decode(sae: SAE, feature_idx : int, feature_norm=1, simple=True) -> Tensor:
    """
    Decodes a feature from a Sparse Autoencoder (SAE) given its feature index.
    Args:
        sae (SAE): The Sparse Autoencoder model containing the decoder weights.
        feature_idx (int): The index of the feature to decode.
        feature_norm (float, optional): The normalization factor to apply to the feature. Defaults to 1.
        simple (bool, optional): If True, returns the decoded feature using a simple linear transformation.
            If False, constructs a feature vector and decodes it using the SAE's decode method. Defaults to True.
    Returns:
        Tensor: A vector in the semantic space of the residual stream.
    """
    if simple:
        return sae.W_dec[feature_idx].unsqueeze(0) * feature_norm
    
    d_sae = sae.W_dec.shape[0]
    
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
    """Computes the attention scores for a specific source feature in a Sparse Autoencoder (SAE) model.
    Given a source feature index, this function calculates which target features are most attended to by the source feature,
    optionally applying layer normalization and/or encoding the query through the SAE encoder.
    Args:
        sae (HookedSAETransformer): The SAE Transformer model instance.
        W_KQ (FactoredMatrix): The key-query projection matrix of shape (d, d).
        src_feature (int): The index of the source feature for which attention scores are computed.
        layer_norm (bool, optional): If True, applies layer normalization to the source and decoder weights. Defaults to False.
        sae_encoder (bool, optional): If True, passes the query through the SAE encoder instead of using the decoder weights. Defaults to False.
    Returns:
        Tensor: The attention scores for the target features, either as encoded by the SAE encoder or projected via the decoder weights.
    Notes:
        - If `layer_norm` is True, both the source feature vector and decoder weights are layer-normalized.
        - If `sae_encoder` is True, the query is passed through the SAE encoder and the result is returned.
        - Otherwise, the query is projected using the decoder weights to obtain the attention scores.
    """
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
    """Computes the attention scores for a given destination feature using the key projection matrix.
        Args:
            sae (HookedSAETransformer): The transformer model containing the decoder and configuration.
            W_KQ (FactoredMatrix): The key/query projection matrix of shape (d, d).
            dest_feature (int): The index of the destination feature to compute attention scores for.
            layer_norm (bool, optional): Whether to apply layer normalization to the destination and decoder weights. Defaults to False.
        Returns:
            Tensor: The attention scores between the destination feature and all SAE features.
    """
    dest = sae_decode(sae, dest_feature, simple=True) # 1 d
    if layer_norm:
        dest = F.layer_norm(dest, (sae.cfg.d_in, ))
    key = dest @ W_KQ
    
    W_dec = sae.W_dec # d_sae d
    
    if layer_norm:
        W_dec = F.layer_norm(W_dec, (sae.cfg.d_in,))
    
    return key @ W_dec.T

@torch.inference_mode()
def token2feat_attn(
    resid : Float[Tensor, "... d_model"],
    sae : HookedSAETransformer,
    W_KQ: Float[Tensor, "d_model d_model"],
    layer_norm = True,
    sae_encoder = False,
) -> Float[Tensor, "... d_sae"]:
    """Computes attention scores between an input token's residual stream and the features of a Sparse Autoencoder (SAE).
    Args:
        resid (Float[Tensor, "seq d_model"]): The residual stream representation of input tokens, with shape (sequence length, model dimension).
        sae (HookedSAETransformer): The SAE model containing encoder and decoder weights.
        W_KQ (Float[Tensor, "d_model d_model"]): The projection matrix used to compute queries from the residual stream.
        layer_norm (bool, optional): If True, applies layer normalization to the residuals and/or SAE decoder weights. Defaults to True.
        sae_encoder (bool, optional): If True, passes the query through the SAE encoder instead of using the decoder weights. Defaults to False.
    Returns:
        Float[Tensor, "... d_sae"]: The attention scores between SAE features (d_sae).
    Notes:
        - If sae_encoder is True, the function returns the encoded query using the SAE encoder.
        - If sae_encoder is False, the function projects the query using the SAE decoder weights (optionally layer-normalized).
        - Layer normalization is applied to the input and/or decoder weights based on the layer_norm flag.
    """
    d_model = resid.shape[-1]
    resid = F.layer_norm(resid, (d_model,))
    
    query = W_KQ @ resid.T
    
    if sae_encoder:
        # pass the query through the SAE encoder 
        # rather than forcing to use W_D
        return sae.encode(query.AB.T)
    
    
    W_dec = sae.W_dec # d_sae d
    
    if layer_norm:
        W_dec = F.layer_norm(W_dec, (sae.cfg.d_in,))
    
    return (W_dec @ query).AB.T
