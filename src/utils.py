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


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

from sae_lens import HookedSAETransformer, SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display

import torch.nn.functional as F

def display_dashboard(
    sae_release="gpt2-small-resid-post-v5-32k",
    sae_id="blocks.10.hook_resid_post",
    latent_idx=0,
    width=800,
    height=600,
):
    """Displays Neuronpedia for one feature and latent index. 
    """
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))


def load_model_sae(
    model_name="gpt2-small",
    sae_release="gpt2-small-resid-post-v5-32k",
    sae_id="blocks.10.hook_resid_post",
    device=device,
    disable_normalization=True
):
    """Loads a model and its SAE"""
    model = HookedSAETransformer.from_pretrained(model_name, device=device)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=str(device),
    )
    d_sae = sae.W_dec.shape[0]
    d_model = sae.W_dec.shape[1]
    
    if disable_normalization:
        # Disable normalization: necessary for SAEs that 
        #   apply LayerNorm before processing
        sae.run_time_activation_norm_fn_in = lambda x: x
        sae.run_time_activation_norm_fn_out = lambda x: x

    return model, sae, d_sae, d_model

@torch.no_grad()
def get_kq(model: HookedTransformer, layer:int, head_idx:int, reverse=False):
    """Gets W_KQ matrix for a specific attention head in a model"""
    W_K = model.blocks[layer].attn.W_K[head_idx]
    W_Q = model.blocks[layer].attn.W_Q[head_idx]

    # Forse inverso?
    if reverse:
        return FactoredMatrix(W_Q, W_K.T)

    return FactoredMatrix(W_K, W_Q.T)