from typing import Tuple
import torch
from torch import Tensor
from jaxtyping import Float, Int
import numpy as np
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint
import random
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformers import PreTrainedTokenizerBase
from rich.table import Table
from rich.console import Console


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

from sae_lens import HookedSAETransformer, SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from IPython.display import HTML, IFrame, clear_output, display

import torch.nn.functional as F

def display_dashboard(
    sae_release="gpt2-small-resid-post-v5-32k",
    sae_id="blocks.10.hook_resid_post",
    latent_idx=0,
    width=600,
    height=400,
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


def topk_feats(
    feats: Float[Tensor, "... d_sae"],
    k: int = 8,
) -> Tuple[Int[Tensor, "... k"], Float[Tensor, "... k"]]:
    """
    Selects the top-k features along the last dimension of the input tensor.
    Args:
        feats (Float[Tensor, "... d_sae"]): Input tensor containing feature values. The last dimension represents the features.
        k (int, optional): Number of top features to select. Defaults to 8.
    Returns:
        Tuple[Int[Tensor, "... k"], Float[Tensor, "... k"]]: 
            A tuple containing:
                - Indices of the top-k features for each entry (same shape as input except last dimension is k).
                - Values of the top-k features for each entry (same shape as input except last dimension is k).
    Notes:
        The returned indices and values are sorted in descending order of feature values within the top-k selection.
    """
    values, idx = torch.topk(feats, k, dim=-1)
    # Sort within the top-k
    sorted_values, sorted_indices = torch.sort(values, descending=True, dim=-1)
    sorted_idx = torch.gather(idx, -1, sorted_indices)
    return sorted_idx, sorted_values

from itertools import chain

def random_toks_with_keywords(model, keywords : str | list[str], seq_len=20):
    """
    Generate a random sequence of tokens, inserting specific keywords at random positions.
    Returns:
        text (str): The generated text.
        kw_idx (list): List of positions where keywords were inserted.
    """
    if isinstance(keywords, str):
        keywords = KEYWORDS[keywords]
    random.shuffle(keywords)
    kw_toks = model.tokenizer([" " + kw for kw in keywords]).input_ids
    kw_toks = list(chain(*kw_toks))
    
    assert len(kw_toks) == len(keywords), "Keywords are being split into multiple tokens."
    
    kw_idx = sorted(random.sample(range(1,seq_len), len(keywords)))
    kw_idx[-1] = seq_len-1
    
    random_toks = random_tokens(model, seq_len)
    for kw, idx in zip(kw_toks, kw_idx):
        random_toks[0,idx] = kw
        
    return random_toks, kw_idx
        
def random_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size seq_len"]:
    """
    Generates a sequence of random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    prefix = (torch.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rnd_tok = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len-1), dtype=torch.int64)
    return torch.cat([prefix, rnd_tok], dim=1)

def precision(a, b, pos_idx):
    """
    Compute precision given predicted and ground truth feature indices and keyword positions.
    """
    assert a.shape == b.shape
    seq_len = a.shape[0]
    neg_idx = [i for i in range(seq_len) if i not in pos_idx]
    
    intersection = torch.zeros(seq_len)
    
    for i in range(seq_len):
        pred_feat = set(a[i].tolist())
        gt_feat = set(b[i].tolist())
        intersection[i] += len(pred_feat & gt_feat)
        
    tp = sum(intersection[pos_idx])
    fp = sum(intersection[neg_idx])
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)
    

KEYWORDS = {
    "colors": ["blue", "red", "green", "yellow", "purple"],
    "animals": ["dog", "cat", "mouse", "horse", "sheep"],
    "fruits": ["apple", "banana", "grape", "peach", "lemon"],
    "emotions": ["happy", "sad", "angry", "scared", "proud"],
    "weather": ["rain", "snow", "wind", "storm", "sun"],
}



def get_most_attended_tokens(token:str, attn:Tensor, tokenizer: PreTrainedTokenizerBase, k=16):
    tok_idx = tokenizer(token).input_ids
    
    assert len(tok_idx) == 1, f'Can only handle one token at a time. Currently {tok_idx}'
    tok_idx = tok_idx[0]
    
    scores = attn[tok_idx]
    max_val, max_idx = torch.topk(scores, k, largest=True, sorted=True)
    return tokenizer.batch_decode(max_idx), max_val
    
def display_most_attended_tokens(tokens, attn, tokenizer:PreTrainedTokenizerBase, k=10):
    table = Table(title="Most Attended Tokens")
    table.add_column("Input Token", style="bold")
    table.add_column("Top Attended Tokens", style="dim")
    for token in tokens:
        top_tokens, scores = get_most_attended_tokens(token, attn, k=k, tokenizer=tokenizer)
        table.add_row(
            repr(token),
            ", ".join(top_tokens),
        )
    console = Console()
    console.print(table)