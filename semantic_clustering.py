# %%
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
model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

#%%
VOCAB_SIZE = model.tokenizer.vocab_size


def ablate_component(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
):
    """Mean-ablate a component"""
    return t.mean(dim=(0,1))

def ablate_reduce_component(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
):
    print(hook.name)
    return t[:,:1024,:]

def stop_computation(t, hook):
    raise StopIteration(f"Stopping model mid-execution at {hook.name}")



@torch.inference_mode()
def vocab_attn(
    model: HookedTransformer,
) -> Float[Tensor, "layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    """
    model.reset_hooks()
    vocab = torch.arange(0, VOCAB_SIZE)
    ablate_components = [
        'hook_pos_embed',
        'blocks.0.hook_attn_out',
    ]
    # This dict will store the cached activations
    activation_cache = {}

    # Hook function to cache activations
    def cache_hook(activation, hook):
        activation_cache[hook.name] = activation
        
    model.cfg.use_attn_in = False
        
    hooks = [
        # ("blocks.1.attn.hook_q", cache_hook),
        # ("blocks.1.attn.hook_k", cache_hook),
        ('blocks.0.ln1.hook_normalized', ablate_reduce_component),
        ('blocks.1.ln1.hook_normalized', cache_hook),  
        ('blocks.1.ln1.hook_normalized', stop_computation),  
    ]
    for component in ablate_components:
        hooks.append((component, ablate_component))
        
    print(hooks)
    try:
        model.run_with_hooks(vocab, fwd_hooks=hooks)
    except StopIteration as e:
        print(e)

    model.reset_hooks()
    return activation_cache['blocks.1.ln1.hook_normalized'][0,:]


attn_in = vocab_attn(model)

# %%
W_Q = model.blocks[1].attn.W_Q[5].clone().detach()
W_K = model.blocks[1].attn.W_K[5].clone().detach()



q = attn_in @ W_Q
k = attn_in @ W_K

# %%
attn = FactoredMatrix(k,q.T).AB

attn.shape
# %%
import seaborn as sns
L=32
sns.heatmap(attn[:L, :L].numpy(force=True),
    xticklabels=model.tokenizer.batch_decode(torch.arange(L)),
    yticklabels = model.tokenizer.batch_decode(torch.arange(L))
)
# %%

def get_most_attended_tokens(token:str, attn:Tensor, k=16, tokenizer=model.tokenizer):
    tok_idx = tokenizer(token).input_ids
    
    assert len(tok_idx) == 1, f'Can only handle one token at a time. Currently {tok_idx}'
    tok_idx = tok_idx[0]
    
    scores = attn[tok_idx]
    max_val, max_idx = torch.topk(scores, k, largest=True, sorted=True)
    return tokenizer.batch_decode(max_idx), max_val
    

get_most_attended_tokens(' red', attn)
# %%
get_most_attended_tokens(' 69', attn)

# %%
get_most_attended_tokens('Monday', attn)
# %%
get_most_attended_tokens(' John', attn)
# %%
get_most_attended_tokens(' +', attn)
# %%
get_most_attended_tokens(' if', attn)

# %%
get_most_attended_tokens(' Italy', attn)

# %%
