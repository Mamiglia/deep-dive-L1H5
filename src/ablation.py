from typing import Callable, Tuple
from transformer_lens import HookedTransformer
from torch import Tensor
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import torch

def ablate_component_mean(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
):
    """Mean-ablate a component"""
    return t.mean(dim=(0,1))

def stop_computation(t, hook):
    raise StopIteration(f"Stopping model mid-execution at {hook.name}")

def ablate_shorten_sequence(
    t : Float[Tensor, "batch pos d_model"],
    hook: HookPoint,
    max_length: int = 1024
):
    print(hook.name)
    return t[:,:max_length,:]

@torch.no_grad()
def get_vocab_attn1_input(
    model: HookedTransformer,
) -> Tensor:
    """
    Computes the layer-normed residual stream vectors for each token in the vocabulary,
    as input to a specific layer's attention mechanism. This is done by running
    the model on all tokens and stopping it right before the target layer's
    attention block.
    """
    model.reset_hooks()
    vocab = torch.arange(0, model.tokenizer.vocab_size, device=model.W_E.device)
    ablate_components = [
        'hook_pos_embed',  # Position embeddings
        'blocks.0.hook_attn_out', # Attention-0 output
    ]
    # This dict will store the cached activations
    activation_cache = {}
    
    out_hook = 'blocks.1.ln1.hook_normalized' # input to the Attention-1 (after LayerNorm)

    # Hook function to cache activations
    def cache_hook(activation, hook):
        activation_cache[hook.name] = activation

    hooks = [
        ('blocks.0.ln1.hook_normalized', ablate_shorten_sequence),  # shorten input of attn-0
        (out_hook, cache_hook), 
        (out_hook, stop_computation),  
    ]
    for component in ablate_components:
        hooks.append((component, ablate_component_mean))
        
    print(hooks)
    try:
        model.run_with_hooks(vocab, fwd_hooks=hooks)
    except StopIteration as e:
        print(e)

    model.reset_hooks()
    return activation_cache[out_hook][0]