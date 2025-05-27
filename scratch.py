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
text = "The cat saw a dog. The man looks at the door. the mouse noticed the elephant"
vocab = list(model.tokenizer.get_vocab().keys())
random_tokens = [random.choice(vocab) for _ in range(20)]

# Replace GPT2's whitespace special char 'Ġ' with a space for readability
def decode_gpt2_token(token):
    return token.replace("Ġ", " ")

text = " ".join([decode_gpt2_token(tok) for tok in random_tokens])
print("Random text:", text)

# Insert "red", "green", "blue" at random positions in the random_tokens list
color_words = ["red", "green", "blue"]
color_words = ["cat", "dog", "mouse"]
for color in color_words:
    idx = random.randint(0, len(random_tokens))
    random_tokens.insert(idx, color)
text = " ".join([decode_gpt2_token(tok) for tok in random_tokens])
# remove double space:
text = text.replace("  ", " ")
print("Random text with colors:", text)

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

dest_feat_idx, _ = topk_feats(feat_attn.AB.T,12)

# %%
for i in range(8):
    display_dashboard(latent_idx=dest_feat_idx[-2][i], sae_id = HOOK_POINT)

# %%
resid = cache[HOOK_POINT]
resid = F.layer_norm(resid, (d_model, ))
src_feats = sae.encode(resid)
src_feat_idx, _ = topk_feats(src_feats,12)
for i in range(8):
    display_dashboard(latent_idx=src_feat_idx[-4][i], sae_id = HOOK_POINT)
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
    
    pred_dest_feat = token2feat_attn(resid, sae, W_KQ, sae_encoder=False).AB.T[t]
    pred_dest_feat_idx, _ = topk_feats(pred_dest_feat, 32)
    
    # union size: number of indices both in pred_dest_feat_idx and dest_feat_idx
    union = set(dest_feat_idx.tolist()) & set(pred_dest_feat_idx.tolist())
    
    print(len(union), union)
    
# %% [markdown]

# Tested whether tokens attend to expected features. 
# Confirmed that animal tokens attend positively at feature #9270 
# Colours respond positively at features #21000, #22330, #21055

# %%
