from typing import Tuple
import random
import torch
from jaxtyping import Int
from transformer_lens import HookedTransformer
from itertools import chain
from src.constants import KEYWORDS

def tokenize_prompt(model: HookedTransformer, prompt: list[str]) -> torch.Tensor:
    toks = model.tokenizer(prompt).input_ids
    flat = list(chain(*toks))
    if not flat or flat[0] != model.tokenizer.bos_token_id:
        flat.insert(0, model.tokenizer.bos_token_id)
    return torch.tensor(flat)

def build_prompt(seq_len: int=64) -> Tuple[list[str], torch.Tensor]:
    if seq_len == 0 or not KEYWORDS:
        return [], torch.tensor([])
    cats = list(KEYWORDS.keys())
    prompt, assigns = [], []
    while len(prompt) < seq_len:
        c = random.choice(cats)
        t = random.choice(KEYWORDS[c])
        prompt.append(' '+t); assigns.append(c)
    prompt = prompt[:seq_len]; assigns=assigns[:seq_len]
    # build ground‐truth attn matrix
    N=seq_len+1
    pat = torch.zeros(N,N, dtype=torch.int)
    for i in range(1,N):
        for j in range(1,N):
            if assigns[i-1]==assigns[j-1] and prompt[i-1]!=prompt[j-1]:
                pat[i,j]=1
    # mask upper and fallback to BOS
    pat = pat.masked_fill(torch.triu(pat>0,0),0)
    pat[:,0][pat.sum(dim=-1)==0]=1
    return prompt, pat

def random_toks_with_keywords(
    model: HookedTransformer, keywords: str|list[str], seq_len=20
) -> tuple[torch.Tensor,list[int]]:
    from src.utils import random_tokens
    kw = keywords if isinstance(keywords,list) else KEYWORDS[keywords]
    kw_ids = list(chain(*model.tokenizer([f" {w}" for w in kw]).input_ids))
    idxs = sorted(random.sample(range(1,seq_len), len(kw_ids)))
    idxs[-1]=seq_len-1
    toks = random_tokens(model, seq_len)
    for w,i in zip(kw_ids,idxs):
        toks[0,i]=w
    return toks, idxs
