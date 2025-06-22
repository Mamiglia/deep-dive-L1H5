# %%
import torch, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain
from transformer_lens import HookedTransformer
from src.ablation import get_vocab_attn1_input
from src.utils import GROUP_TOKENS, clean_mem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
attn_in = get_vocab_attn1_input(model).to(device)
clean_mem()

group_toks = list(chain(*model.tokenizer(GROUP_TOKENS).input_ids))

W_Q = model.blocks[1].attn.W_Q[5].detach()
W_K = model.blocks[1].attn.W_K[5].detach()
W_QK = W_Q@W_K.T
W_sym = (W_QK + W_QK.T)/2
W_skew= (W_QK - W_QK.T)/2

E = attn_in[group_toks]
attn_base= E@W_QK@E.T/np.sqrt(64)
attn_sym = E@W_sym@E.T/np.sqrt(64)
attn_skew= E@W_skew@E.T/np.sqrt(64)

fig,axes=plt.subplots(1,3,figsize=(18,5))
for ax,data,title in zip(axes,[attn_base,attn_sym,attn_skew],
                         ["Base","Symmetric","Skew"]):
    sns.heatmap(data.cpu(),cmap='icefire',center=0,ax=ax,xticklabels=False,yticklabels=False,
                vmin=attn_base.min().item(),vmax=attn_base.max().item())
    ax.set_title(f"{title} Attention")
plt.tight_layout(); plt.show()

# %%
