import torch, numpy as np, pandas as pd
from tqdm import tqdm
import seaborn as sns, matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from sae_lens import HookedSAETransformer
from src.ablation import get_vocab_attn1_input, clean_mem
from src.metrics import compute_rank, mrr, accuracy_k, precision_k, spearman

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
attn_in = get_vocab_attn1_input(model).to(device)
clean_mem()

WQ, WK = model.blocks[1].attn.W_Q[5], model.blocks[1].attn.W_K[5]
S, V = torch.linalg.eig(WQ@WK.T)
UQ, UK = V[:,:64], torch.linalg.inv(V)[:64]
S = S[:64]

LIM = attn_in.shape[0]
E = attn_in[:LIM]

# ground truth ranks
attn_ideal = (E@(WQ@WK.T)@E.T).cpu()
gt_rank, gt_sorted = compute_rank(attn_ideal)

records=[]
for k in tqdm([1,2,4,8,16,32]):
  for scale in np.arange(-5,1.3,0.25):
    s=S.clone(); s[:k]*=scale
    attn_mod=(E@UQ@torch.diag(s)@UK@E.T).cpu()
    rank, sorted_ = compute_rank(attn_mod)
    records.append({
      'k':k,'scale':round(scale,2),
      'mean_rank':rank.float().mean().item(),
      'spearman':spearman(sorted_,gt_sorted),
      'mrr':mrr(rank),
      'acc128':accuracy_k(rank,128),
      'prec128':precision_k(sorted_,gt_sorted,128)
    })
    clean_mem()
df=pd.DataFrame(records)
df.to_csv("attention_records.csv",index=False)

# quick heatmaps
vals=['mean_rank','spearman','mrr']
fig,axes=plt.subplots(len(vals),1,figsize=(6,4*len(vals)))
for ax,col in zip(axes,vals):
  sns.heatmap(df.pivot('scale','k',col),ax=ax,cmap='viridis')
  ax.set_title(col)
plt.tight_layout(); plt.show()
