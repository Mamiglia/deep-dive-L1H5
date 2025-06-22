# %%
from itertools import chain
import torch, seaborn as sns, matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from src.ablation import get_vocab_attn1_input
from src.utils import GROUP_TOKENS, clean_mem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
attn_in = get_vocab_attn1_input(model).to(device)
clean_mem()

group_toks = model.tokenizer(GROUP_TOKENS).input_ids
group_toks = list(chain(*group_toks))

# %%
E = attn_in[group_toks].clone().detach()
W_Q = model.blocks[1].attn.W_Q[5].detach()
W_K = model.blocks[1].attn.W_K[5].detach()
W_QK = W_Q @ W_K.T
W_sym = (W_QK + W_QK.T) / 2
W_skew = (W_QK - W_QK.T) / 2
# Get eigenvalues and eigenvectors of symmetric matrix
S, Q = torch.linalg.eigh(W_sym)

# Sort by absolute value in descending order
abs_S = torch.abs(S)
sorted_indices = torch.argsort(abs_S, descending=True)
S = S[sorted_indices]
Q = Q[:, sorted_indices]

S = S[:64]  # Zero out noise components since W has rank 64
Q = Q[:, :64]  # Keep only the first 64 eigenvectors
# Create a figure with subplots for different scaling values
# Create a figure with subplots for different scaling values
scaling_values = [1.1, 0, -0.5]
titles = [f"Scale negative eigenvalues by {s}" if s != 0 else "Zero out negative eigenvalues" for s in scaling_values]

fig, axes = plt.subplots(1, len(scaling_values), figsize=(20, 6))
if len(scaling_values) == 1:  # Handle case of single subplot
    axes = [axes]

for i, (scale, title) in enumerate(zip(scaling_values, titles)):
    # Clone the eigenvalues to avoid modifying the original
    S_modified = S.clone()
    
    # Apply the scaling to negative eigenvalues
    S_modified[S_modified < 0] *= scale
    
    # Compute attention with modified eigenvalues
    attn_svd = E @ ((Q @ torch.diag(S_modified) @ Q.T) + W_skew) @ E.T
    
    # Plot the heatmap
    sns.heatmap(
        attn_svd.cpu().numpy(force=True), 
        cmap='icefire', 
        center=0,
        ax=axes[i],
        xticklabels=False,
        yticklabels=False,
        # vmax=60
    )
    axes[i].set_title(title)

    
plt.tight_layout()
plt.show()
# %%
