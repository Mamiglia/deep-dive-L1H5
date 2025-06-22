#%%
import torch, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer
from src.ablation import get_vocab_attn1_input
from src.utils import clean_mem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small", device=device).to(device)
attn_in = get_vocab_attn1_input(model).to(device)
clean_mem()

# build IDX from token_frequencies.csv
token_freq = pd.read_csv("out/token_frequencies.csv")
IDX = token_freq.id[:8192]

W_Q=model.blocks[1].attn.W_Q[5]; W_K=model.blocks[1].attn.W_K[5]

# %%
np.random.seed(1)  # For reproducibility

E = attn_in.detach().clone()[IDX]#[np.random.choice(attn_in.shape[0], 5000, replace=False)]  # Use a subset for visualization
W_QK = W_Q @ W_K.T

# Assuming E, W_QK are defined and on the correct device
attn_base = (E @ W_QK @ E.T).cpu() #/ np.sqrt(64)
# get logprobs
attn_base = attn_base.log_softmax(dim=-1)  # Convert to log probabilities
# Normalize attention values to have unit diagonal
attn_base = attn_base - attn_base.diagonal() # Subtract diagonal to center around zero
# attn_base /= attn_base.diagonal()#.reshape(-1, 1)  # Normalize each row by its diagonal value
# attn_base = attn_base.clamp(-10, 10)  # Clip values for better visualization

E = torch.nn.functional.normalize(E)  # Normalize E to unit length
cos_sim = (E @ E.T ).cpu()  # Cosine similarity matrix
cos_sim.fill_diagonal_(1)  # Set diagonal to 1 to avoid machine precision issues

# Calculate statistics based on cosine similarity windows using torch
window_size = 0.03
cos_min, cos_max = cos_sim.min(), cos_sim.max()
bins = torch.linspace(cos_min, 1, 50)  # Create overlapping bins with step 0.01

# Use tensors directly
print(f'Processing {cos_sim.numel()} points')

# Initialize lists to store results
cos_sim_centers = []
mean_attn_values = []
std_attn_values = []
window_counts = []
percentile_25_values = []  # For 25th percentile
percentile_75_values = []  # For 75th percentile

# For each bin center, calculate statistics for all points in [center, center+window_size]
for center in tqdm(bins):
    mask = (cos_sim >= center) & (cos_sim <= center + window_size)  # Create a mask for the current window
    
    # Get values in this window
    window_values = attn_base[mask]

    cos_sim_centers.append(center.item() + window_size / 2)  # Use middle of window as x-coordinate
    mean_attn_values.append(window_values.mean().item())
    std_attn_values.append(window_values.std().item())
    window_counts.append(mask.sum().item())

    # Calculate percentiles using kthvalue
    if len(window_values) > 0:
        k_25 = max(1, int(0.25 * len(window_values)))
        k_75 = max(1, int(0.75 * len(window_values)))
        percentile_25_values.append(torch.kthvalue(window_values, k_25).values.item())
        percentile_75_values.append(torch.kthvalue(window_values, k_75).values.item())
    else:
        percentile_25_values.append(0.0)  # Default value for empty windows
        percentile_75_values.append(0.0)

# Create smoothed dataframe
df_smoothed = pd.DataFrame({
    'cos_sim': cos_sim_centers,
    'mean_attn': mean_attn_values,
    'std_attn': std_attn_values,
    'count': window_counts,
    'percentile_25': percentile_25_values,
    'percentile_75': percentile_75_values
})

# Filter out windows with too few points for better visualization
# df_smoothed = df_smoothed[df_smoothed['count'] > 50]

print(f'Smoothed {len(df_smoothed)} points with window size {window_size}')

# Calculate confidence interval (95% CI ≈ 1.96 * std / sqrt(n))
df_smoothed['ci_upper'] = df_smoothed['mean_attn'] + 2.576 * df_smoothed['std_attn'] / (df_smoothed['count'] ** 0.5)
df_smoothed['ci_lower'] = df_smoothed['mean_attn'] - 2.576 * df_smoothed['std_attn'] / (df_smoothed['count'] ** 0.5)

# %%
# Plot
plt.figure(figsize=(12, 7))

# Define a better color palette
palette = sns.color_palette("viridis", 5)  # Using viridis as the main palette
main_color = palette[3]
fill_color = palette[4]
baseline_color = palette[0]
scatter_color = palette[2]

print(f'Plotting {len(df_smoothed)} smoothed points')
# Create a barplot with error bars
ax = plt.gca()
bars = ax.bar(
    df_smoothed['cos_sim'],
    df_smoothed['mean_attn'] / np.log(2),
    width=(bins[1] - bins[0])*.9,  # Adjust width as needed
    color=main_color,
    alpha=0.8,
    label='Avg. logprob difference',
    yerr=[
        abs((df_smoothed['mean_attn'] - df_smoothed['ci_lower']) / np.log(2)),
        abs((df_smoothed['ci_upper'] - df_smoothed['mean_attn']) / np.log(2))
    ],
    capsize=3,
    error_kw={'ecolor': palette[2], 'elinewidth': 2, 'label': '99% Confidence Interval'}
)


# Add horizontal baseline at y=0
# plt.axhline(y=0, color=baseline_color, linestyle='--', linewidth=1, label='Current token')

# Sample tokens based on similarity to 'red'
# reference_token = ' red'

# reference_idx = np.where(VOCAB == reference_token)[0][0]  # Get index of 'red' token

# # Calculate cosine similarity with reference token
# token_similarities = cos_sim[reference_idx]  # Cosine similarity of all tokens to 'red'

# # Create bins of size 0.05
# bin_size = 0.05
# bin_starts = torch.arange(0.3, 1.0, bin_size)

# # Add specific tokens to always display
# special_tokens = [' blue', ' green', 'red', 'RED', ' Red', ' RED', 'Red']
# # special_tokens = [' north', ' south', ' east', ' west', ' East']
# special_indices = []
# for token in special_tokens:
#     try:
#         idx = np.where(VOCAB == token)[0][0]
#         special_indices.append(idx)
#         sim = token_similarities[idx].item()
#         attn = attn_base[reference_idx, idx].item() / np.log(2)
#         plt.scatter(sim, attn, color=scatter_color, s=70, zorder=6, 
#                    label='Token similarity and logprob difference to "_red"' if token == special_tokens[0] else None)
#         plt.annotate(token.replace(' ', '_'),
#                     xy=(sim, attn),
#                     xytext=(5, 5),
#                     textcoords='offset points',
#                     fontsize=12,
#                     color='white',
#                     bbox=dict(boxstyle="round,pad=0.3", fc=palette[2], ec=palette[0], alpha=0.8))
#         print(f'Added special token: {token} (idx: {idx}), similarity: {sim:.3f}, attention: {attn:.3f}')
#     except:
#         print(f"Could not find token: {token}")

# torch.manual_seed(0)  # For reproducibility
# # Plot samples for each bin
# for bin_start in bin_starts:
#     print(f'Processing bin starting at {bin_start:.2f}')
#     bin_end = bin_start + bin_size
#     # Find tokens with similarity in this bin
#     mask = (token_similarities >= bin_start) & (token_similarities <= bin_end)
#     # Get indices of tokens in this bin
#     bin_indices = torch.where(mask)[0]
#     # Sample up to 3 tokens from this bin
#     sample_size = min(3, len(bin_indices))
#     samples = bin_indices[torch.randperm(len(bin_indices))[:sample_size]]
    
#     for sample_idx in samples:
#         # Skip if this is one of our special tokens
#         if sample_idx in special_indices:
#             continue
#         # Get token text and attention value
#         token_text = VOCAB.iloc[sample_idx.item()]
#         print(f'Processing token: {token_text} (idx: {sample_idx})')
#         similarity = token_similarities[sample_idx].item()
#         attention_value = attn_base[reference_idx, sample_idx].item() / np.log(2)
        
#         # Plot the point
#         plt.scatter(similarity, attention_value, color=scatter_color, s=50, zorder=5,)
        
#         # Add token text as annotation
#         plt.annotate(token_text.replace(' ', '_'),
#                         xy=(similarity, attention_value),
#                         xytext=(5, 5),
#                         textcoords='offset points',
#                         fontsize=12,
#                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=palette[1], alpha=0.8))

# Labels and title
plt.xlabel('Cosine Similarity', fontsize=14)
plt.ylabel('Logprob difference to subject token', fontsize=14)
plt.title('Attention scores vs Cosine Similarity', fontsize=16, fontweight='bold')
plt.legend(frameon=True, facecolor='white', edgecolor=palette[1])
plt.ylim(-18, 5)
plt.xlim(left=0.5, right=1)
plt.grid(True, alpha=0.3)
# plt.xscale('log')
# Add arrow in bottom right corner
plt.annotate('more similar', xy=(0.63, -17.3), xytext=(0.55, -17.3),
             arrowprops=dict(shrink=0.05, headwidth=6, width=1),
             ha='center', va='center', fontsize=14,)

# Add vertical arrow showing "more attention"
plt.annotate('more attention', xy=(0.99, -8), xytext=(0.99, -14.5),
             arrowprops=dict(shrink=0.05, headwidth=6, width=1),
             ha='center', va='center', fontsize=14, rotation=90)

plt.tight_layout()
plt.show()

# %%
