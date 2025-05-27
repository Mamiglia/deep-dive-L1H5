# Feature-Space Attention Analysis

**Research Question**: *Are attention heads meaningful in the feature space?*

This repository explores whether transformer attention mechanisms can be better understood by analyzing them through the lens of learned features from Sparse Autoencoders (SAEs), rather than at the token level.

## 🎯 Core Hypothesis

Traditional attention analysis focuses on *which tokens attend to which tokens*. We propose analyzing *which semantic features attend to which semantic features*. If attention heads are truly meaningful computational units, their behavior should decompose cleanly into interpretable feature-to-feature relationships.

## 🔬 Methodology

### Mathematical Foundation

For any attention head, the attention pattern is fundamentally:
```
A = softmax(Q K^T) = softmax((W_q x)(W_k x)^T) = softmax(x^T W_k^T W_q x)
```

We define the **attention projection matrix** as:
```
M = W_k^T W_q
```

This matrix captures the core "matching" behavior of attention - how the model maps from "what I am" (key) to "what I'm looking for" (query).

### Feature-Space Decomposition

Using Sparse Autoencoders trained on transformer activations, we decompose attention into feature-level interactions:

1. **Extract features**: `f = SAE.encode(x)` 
2. **Project through attention**: `f_projected = SAE.encode(M @ x)`
3. **Analyze feature-feature attention**: Which features in `f` correspond to which features in `f_projected`?

### Key Insights We're Testing

- **Identity-like attention** (`M ≈ I`): Features attend to themselves
- **Semantic clustering** (`M ≈ I_r, r<d`): Features attend to semantically similar features  
- **Cross-semantic attention** (`M ≠ I`): Complex feature transformations drive attention

## 🛠️ Implementation

### Core Functions

**`src/maps.py`**: Feature-space attention mapping
- `feat_attn_scores_q()`: Computes which features a source feature attends to
- `feat_attn_scores_k()`: Computes which features attend to a destination feature
- `sae_decode()`: Converts feature indices back to activation space

**`src/utils.py`**: Model and data utilities
- `load_model_sae()`: Loads GPT-2 models with pre-trained SAEs
- `get_kq()`: Extracts W_K^T W_Q matrix for specific attention heads
- `display_dashboard()`: Interactive feature visualization via Neuronpedia

### Current Analysis Pipeline

```python
# Load model and SAE
model, sae, d_sae, d_model = load_model_sae(sae_id="blocks.4.hook_resid_post")

# Extract attention projection matrix
W_KQ = get_kq(model, layer=5, head_idx=7)

# Run forward pass
logits, cache = model.run_with_cache("Mary gave John a book because she was leaving.")

# Extract residual stream activations
resid = cache["blocks.4.hook_resid_post"]

# Compute feature-level attention projections
feat_attn = token2feat_attn(resid, sae, W_KQ)

# Analyze: which features does "she" attend to?
she_token_idx = -4
predicted_features = topk_feats(feat_attn[she_token_idx], k=12)
```

## 🔍 Current Findings

### Preliminary Results

From `scratch.py` analysis on the sentence *"Mary gave John a book because she was leaving"*:

- **Token-level attention**: Standard attention visualization shows expected patterns
- **Feature-level predictions**: Our method predicts which features the model should attend to
- **Validation**: Comparing predicted attention targets vs. actual attention targets in feature space

### Key Observations

1. **Feature overlap**: Measuring intersection between predicted destination features and actual destination features
2. **Attention head specificity**: Different heads show different patterns of feature-to-feature attention
3. **LayerNorm effects**: Need to account for normalization in attention computation

## 🚧 Current Challenges

### Technical Issues

- **LayerNorm complications**: `softmax(LN(x)^T M LN(x))` breaks clean algebraic decomposition
- **SAE completeness**: Pre-trained SAEs may not capture all attention-relevant features
- **Feature reconstruction**: Balancing SAE reconstruction quality with interpretability

### Methodological Questions

- **Ground truth validation**: How do we verify feature-level attention patterns?
- **Cross-head interactions**: How do multiple attention heads compose in feature space?
- **Scale dependency**: Do patterns hold across different model sizes?


## 📊 Evaluation Metrics

- **Reconstruction accuracy**: How well does feature-space attention predict token-space attention?
- **Interpretability gain**: Can we generate meaningful "feature X attends to feature Y" explanations?
- **Pattern recovery**: Do we recover known circuits (IOI, induction) at feature level?
- **Computational efficiency**: Cost vs. benefit of feature-space analysis

## 🔄 Reproducibility

All experiments use:
- **Model**: GPT-2 Small
- **SAEs**: Pre-trained from SAELens (`gpt2-small-resid-post-v5-32k`)
- **Analysis layer**: Layer 5, Head 7 (configurable)
