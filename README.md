# Investigating Semantic Attention in gpt2-small's Head L1H5

This repository contains the code and analysis for a research project investigating the unusual behavior of attention head L1H5 in gpt2-small. This head demonstrates a unique ability to attend to semantically similar tokens while actively suppressing self-attention, seemingly independent of positional information.

The goal of this project is to provide a mechanistic explanation for this behavior and to develop generalizable techniques for analyzing attention patterns in Large Language Models (LLMs).
## Core Concepts and Research Focus

Our investigation into gpt2-small's attention head L1H5 reveals a fascinating mechanism: it selectively attends to semantically related tokens while actively suppressing self-attention. This behavior follows three simple rules:

1.  **Semantic Clustering**: Tokens attend to others in the same semantic category (e.g., `cat` to `dog`, `red` to `blue`).

2.  **Self-Suppression**: Tokens do not attend to themselves or other instances of the same token.

3.  **Fallback to Beginning**: If no other in-category tokens are present, attention falls back to the `<bos>` (beginning of sequence) token.

The core of this research involves:

-   **Characterizing the Behavior**: Designing prompts and metrics to reliably trigger and measure L1H5's unique attention patterns.

-   **Component Ablation**: Identifying which parts of the model (e.g., token embeddings, MLP layers) are crucial for this specific head's function, notably observing its independence from positional information.

-   **Mechanism Discovery**: Decomposing the attention head's QK circuit (`W_QK`) into symmetric and skew-symmetric parts to pinpoint how self-suppression is achieved. Our findings suggest that negative eigenvalues within the symmetric component play a key role.

-   **Causal Steering**: Demonstrating direct control over the self-suppression behavior by manipulating these negative eigenvalues or using gradient-based methods, which validates our mechanistic hypothesis.

This repository provides the computational framework and analysis scripts to reproduce and extend these findings.
## Project Structure

A brief overview of each analysis script:

- **characterize_head.py**: Examine L1H5’s semantic attention via custom prompts and visualize patterns.
- **component_ablation.py**: Ablate embeddings, MLPs, and attention to identify components critical for L1H5.
- **clustering.py**: Cluster tokens based on L1H5 attention similarities and generate community graphs.
- **cosine_attention_analysis.py**: Relate attention scores to cosine similarity of token embeddings.
- **gradient_steering.py**: Steer self-attention behavior by gradient-based updates to Q/K weights.
- **skew_analysis.py**: Decompose W_QK into symmetric/skew parts to understand self-suppression mechanisms.
- **svd_sym_skew_heatmap.py**: Visualize heatmaps of symmetric vs. skew-symmetric attention components.
- **svd_rank_ablation.py**: Measure effects of manipulating top singular values on attention rank metrics.
- **svd_group_scaling_heatmap.py**: Scale eigenvalues for group tokens and plot resulting heatmaps.
- **semantic_head.py**: Initial exploratory analysis of semantic attention using Sparse Autoencoders.
- **token_freq.py**: Compute GPT-2 token frequencies for vocabulary selection.

## Installation and Usage

To run the code in this repository, you'll need Python and the necessary machine learning and scientific computing libraries.
### Prerequisites

Standard Python libraries for deep learning, natural language processing, data analysis, and visualization. You can install them via pip:
```bash
pip install torch transformer_lens sae_lens numpy pandas seaborn matplotlib scikit-learn rich circuitsvis leidenalg igraph tqdm
```

