# %%

# # ABLATION

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_gpt2_model(model_name="gpt2-small"):
    """Load a GPT-2 model from TransformerLens"""
    model = HookedTransformer.from_pretrained(model_name)
    return model

def calculate_perplexity(model, texts, batch_size=8, max_length=512):
    """Calculate perplexity on a list of texts"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            tokens = model.tokenizer(batch_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=max_length
            ).input_ids.to('cuda')
            
            # Get logits
            logits = model(tokens)
            
            # Calculate loss for each sequence in batch
            for j, text_tokens in enumerate(tokens):
                # Remove padding and get actual length
                actual_length = (text_tokens != model.tokenizer.pad_token_id).sum().item()
                if actual_length <= 1:
                    continue
                    
                # Get logits and targets for this sequence
                seq_logits = logits[j, :actual_length-1, :]  # Exclude last position
                seq_targets = text_tokens[1:actual_length]    # Exclude first token (BOS)
                
                # Calculate cross-entropy loss
                loss = F.cross_entropy(seq_logits, seq_targets, reduction='sum')
                total_loss += loss.item()
                total_tokens += len(seq_targets)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss

def evaluate_next_token_prediction(model, prompts, top_k=5):
    """Evaluate next token prediction accuracy"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Evaluating predictions"):
            tokens = model.to_tokens(prompt, prepend_bos=True)
            logits = model(tokens)
            
            # Get logits for last position
            last_logits = logits[0, -1, :]
            
            # Get top-k predictions
            top_k_logits, top_k_indices = torch.topk(last_logits, top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Convert to tokens
            top_k_tokens = [model.to_string(idx) for idx in top_k_indices]
            
            results.append({
                'prompt': prompt,
                'predictions': list(zip(top_k_tokens, top_k_probs.cpu().numpy())),
                'entropy': -torch.sum(F.softmax(last_logits, dim=-1) * F.log_softmax(last_logits, dim=-1)).item()
            })
    
    return results

def evaluate_on_wikitext(model, split='test', num_samples=1000):
    """Evaluate model on WikiText dataset"""
    # Load WikiText-2 dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    
    # Filter out empty texts and take a sample
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
    # texts = texts[:num_samples]
    
    print(f"Evaluating on {len(texts)} WikiText samples...")
    
    # Calculate perplexity
    perplexity, avg_loss = calculate_perplexity(model, texts)
    
    return {
        'perplexity': perplexity,
        'average_loss': avg_loss,
        'num_samples': len(texts)
    }

def run_qualitative_tests(model):
    """Run qualitative tests with various prompts"""
    test_prompts = [
        "The capital of France is",
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect",
        "Once upon a time, in a land far away,",
        "The quick brown fox",
        "To be or not to be, that is the",
        "Machine learning is",
        "The weather today is"
    ]
    
    print("\n" + "="*50)
    print("QUALITATIVE EVALUATION")
    print("="*50)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Generate continuation
        continuation = model.generate(
            prompt, 
            max_new_tokens=20, 
            temperature=0.7,
            do_sample=True,
            stop_at_eos=True
        )
        print(f"Generated: {continuation}")
        
        # Show top predictions for next token
        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens)
        last_logits = logits[0, -1, :]
        
        top_5_logits, top_5_indices = torch.topk(last_logits, 5)
        top_5_probs = F.softmax(top_5_logits, dim=-1)
        
        print("Top 5 next token predictions:")
        for i, (idx, prob) in enumerate(zip(top_5_indices, top_5_probs)):
            token = model.to_string(idx)
            print(f"  {i+1}. '{token}' (prob: {prob:.3f})")

def analyze_attention_patterns(model, text, layer_idx=-1, head_idx=0):
    """Analyze attention patterns for a given text"""
    tokens = model.to_tokens(text, prepend_bos=True)
    
    # Run with cache to get attention patterns
    logits, cache = model.run_with_cache(tokens)
    
    # Get attention pattern from specified layer and head
    attention = cache[f'blocks.{layer_idx}.attn.pattern'][0, head_idx]  # [seq_len, seq_len]
    
    # Convert tokens to strings for visualization
    token_strs = [model.to_string(tok) for tok in tokens[0]]
    
    return attention.cpu().numpy(), token_strs

def patch_attention_head_output(model, layer_idx, head_idx):
    """Patch out (zero) the output of a specific attention head"""
    def head_output_hook(value, hook):
        # value shape: [batch, seq_len, n_heads, d_head]
        value[:, :, head_idx, :] = value[:, :, head_idx, :].mean(dim=(0,1 ), keepdim=True)  # Zero out the head output
        return value
    
    # Add hook to the attention output (use hook_z instead of hook_result)
    hook_name = f"blocks.{layer_idx}.attn.hook_z"
    model.add_hook(hook_name, head_output_hook)
    return model

# %%"""Main evaluation function"""
print("Loading GPT-2 model...")
model = load_gpt2_model("gpt2-small")  # Options: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl

print(f"Model loaded: {model.cfg.model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Patch out layer 1, head 5
print("Patching out layer 1, head 5...")
model = patch_attention_head_output(model, layer_idx=1, head_idx=5)

# %%

# Run qualitative tests
# run_qualitative_tests(model)

# Evaluate on WikiText (comment out if you don't want to download dataset)
print("\n" + "="*50)
print("QUANTITATIVE EVALUATION ON WIKITEXT (with layer 1, head 5 patched)")
print("="*50)

wikitext_results = evaluate_on_wikitext(model, num_samples=100)  # Reduced for speed
print(f"WikiText Perplexity: {wikitext_results['perplexity']:.2f}")
print(f"Average Loss: {wikitext_results['average_loss']:.4f}")
    # # Analyze attention for a sample text
    # print("\n" + "="*50)
    # print("ATTENTION ANALYSIS")
    # print("="*50)
    
    # sample_text = "The cat sat on the mat and looked around."
    # attention, tokens = analyze_attention_patterns(model, sample_text)
    
    # print(f"Analyzing attention for: '{sample_text}'")
    # print(f"Attention matrix shape: {attention.shape}")
    # print(f"Tokens: {tokens}")
    
    # Show attention weights from last token to all previous tokens
    # last_token_attention = attention[-1, :]
    # print(f"\nAttention from '{tokens[-1]}' to all tokens:")
    # for i, (token, weight) in enumerate(zip(tokens, last_token_attention)):
    #     print(f"  {token}: {weight:.3f}")

