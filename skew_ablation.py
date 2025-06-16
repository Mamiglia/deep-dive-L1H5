import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
import re
import random

def head_ablation(value, hook, components_to_ablate, cache_dict, model):
    # Access the attention input from our cache - use the resid_pre hook instead
    attn_input_hook_name = 'blocks.1.ln1.hook_normalized'
    attn_input = cache_dict[attn_input_hook_name]
    
    # Get the attention layer
    layer_idx = 1
    attn_layer = model.blocks[layer_idx].attn
    
    # Extract weight matrices
    W_Q = attn_layer.W_Q[5]  # Head 5 query weights
    W_K = attn_layer.W_K[5]  # Head 5 key weights  
    W_V = attn_layer.W_V[5]  # Head 5 value weights
    
    # Get biases
    b_Q = attn_layer.b_Q[5] if attn_layer.b_Q is not None else 0
    b_K = attn_layer.b_K[5] if attn_layer.b_K is not None else 0
    b_V = attn_layer.b_V[5] if attn_layer.b_V is not None else 0
    
    # Experiment: Set W_K = W_Q (or other modifications)
    # W_K_modified = torch.clone(W_K)  # Use key weights for keys
    # Alternative experiments:
    # W_K_modified = torch.zeros_like(W_K)  # Zero out key computation
    # W_K_modified = W_V.clone()  # Use value weights for keys
    
    # W_QK = W_Q @ W_K.T
    # W_sym = (W_QK + W_QK.T) / 2
    # W_skew = (W_QK - W_QK.T) / 2
    # # S, Q = torch.linalg.eigh(W_sym)
    # W_Qp, S, W_Kp = torch.linalg.svd(W_skew, full_matrices=False)
    # W_Qp = W_Qp[:,:64] @ torch.diag(S[:64])  # Keep top 64 components
    # W_Kp = W_Kp[:64,:].T 
    
    W_Qp = W_Q
    W_Kp = W_K

    
    # Manually compute queries, keys, values
    queries = torch.einsum('bsd,dh->bsh', attn_input, W_Qp) + b_Q
    keys = torch.einsum('bsd,dh->bsh', attn_input, W_Kp) + b_K
    values = torch.einsum('bsd,dh->bsh', attn_input, W_V) + b_V
    
    # Compute attention scores
    scores = torch.einsum('bsh,bth->bst', queries, keys) / (attn_layer.cfg.d_head ** 0.5)
    
    # Apply causal mask
    seq_len = scores.shape[1]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Compute attention pattern
    attn_pattern = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    attn_out = torch.einsum('bst,bth->bsh', attn_pattern, values)
    
    # Replace head 5's output with our modified computation
    value[:, :, 5, :] = attn_out.mean(dim=(0,1))
    
    return value
    
def create_succession_test_cases():
    """Create test cases specifically for succession/pairs behavior"""
    
    # Ordinal numbers
    ordinal_cases = [
        ("The first, second, and", " third"),
        ("Chapter one, chapter two, chapter", " three"),
        ("Grade 1, Grade 2, Grade", " 3"),
        ("Level I, Level II, Level", " III"),
        ("Season 1, Season 2, Season", " 3"),
        ("Round one, round two, round", " three"),
        ("First place, second place,", " third"),
        ("Primary, secondary,", " tertiary"),
        ("Initial, intermediate,", " final"),
    ]
    
    # Days of the week
    weekday_cases = [
        ("Monday, Tuesday,", " Wednesday"),
        ("Wednesday, Thursday,", " Friday"),
        ("Friday, Saturday,", " Sunday"),
        ("Sunday, Monday,", " Tuesday"),
        ("Tuesday, Wednesday, Thursday,", " Friday"),
        ("On Monday and Tuesday, then", " Wednesday"),
        ("From Monday to Tuesday to", " Wednesday"),
        ("Last Monday, this Tuesday, next", " Wednesday"),
        ("Mon, Tue,", " Wed"),
        ("Thu, Fri,", " Sat"),
    ]
    
    # Months
    month_cases = [
        ("January, February,", " March"),
        ("March, April,", " May"),
        ("May, June,", " July"),
        ("September, October,", " November"),
        ("October, November,", " December"),
        ("December, January,", " February"),  # Wrapping around
        ("In January and February, then", " March"),
        ("From June to July to", " August"),
        ("Jan, Feb,", " Mar"),
        ("Apr, May,", " Jun"),
        ("Jul, Aug,", " Sep"),
        ("Oct, Nov,", " Dec"),
    ]
    
    # Colors (common color sequences/spectrums)
    color_cases = [
        ("red, orange,", " yellow"),  # Rainbow order
        ("orange, yellow,", " green"),
        ("yellow, green,", " blue"),
        ("green, blue,", " purple"),
        ("blue, purple,", " pink"),  # Less standard but possible
        ("Red, orange, yellow,", " green"),
        ("The colors red, orange,", " yellow"),
        ("Primary colors: red, blue,", " yellow"),  # Not sequential but grouped
        ("Traffic light: red, yellow,", " green"),
        ("RGB: red, green,", " blue"),
        ("black, white,", " gray"),  # Grayscale
        ("light, medium,", " dark"),
    ]
    
    # Single letters (alphabetical)
    letter_cases = [
        ("A, B,", " C"),
        ("B, C,", " D"),
        ("X, Y,", " Z"),
        ("a, b,", " c"),
        ("d, e,", " f"),
        ("Option A, Option B, Option", " C"),
        ("Plan A, Plan B, Plan", " C"),
        ("Section a, section b, section", " c"),
        ("Part (a), part (b), part", " (c)"),
        ("Point A, Point B, Point", " C"),
        ("Step A, Step B, Step", " C"),
        ("Phase I, Phase II, Phase", " III"),
    ]
    
    # Pronouns (grammatical sequences/pairs)
    pronoun_cases = [
        ("I, you,", " he"),  # Person sequence
        ("me, you,", " him"),
        ("my, your,", " his"),
        ("mine, yours,", " his"),
        ("myself, yourself,", " himself"),
        ("He, she,", " they"),  # Gender progression
        ("him, her,", " them"),
        ("his, hers,", " theirs"),
        ("Men, women,", " children"),
        ("Boys, girls,", " children"),
        ("Father, mother,", " child"),
        ("Man, woman,", " child"),
        ("Male, female,", " neutral"),
    ]
    
    # Countries (geographical/alphabetical groupings)
    country_cases = [
        ("France, Germany,", " Italy"),  # European neighbors
        ("Canada, Mexico,", " USA"),     # North American
        ("China, Japan,", " Korea"),     # East Asian
        ("England, Scotland,", " Wales"), # UK
        ("Argentina, Brazil,", " Chile"), # South American (alphabetical)
        ("Australia, Canada,", " Denmark"), # Alphabetical
        ("India, Indonesia,", " Iran"),   # Alphabetical
        ("Spain, Sweden,", " Switzerland"), # Alphabetical
        ("Kenya, Libya,", " Morocco"),    # African (alphabetical)
        ("Peru, Poland,", " Portugal"),   # Alphabetical
        ("Egypt, France,", " Germany"),   # Alphabetical
        ("Austria, Belgium,", " Canada"), # Alphabetical
    ]
    
    # Numbers (various patterns)
    numerical_cases = [
        ("1, 2,", " 3"),
        ("2, 3,", " 4"),
        ("5, 6,", " 7"),
        ("8, 9,", " 10"),
        ("10, 11,", " 12"),
        ("2015, 2016,", " 2017"),
        ("2020, 2021,", " 2022"),
        ("10, 20,", " 30"),
        ("5, 10,", " 15"),
        ("100, 200,", " 300"),
        ("The years 2010, 2011,", " 2012"),
        ("Ages 5, 6,", " 7"),
        ("Rooms 101, 102,", " 103"),
        ("Page 1, page 2, page", " 3"),
        ("Chapter 1, Chapter 2, Chapter", " 3"),
        ("Volume 1, Volume 2, Volume", " 3"),
        ("First, second,", " third"),  # Word numbers
        ("One, two,", " three"),
        ("Twenty, thirty,", " forty"),
        ("Fifty, sixty,", " seventy"),
    ]
    
    # Roman numerals  
    roman_cases = [
        ("I, II,", " III"),
        ("II, III,", " IV"),
        ("III, IV,", " V"),
        ("Chapter I, Chapter II, Chapter", " III"),
        ("Part I, Part II, Part", " III"),
        ("Volume I, Volume II, Volume", " III"),
        ("Section I, Section II, Section", " III"),
        ("Book I, Book II, Book", " III"),
        ("Act I, Act II, Act", " III"),
    ]
    
    # Mixed categories for robustness
    mixed_cases = [
        ("Morning, afternoon,", " evening"),  # Times of day
        ("Spring, summer,", " fall"),         # Seasons
        ("Breakfast, lunch,", " dinner"),     # Meals
        ("Child, teenager,", " adult"),       # Life stages
        ("Small, medium,", " large"),         # Sizes
        ("Low, medium,", " high"),            # Levels
        ("Past, present,", " future"),        # Time
        ("Beginning, middle,", " end"),       # Story parts
        ("Introduction, body,", " conclusion"), # Essay parts
    ]
    
    return (ordinal_cases + weekday_cases + month_cases + color_cases + 
            letter_cases + pronoun_cases + country_cases + numerical_cases + 
            roman_cases + mixed_cases)
def succession_logit_diff_metric(model, ablate_components=None):
    """
    Measure logit difference on succession-specific tasks
    """
    test_cases = create_succession_test_cases()
    
    results = []
    
    for prompt, correct_answer in tqdm(test_cases, desc="Testing succession patterns"):
        try:
            correct_id = model.to_single_token(correct_answer)
        except:
            # Skip if answer isn't a single token
            continue
            
        tokens = model.to_tokens(prompt, prepend_bos=True)
        
        # Clean run
        clean_logits = model(tokens)
        clean_correct_logit = clean_logits[0, -1, correct_id].item()
        
        # Get top wrong answer for comparison
        top_logits, top_indices = torch.topk(clean_logits[0, -1], 10)
        wrong_candidates = [idx for idx in top_indices if idx != correct_id]
        if wrong_candidates:
            wrong_id = wrong_candidates[0]  # Top wrong answer
            clean_wrong_logit = clean_logits[0, -1, wrong_id].item()
        else:
            continue
            
        clean_diff = clean_correct_logit - clean_wrong_logit
        
        # Ablated run with caching
        if ablate_components:
            # Set up hooks to cache attention input and apply ablation
            cache_dict = {}
            
            def cache_input(activations, hook):
                cache_dict[hook.name] = activations
                return activations
            
            def ablate_with_cache(activations, hook):
                return head_ablation(activations, hook, ablate_components, cache_dict, model)
            
            # Cache the attention input and apply ablation - use resid_pre instead
            fwd_hooks = [
                ('blocks.1.ln1.hook_normalized', cache_input),
                (ablate_components[0], lambda x, hook: ablate_with_cache(x, hook))
            ]
            
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            ablated_correct_logit = ablated_logits[0, -1, correct_id].item()
            ablated_wrong_logit = ablated_logits[0, -1, wrong_id].item()
            ablated_diff = ablated_correct_logit - ablated_wrong_logit
        else:
            ablated_diff = clean_diff
        
        effect = clean_diff - ablated_diff
        
        results.append({
            'prompt': prompt,
            'correct': correct_answer,
            'clean_diff': clean_diff,
            'ablated_diff': ablated_diff,
            'effect': effect,
            'wrong_answer': model.to_string(wrong_id)
        })
    
    return results


def extract_succession_patterns_from_wikitext(dataset, num_samples=1000):
    """
    Extract succession patterns from WikiText dataset - completely rewritten
    """
    patterns = []
    
    # Define succession sequences that head 1.5 should handle
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    
    weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'violet',
              'black', 'white', 'gray', 'grey', 'pink', 'brown']
    
    letters = list('abcdefghijklmnopqrstuvwxyz')
    
    pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their']
    
    countries = ['usa', 'canada', 'mexico', 'france', 'germany', 'italy', 'spain', 
                 'uk', 'britain', 'england', 'russia', 'china', 'japan', 'india',
                 'australia', 'brazil', 'argentina']
    
    ordinals = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh']
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    
    # Create lookup dictionaries for finding successors
    def create_successor_map(seq_list):
        return {seq_list[i]: seq_list[i+1] for i in range(len(seq_list)-1)}
    
    month_successors = create_successor_map(months)
    weekday_successors = create_successor_map(weekdays)
    color_successors = create_successor_map(colors)
    letter_successors = create_successor_map(letters)
    pronoun_successors = create_successor_map(pronouns)
    country_successors = create_successor_map(countries)
    ordinal_successors = create_successor_map(ordinals)
    number_successors = create_successor_map(numbers)
    
    all_successors = {
        **month_successors, **weekday_successors, **color_successors, 
        **letter_successors, **pronoun_successors, **country_successors,
        **ordinal_successors, **number_successors
    }
    
    sample_count = 0
    texts_processed = 0
    
    for item in tqdm(dataset, desc="Mining WikiText for succession patterns"):
        if sample_count >= num_samples or texts_processed >= 5000:
            break
            
        text = item['text'].lower()  # Convert to lowercase for matching
        if len(text.strip()) < 50:
            continue
            
        texts_processed += 1
        words = re.findall(r'\b\w+\b', text)
        
        # Look for any word that has a successor, followed by patterns
        for i in range(len(words) - 2):
            current_word = words[i]
            next_word = words[i + 1]
            
            # Check if current word has a known successor
            if current_word in all_successors:
                expected_successor = all_successors[current_word]
                
                # Look for patterns like "X, Y" or "X and Y" or "X then Y"
                if i + 2 < len(words):
                    # Pattern: "word1, word2, ..." -> predict word3
                    if (next_word == ',' or next_word in ['and', 'then', 'to']) and i + 2 < len(words):
                        following_word = words[i + 2]
                        if following_word == expected_successor:
                            # Found a succession pattern! Create test case
                            context_start = max(0, i - 5)  # Include some context
                            context_words = words[context_start:i + 2]  # Up to but not including the answer
                            context = ' '.join(context_words)
                            
                            patterns.append((context, f" {expected_successor}"))
                            sample_count += 1
                            continue
                
                # Pattern: "from X to Y" 
                if next_word == 'to' and i + 2 < len(words):
                    following_word = words[i + 2]
                    if following_word == expected_successor:
                        context_start = max(0, i - 5)
                        context_words = words[context_start:i + 2]
                        context = ' '.join(context_words)
                        patterns.append((context, f" {expected_successor}"))
                        sample_count += 1
                        continue
                
                # Pattern: direct succession "X Y" where Y is successor
                if next_word == expected_successor:
                    # Create a prompt that ends with X, expecting Y
                    context_start = max(0, i - 5)
                    context_words = words[context_start:i + 1]  # Include current word
                    context = ' '.join(context_words)
                    patterns.append((context, f" {expected_successor}"))
                    sample_count += 1
        
        # Also look for simple number sequences
        for i in range(len(words) - 2):
            try:
                if (words[i].isdigit() and words[i + 1].isdigit() and 
                    int(words[i + 1]) == int(words[i]) + 1):
                    expected_next = str(int(words[i + 1]) + 1)
                    context_start = max(0, i - 3)
                    context_words = words[context_start:i + 2]
                    context = ' '.join(context_words)
                    patterns.append((context, f" {expected_next}"))
                    sample_count += 1
            except (ValueError, IndexError):
                continue
        
        # Look for letter sequences (a, b, c)
        for i in range(len(words) - 2):
            if (len(words[i]) == 1 and len(words[i + 1]) == 1 and 
                words[i].isalpha() and words[i + 1].isalpha()):
                if ord(words[i + 1]) == ord(words[i]) + 1:
                    expected_next = chr(ord(words[i + 1]) + 1)
                    if expected_next.isalpha() and expected_next <= 'z':
                        context_start = max(0, i - 3)
                        context_words = words[context_start:i + 2]
                        context = ' '.join(context_words)
                        patterns.append((context, f" {expected_next}"))
                        sample_count += 1
    
    print(f"Found {len(patterns)} succession patterns from {texts_processed} WikiText entries")
    
    # Show some examples of what we found
    if patterns:
        print("Sample patterns found:")
        for i, (prompt, answer) in enumerate(patterns[:10]):
            print(f"  '{prompt}' → '{answer}'")
    
    return patterns

def wikitext_succession_metric(model, ablate_components=None, num_samples=500):
    """
    Measure succession performance on WikiText dataset
    """
    print("Loading WikiText dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    print("Extracting succession patterns from WikiText...")
    test_cases = extract_succession_patterns_from_wikitext(dataset, num_samples)
    
    print(f"Found {len(test_cases)} succession patterns in WikiText")
    if len(test_cases) < 10:
        print("Warning: Very few patterns found. WikiText might not have enough succession patterns.")
        return None

    results = []
    valid_cases = 0
    
    for prompt, correct_answer in tqdm(test_cases, desc="Testing WikiText succession"):
        try:
            correct_id = model.to_single_token(correct_answer)
        except:
            continue
            
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 100:  # Skip very long contexts
            continue
            
        # Clean run
        clean_logits = model(tokens)
        clean_correct_logit = clean_logits[0, -1, correct_id].item()
        
        # Get probability rank of correct answer
        sorted_logits = torch.sort(clean_logits[0, -1], descending=True)
        clean_rank = (sorted_logits.indices == correct_id).nonzero().item()
        clean_prob = F.softmax(clean_logits[0, -1], dim=-1)[correct_id].item()
        
        # Ablated run with caching
        if ablate_components:
            cache_dict = {}
            
            def cache_input(activations, hook):
                cache_dict[hook.name] = activations
                return activations
            
            def ablate_with_cache(activations, hook):
                return head_ablation(activations, hook, ablate_components, cache_dict, model)
            
            # Use resid_pre hook instead of non-existent q_input hook
            fwd_hooks = [
                ('blocks.1.ln1.hook_normalized', cache_input),
                (ablate_components[0], lambda x, hook: ablate_with_cache(x, hook))
            ]
            
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            ablated_correct_logit = ablated_logits[0, -1, correct_id].item()
            sorted_ablated = torch.sort(ablated_logits[0, -1], descending=True)
            ablated_rank = (sorted_ablated.indices == correct_id).nonzero().item()
            ablated_prob = F.softmax(ablated_logits[0, -1], dim=-1)[correct_id].item()
        else:
            ablated_correct_logit = clean_correct_logit
            ablated_rank = clean_rank
            ablated_prob = clean_prob
        
        logit_effect = clean_correct_logit - ablated_correct_logit
        rank_change = ablated_rank - clean_rank
        prob_drop = clean_prob - ablated_prob
        
        results.append({
            'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'correct': correct_answer,
            'logit_effect': logit_effect,
            'rank_change': rank_change,
            'prob_drop': prob_drop,
            'clean_rank': clean_rank,
            'ablated_rank': ablated_rank
        })
        
        valid_cases += 1
        if valid_cases >= 200:  # Limit for speed
            break
    
    return results

def comprehensive_succession_ablation(model, layer_idx, head_idx):
    """
    Comprehensive ablation study for succession head
    """
    ablate_component = f"blocks.{layer_idx}.attn.hook_z"
    ablate_components = [ablate_component]

    print(f"Ablating {ablate_component}")
    print("="*60)
    
    results = {}
    
    # 1. Curated succession test cases
    print("1. Testing curated succession patterns...")
    succession_results = succession_logit_diff_metric(model, ablate_components)
    
    if succession_results:
        avg_effect = np.mean([r['effect'] for r in succession_results])
        results['curated_avg_logit_effect'] = avg_effect
        print(f"   Average logit difference effect: {avg_effect:.4f}")
        
        # Show some examples
        print("   Sample results:")
        for i, r in enumerate(succession_results[:5]):
            print(f"     '{r['prompt']}' → '{r['correct']}' (effect: {r['effect']:.3f})")
    
    # 2. WikiText succession patterns
    print("\n2. Testing WikiText succession patterns...")
    wikitext_results = wikitext_succession_metric(model, ablate_components, num_samples=300)
    
    if wikitext_results and len(wikitext_results) > 0:
        avg_logit_effect = np.mean([r['logit_effect'] for r in wikitext_results])
        avg_rank_change = np.mean([r['rank_change'] for r in wikitext_results])
        avg_prob_drop = np.mean([r['prob_drop'] for r in wikitext_results])
        
        results['wikitext_avg_logit_effect'] = avg_logit_effect
        results['wikitext_avg_rank_change'] = avg_rank_change  
        results['wikitext_avg_prob_drop'] = avg_prob_drop
        
        print(f"   Average logit effect: {avg_logit_effect:.4f}")
        print(f"   Average rank change: {avg_rank_change:.2f}")
        print(f"   Average probability drop: {avg_prob_drop:.4f}")
        
        # Show examples where effect was strongest
        sorted_results = sorted(wikitext_results, key=lambda x: x['logit_effect'], reverse=True)
        print("   Strongest effects:")
        for r in sorted_results[:3]:
            print(f"     '{r['prompt']}' → '{r['correct']}' (logit Δ: {r['logit_effect']:.3f}, rank: {r['clean_rank']}→{r['ablated_rank']})")
    else:
        print("   No succession patterns found in WikiText sample")
    
    # 3. Summary
    print(f"\n3. Summary for head {layer_idx}.{head_idx}:")
    print("="*40)
    for metric, value in results.items():
        print(f"   {metric}: {value:.4f}")
        
    # Interpretation
    if 'curated_avg_logit_effect' in results:
        effect = results['curated_avg_logit_effect']
        if effect > 0.5:
            print(f"   → STRONG succession effect detected!")
        elif effect > 0.1:
            print(f"   → Moderate succession effect detected")
        else:
            print(f"   → Weak/no succession effect")
    
    return results

# Example usage
def main():
    print("Loading GPT-2 model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    # Test the specific succession head
    results = comprehensive_succession_ablation(model, layer_idx=1, head_idx=5)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()