from transformers import GPT2TokenizerFast
from collections import Counter
import datasets

from tqdm import tqdm
import csv

# Load GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Load a large English dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Count token frequencies
counter = Counter()
for text in tqdm(dataset["text"]):
    tokens = tokenizer.encode(text)
    counter.update(tokens)

# Save token frequencies as CSV with columns: token, id, count

with open("token_frequencies.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["token", "id", "count"])
    for token_id, freq in counter.most_common():
        token = tokenizer.decode([token_id]).replace("\n", "\\n")
        writer.writerow([token, token_id, freq])
