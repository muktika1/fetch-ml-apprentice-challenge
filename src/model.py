#model.py
from transformers import AutoTokenizer, AutoModel
import torch

# 1. Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Example sentences
sentences = [
    "Snapping receipts with Fetch turns my shopping sprees into gift card galore.",
    "Fetch rewards me for every coffee run; my caffeine addiction finally pays off!",
    "Thanks to Fetch, my online shopping habits are funding my next vacation.",
    "Who knew scanning grocery receipts could lead to free movie tickets? Fetch did!",
    "Fetch makes my dining-out splurges feel like savvy investments in future meals."
]

# 3. Tokenize
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 4. Forward pass
with torch.no_grad():
    model_output = model(**encoded_input)

# 5. Mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# 6. Normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

# 7. Print the first sentence's embedding
print("Embedding for first sentence:\n", sentence_embeddings[0])
print("Shape of all embeddings:", sentence_embeddings.shape)

# 8. Save all sentence embeddings to a file for visibility
output_path = "src/outputs/sample_embeddings.txt"
with open(output_path, "w") as f:
    for i, sentence in enumerate(sentences):
        f.write(f"Sentence {i+1}: {sentence}\n")
        f.write(f"Embedding: {sentence_embeddings[i].tolist()}\n\n")

print(f"Embeddings saved to {output_path}")
