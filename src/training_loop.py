import torch
import torch.nn as nn
from transformers import AutoTokenizer
from multitask_model import MultiTaskSentenceModel

# 1. Hypothetical training data (3 sentences + fake labels for 3 tasks)
sentences = [
    "Your receipt has been uploaded to your account.",
    "The receipt image is blurry and hard to read.",
    "Looking for deals on electronics and coffee brands."
]

sentence_labels = torch.tensor([0, 0, 2])         # Task A: Sentence classification (5 classes)
quality_labels = torch.tensor([0, 1, 2])          # Task B1: Receipt quality (3 classes)
intent_labels = torch.tensor([3, 1, 0])           # Task B2: Query intent (4 classes)

# 2. Load tokenizer and model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MultiTaskSentenceModel()

# 3. Tokenize inputs
encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 4. Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 5. Training step (forward + loss + backward + update)
model.train()
outputs = model(encoded["input_ids"], encoded["attention_mask"])

# 6. Compute task-specific losses
loss_sentence = loss_fn(outputs["sentence_type"], sentence_labels)
loss_quality = loss_fn(outputs["receipt_quality"], quality_labels)
loss_intent = loss_fn(outputs["query_intent"], intent_labels)

# 7. Combine losses (simple sum)
total_loss = loss_sentence + loss_quality + loss_intent

# 8. Backprop and update
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("Training step complete.")
print(f"Total Loss: {total_loss.item():.4f}")


