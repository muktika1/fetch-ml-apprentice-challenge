# fetch-ml-apprentice-challenge

## Task 1: Sentence Transformer Implementation
🧠 Model Choice
For sentence embeddings, I selected the all-MiniLM-L6-v2 model from HuggingFace. This model is specifically fine-tuned for generating high-quality semantic sentence embeddings using contrastive learning. I considered other models such as BAAI/bge-small-en-v1.5 and all-mpnet-base-v2, but chose MiniLM due to its excellent trade-off between speed and performance.

🛠️ Architecture Choices Framework: I used PyTorch with HuggingFace Transformers for flexibility, control, and extensibility — particularly to support later tasks involving multi-task learning and custom model heads.

Embedding Strategy: I extracted sentence embeddings by applying mean pooling over token embeddings from the last hidden state. This method has been shown to outperform using the [CLS] token alone in many downstream tasks.

Tokenizer: The AutoTokenizer from HuggingFace was used to preprocess input sentences into token IDs, attention masks, and proper formatting for the model.

Sentence Examples: To test embedding quality and thematic relevance, I created 5 custom sentences inspired by real-world use of the Fetch Rewards app (e.g., receipt scanning, reward redemptions, lifestyle integrations).

Normalization: I applied L2 normalization to each final embedding vector to prepare for cosine similarity comparisons and to align with standard semantic search practices.

🧪 Testing Ran 5 Fetch-themed sentences through the model.

Verified that each sentence was converted into a fixed-length 384-dimensional embedding vector.

Printed the first sentence’s embedding and confirmed the full shape of the output as (5, 384).
