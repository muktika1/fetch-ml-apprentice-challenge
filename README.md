# fetch-ml-apprentice-challenge

## Task 1: Sentence Transformer Implementation

🧠 Model Choice

For sentence embeddings, I selected the all-MiniLM-L6-v2 model from HuggingFace. This model is specifically fine-tuned for generating high-quality semantic sentence embeddings using contrastive learning. I considered other models such as BAAI/bge-small-en-v1.5 and all-mpnet-base-v2, but chose MiniLM due to its excellent trade-off between speed and performance.

🛠️ Architecture Choices Framework: 

I used PyTorch with HuggingFace Transformers for flexibility, control, and extensibility — particularly to support later tasks involving multi-task learning and custom model heads.

Embedding Strategy: I extracted sentence embeddings by applying mean pooling over token embeddings from the last hidden state. This method has been shown to outperform using the [CLS] token alone in many downstream tasks.

Tokenizer: The AutoTokenizer from HuggingFace was used to preprocess input sentences into token IDs, attention masks, and proper formatting for the model.

Sentence Examples: To test embedding quality and thematic relevance, I created 5 custom sentences inspired by real-world use of the Fetch Rewards app (e.g., receipt scanning, reward redemptions, lifestyle integrations).

Normalization: I applied L2 normalization to each final embedding vector to prepare for cosine similarity comparisons and to align with standard semantic search practices.

🧪 Testing 

Ran 5 Fetch-themed sentences through the model.

Verified that each sentence was converted into a fixed-length 384-dimensional embedding vector.

Printed the first sentence’s embedding and confirmed the full shape of the output as (5, 384).

### Why I Didn't Use sentence-transformers

While the `sentence-transformers` library provides a convenient `.encode()` method for sentence embeddings, I chose to implement the sentence transformer manually using HuggingFace Transformers and PyTorch. This allowed me to:
- Customize pooling (mean pooling instead of CLS token)
- Maintain control over the architecture for multi-task expansion in later tasks
- Demonstrate familiarity with transformer internals, which aligns with Fetch's emphasis on engineering depth and adaptability

## Task 2: Multi-Task Learning Expansion

Task 2A: Sentence Classification (Multi-Task Learning – Part 1)

🧠 Task Overview

For Task A of the multi-task learning setup, I expanded the original sentence transformer model to include a classification head that predicts the category of a given sentence. The head outputs logits corresponding to five Fetch-aligned sentence classes:

- receipt

- fraud_alert

- offer

- help_request

- generic_text

These classes were chosen to align with real-world Fetch domains such as receipt parsing, fraud detection, promotions, support inquiries, and general communications.

🧱 Architecture Changes

I created a custom PyTorch model class MultiTaskSentenceModel, which:

Loads a MiniLM transformer backbone (all-MiniLM-L6-v2)

Applies mean pooling to convert token embeddings into a single sentence vector

Passes the pooled vector into a classification head: nn.Linear(384, 5)

The model outputs raw logits for 5 sentence types.

No training has been applied yet — the classification head is initialized with random weights.


Task 2B: Multi-Task Learning Expansion – Receipt Quality & Query Intent

🧠 Task Overview

For the second part of the multi-task learning expansion, I implemented two additional output heads on top of the shared sentence transformer backbone:

🔹 Task B1: Receipt Quality Classification

This head classifies a sentence into one of the following categories:

- good_quality

- blurry

- incomplete

This task was inspired by Fetch’s focus on receipt understanding and processing, and simulates how a model might assess whether a receipt image or its description is complete and usable.

🔹 Task B2: Search/Query Intent Classification

This head predicts the intent behind a user query or sentence, with labels such as:

- brand_search

- deal_search

- support_search

- promo_search

This simulates potential ML tasks related to search ranking, ad targeting, and personalization, which align with Fetch’s stated priorities in their machine learning roadmap.

🏗️ Architecture

Both tasks share the same transformer encoder (MiniLM) and pooled sentence embeddings. On top of this, I added:

A 3-class classification head for receipt quality

A 4-class classification head for query intent

This structure supports efficient multi-task learning, leveraging shared semantics while supporting task-specific objectives.

## Task 3: Training Considerations

🔹 Scenario 1: Freezing the Entire Network

Implication: No part of the model — including the transformer backbone and task-specific heads — is updated during training.

This approach is typically used only during inference, where a model is deployed for predictions but not trained further. In this scenario, the model can produce outputs based on its pretrained knowledge, but cannot adapt or learn task-specific representations. This is not suitable for the multi-task setting in this project, since both the classification and auxiliary task heads are randomly initialized and require training.

Conclusion: Freezing the entire model prevents learning and is not appropriate for this use case.

🔹 Scenario 2: Freezing Only the Transformer Backbone

Implication: The transformer (MiniLM) remains unchanged, but the task-specific heads (for sentence classification, receipt quality, and query intent) are trained.

This is a common and effective transfer learning strategy, especially when the pretrained backbone already provides high-quality sentence embeddings. Freezing the backbone:

- Preserves the language knowledge learned during large-scale pretraining
- Speeds up training by reducing the number of parameters to update
- Reduces the risk of overfitting on small downstream datasets

This approach is particularly useful when the goal is to adapt the model to specific tasks (like the ones in this project) without retraining the entire transformer.

Conclusion: This is a highly recommended approach for fine-tuning task-specific heads in real-world scenarios, especially when resources or data are limited.

🔹 Scenario 3: Freezing One Task-Specific Head

Implication: One of the heads (e.g., sentence classification) is frozen while others are trainable.

This strategy is helpful in continual learning or multi-stage training pipelines. For example, if the receipt quality classifier has already been trained and deployed, you may wish to keep it frozen while training a new query intent classifier. This allows you to extend the model's capabilities without degrading performance on previously learned tasks.

It’s also useful in experiments where you want to evaluate cross-task generalization — seeing how well a frozen head retains performance while another is adapted to new data.

Conclusion: Freezing one head supports task isolation and avoids catastrophic forgetting during multi-task learning.

### 🔄 Transfer Learning Strategy

In a scenario where training data is limited — for example, only a few hundred annotated examples for each task — transfer learning becomes essential to achieving strong performance without overfitting.

#### 1. Pre-trained Model Choice

I would choose `sentence-transformers/all-MiniLM-L6-v2` as the base model. It is lightweight, fast, and has been fine-tuned specifically for sentence-level semantic understanding using contrastive learning. This makes it ideal for generating embeddings that can generalize well across tasks like classification, quality assessment, and intent prediction.

#### 2. Freezing / Unfreezing Strategy

To balance generalization and task-specific adaptation, I would:

- **Freeze the MiniLM transformer backbone**
- **Train only the task-specific heads** (sentence classification, receipt quality, and query intent)

This strategy ensures that the model retains the rich semantic representations learned during large-scale pretraining, while allowing each task-specific head to learn mappings suited to the specific output classes.

If more data becomes available or higher performance is required, I would consider **gradually unfreezing** some transformer layers (starting from the top) to allow for deeper task-specific fine-tuning without destabilizing the lower-level language understanding.

#### 3. Rationale

This approach minimizes the risk of overfitting on limited data, reduces computational overhead during training, and leverages the strengths of transfer learning by building on an already capable language encoder. It aligns well with Fetch’s real-world requirements for scalable and maintainable ML systems.

### 🧠 Summary of Key Decisions & Insights (Task 3)

My overall strategy prioritizes reusing strong pretrained representations (MiniLM) while focusing learning on task-specific heads. Freezing the transformer backbone allows the model to retain language generalization while training the lightweight heads for sentence classification, receipt quality, and query intent. In a low-data scenario, this approach balances efficiency and performance. I also explored when it may be helpful to freeze specific heads to support continual learning or avoid degrading prior task performance. These freezing strategies reflect common practices in real-world ML systems, especially those used in production at scale, such as at Fetch.

## Task 4: Multi-Task Training Loop Implementation
To demonstrate how the model would be trained in a multi-task setting, I implemented a mock training loop using hypothetical data for each of the three tasks:

Sentence classification (Task A)

Receipt quality classification (Task B1)

Query intent classification (Task B2)

🔧 Assumptions

Each input sentence is associated with labels for all three tasks.

Labels are integer class IDs (e.g., 0 = "receipt", 2 = "offer").

The model is trained using CrossEntropyLoss for each head.

🧠 Design Choices

Loss Calculation:

I computed a separate loss for each task-specific head and combined them using a simple sum. In a real-world setup, I might use weighted loss if one task is more important or harder than the others.

Optimizer:

A single Adam optimizer is used to update all trainable parameters.

Freezing Strategy:

In a low-data scenario, I would freeze the transformer backbone and only train the heads.

🔄 Forward Pass

The model returns a dictionary with logits for all three heads:

- sentence_type

- receipt_quality

- query_intent

These are passed to individual CrossEntropyLoss functions based on the labels.

📊 Metrics

Although I didn’t compute metrics in code, I would track:

Accuracy and loss for each task

Per-task learning curves to monitor training dynamics

Optional: Macro F1 scores for imbalanced datasets

In production, I would also:

Use early stopping per task

Log gradients to ensure no head dominates the loss

Consider using task-specific learning rates or schedulers

🧠 Summary of Key Decisions & Insights (Task 4)

My training setup demonstrates a clean and modular approach to multi-task learning. I designed the model to return structured outputs per task, and used individual loss functions to ensure isolated feedback for each head. By summing losses and updating the shared encoder and heads together, I allow the model to learn both general sentence representations and task-specific objectives.

This mirrors the type of scalable, maintainable ML workflows expected in a real-world ML engineering role — particularly in a company like Fetch, where multi-headed models may serve different aspects of the platform (receipts, fraud, search, etc.).

## 🐳 Docker (Optional Bonus)

To build and run this project in a Docker container:

```bash
docker build -t fetch-ml-app .
docker run fetch-ml-app


