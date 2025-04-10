import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskSentenceModel(nn.Module):
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_classes_sentence=5,    # Sentence classification (Task A)
        num_classes_quality=3,     # Receipt quality (Task B1)
        num_classes_intent=4       # Search/query intent (Task B2)
    ):
        super(MultiTaskSentenceModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size  # 384 for MiniLM

        # Task A: Sentence classification
        self.classification_head = nn.Linear(self.hidden_size, num_classes_sentence)

        # Task B1: Receipt quality classification
        self.quality_head = nn.Linear(self.hidden_size, num_classes_quality)

        # Task B2: Search/query intent classification
        self.intent_head = nn.Linear(self.hidden_size, num_classes_intent)

    def mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / \
               torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(model_output.last_hidden_state, attention_mask)

        logits_sentence = self.classification_head(pooled_output)
        logits_quality = self.quality_head(pooled_output)
        logits_intent = self.intent_head(pooled_output)

        return {
            "sentence_type": logits_sentence,
            "receipt_quality": logits_quality,
            "query_intent": logits_intent
        }

