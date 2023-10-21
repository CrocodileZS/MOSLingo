import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self, plm, num_classes):
        super().__init__()
        self.plm = BertModel.from_pretrained(plm)
        linear_input_size = self.plm.embeddings.position_embeddings.embedding_dim
        linear_output_size = num_classes
        self.linear = nn.Linear(in_features=linear_input_size, out_features=linear_output_size)

    def forward(self, input_ids, attention_mask):
        x = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        logits = self.linear(x)
        pred_prob = F.softmax(logits)
        return pred_prob
