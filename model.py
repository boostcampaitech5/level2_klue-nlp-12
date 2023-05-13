import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)

from loss import *


class REBaseModel(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels
        
        self.plm = AutoModel.from_pretrained(config.model['name'],
                                             config=self.model_config)
        self.plm.resize_token_embeddings(vocab_size)
        self.classifier = nn.Linear(self.plm.config.hidden_size, config.num_labels)
        

    @autocast()
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        outputs = self.plm(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return {
            'logits': logits,
        }
