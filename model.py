import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

from loss import *


class REBaseModel(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModelForSequenceClassification.from_pretrained(config.model['name'],
                                                                      config=self.model_config)
        self.plm.resize_token_embeddings(vocab_size)

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
        logits = outputs['logits']
        return {
            'logits': logits,
        }
