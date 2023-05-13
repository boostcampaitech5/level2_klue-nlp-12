import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    AutoConfig,
    AutoModel,
)

from loss import *


class REBiGRUModel(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModel.from_pretrained(config.model['name'],
                                             config=self.model_config)
        self.plm.resize_token_embeddings(vocab_size)

        self.hidden_size = self.model_config.hidden_size  # 1024 for roberta-large
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True,  # (bsz, seq, feature) if True else (seq, bsz, feature)
                          bidirectional=True)
        self.classifier = nn.Linear(self.hidden_size * 2, config.num_labels)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask, labels=None):
        # (bsz, max_seq_len, hidden_size)
        outputs = self.plm(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask).last_hidden_state
        hidde