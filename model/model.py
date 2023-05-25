import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)

from model.loss import *


class BaseREModel(nn.Module):
    """Pre-trained Language Model로부터 나온 logits를 FC layer에 통과시키는 기본 분류기."""
    def __init__(self, config, new_num_tokens: int):
        """
        Args:
            config: 사용자 config.
            new_num_tokens: tokenizer의 길이. Additional special tokens 수를 포함.
        """
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModelForSequenceClassification.from_pretrained(config.model['name'],
                                                                      config=self.model_config)

        if self.model_config.vocab_size != new_num_tokens:
            self.plm.resize_token_embeddings(new_num_tokens)

    @autocast()
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.plm(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        logits = outputs['logits']
        return {
            'logits': logits,
        }

class CustomModel(nn.Module):
    def __init__(self, config, new_num_tokens: int):
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels
        
        self.plm = AutoModelForSequenceClassification.from_pretrained(config.model['name'],
                                                                      config=self.model_config)

        if self.model_config.vocab_size != new_num_tokens:
            self.plm.resize_token_embeddings(new_num_tokens)
        
        self.hidden_size = self.model_config.hidden_size
        
        self.entity_embedding = nn.Embedding(3, self.hidden_size)
        nn.init.xavier_normal_(self.entity_embedding.weight)

        self.weight = nn.Parameter(torch.Tensor(1))  # Learnable weight parameter
        nn.init.uniform_(self.weight)

        # self.reduction_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)

    @autocast()
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        entity_ids=None,
        labels=None,
    ):
        # entity_ids = entity_ids.long()
        entity_embeddings = self.entity_embedding(entity_ids)          # torch.tensor([64, 180, 1024])
        input_embeddings = self.plm.get_input_embeddings()(input_ids)  # torch.tensor([64, 180, 1024])

        # 단순히 더한 버전
        # combined_embeddings = input_embeddings + entity_embeddings

        # concat 버전
        # combined_embeddings = torch.cat([input_embeddings,entity_embeddings], dim=-1) # torch.tensor([64, 180, 2048])
        # combined_embeddings = self.reduction_layer(combined_embeddings) # torch.tensor([64, 180, 1024])

        # weighted sum 버전
        combined_embeddings = self.weight * input_embeddings + (1 - self.weight) * entity_embeddings

        outputs = self.plm.roberta(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.plm.classifier(outputs['last_hidden_state'])

        return {
            'logits': logits,
        }
    
class BiGRUREModel(nn.Module):
    """
    Pre-trained Language Model로부터 나온 logits를 Bi-driectional GRU에 통과시킨 후
    hidden states 정보를 FC layer에 통과시킨 분류기.
    """
    def __init__(self, config, new_num_tokens: int):
        """
        Args:
            config: 사용자 config.
            new_num_tokens: tokenizer의 길이. Additional special tokens 수를 포함.
        """
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModel.from_pretrained(config.model['name'],
                                             config=self.model_config)

        if self.model_config.vocab_size != new_num_tokens:
            self.plm.resize_token_embeddings(new_num_tokens)

        self.hidden_size = self.model_config.hidden_size  # 1024 for roberta-large
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True,  # (bsz, seq, feature) if True else (seq, bsz, feature)
                          bidirectional=True)
        self.init_gru()
        self.classifier = nn.Linear(self.hidden_size * 2, config.num_labels)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        self.classifier.bias.data.fill_(0)

    def init_gru(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    @autocast()
    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, labels=None):
        outputs = self.plm(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask).last_hidden_state
        _, next_hidden = self.gru(outputs)
        outputs = torch.cat([next_hidden[0], next_hidden[1]], dim=1)
        logits = self.classifier(outputs)
        return {
            'logits': logits,
        }


class BiLSTMREModel(nn.Module):
    """Pre-trained Language Model로부터 나온 logits를 Bi-driectional LSTM에 통과시킨 후
    hidden states 정보를 FC layer에 통과시킨 분류기.
    """
    def __init__(self, config, new_num_tokens: int):
        """
        Args:
            config: 사용자 config.
            new_num_tokens: tokenizer의 길이. Additional special tokens 수를 포함.
        """
        super().__init__()

        self.model_config = AutoConfig.from_pretrained(config.model['name'])
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModel.from_pretrained(config.model['name'],
                                             config=self.model_config)

        if self.model_config.vocab_size != new_num_tokens:
            self.plm.resize_token_embeddings(new_num_tokens)

        self.hidden_size = self.model_config.hidden_size  # 1024 for roberta-large
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,  # (bsz, seq, feature) if True else (seq, bsz, feature)
                            bidirectional=True)
        self.init_lstm()
        self.classifier = nn.Linear(self.hidden_size * 2, config.num_labels)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        self.classifier.bias.data.fill_(0)

    def init_lstm(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    @autocast()
    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, labels=None):
        outputs = self.plm(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask).last_hidden_state
        _, (next_hidden, _) = self.lstm(outputs)
        outputs = torch.cat([next_hidden[0], next_hidden[1]], dim=1)
        logits = self.classifier(outputs)
        return {
            'logits': logits,
        }
