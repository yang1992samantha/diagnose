from abc import ABC

import torch
import torch.nn as nn
from transformers import AutoModel

class PLMTokenEncoder(nn.Module, ABC):
    def __init__(self, opt):
        super(PLMTokenEncoder, self).__init__()
        self.model_name = 'AutoModel'
        self.word_embedding = AutoModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, X, masks):
        embed = self.word_embedding(X, attention_mask=masks).last_hidden_state
        pooled = self.dropout(embed)
        return pooled
