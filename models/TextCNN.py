from abc import ABC

import torch
import torch.nn as nn
from modules import EmbeddingLayer,PLMTokenEncoder

import numpy as np

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()

        self.class_num = opt.class_num
        if '.embeds' in opt.bert_path:
            self.word_embedding = EmbeddingLayer(opt)
        else:
            self.base_layer = PLMTokenEncoder(opt)

        self.embedding_dim = opt.embedding_dim
        self.class_num = opt.class_num
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.filter_sizes = (3, 4, 5)
        self.num_filters = 256
        self.convs = nn.ModuleList(
                [nn.Conv1d(self.embedding_dim, self.num_filters, k) for k in self.filter_sizes])

        len_filter_sizes = len(self.filter_sizes)
        num_filters = self.num_filters

        self.cls_layer1 = nn.Linear(len_filter_sizes * num_filters, 512)
        nn.init.xavier_normal_(self.cls_layer1.weight)
        self.cls_layer2 = nn.Linear(512, opt.class_num)
        nn.init.xavier_normal_(self.cls_layer2.weight)


    def maxpool(self,X):
        max_x,_ = X.max(dim = 2) # [batch_size, 256]
        return max_x

    def forward(self, sentence,mask):
        if hasattr(self,'word_embedding'):
            embed = self.word_embedding(sentence).transpose(1,2)
        else:
            embed = self.base_layer(sentence,mask).transpose(1,2)
        cnn_out = torch.cat([self.maxpool(conv(embed)) for conv in self.convs], 1)
        hidden_vec = torch.relu(self.cls_layer1(cnn_out))
        y_hat = self.cls_layer2(hidden_vec)
        return y_hat