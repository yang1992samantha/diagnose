from abc import ABC

import torch
import torch.nn as nn
from graph import BaseGraph

# from fengshen import LongformerModel

import numpy as np

import time
from modules import DiagGNN
from modules import PLMWithMaskEncoder
from modules import EntityEmbeddingLayer
from transformers import AutoTokenizer,BertTokenizer
from graph import BaseGraph

class Model(nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'GraphModel'
        self.class_num = opt.class_num
        self.embedding_dim = opt.embedding_dim
        self.label_nodes = opt.label_nodes

        node_embed_dim = opt.entity_embedding_dim
        node_type_embed_dim = 50
        self.sample_type = opt.sample_type

        self.base_layer = PLMWithMaskEncoder(opt)
        # self.base_layer_textcnn = TextCNNEncoder(opt)
        self.entity_embed = EntityEmbeddingLayer(opt)

        self.entity_type_embed = nn.Embedding(100, node_type_embed_dim)
        self.logic_entity_mlp = nn.Sequential(
            nn.Linear(opt.embedding_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,opt.embedding_dim)
        )
        self.node_type_w = nn.Linear(opt.embedding_dim,opt.embedding_dim)

        self.dropout = nn.Dropout(opt.dropout_rate)

        self.node_proj = nn.Linear(opt.embedding_dim,node_embed_dim)

        # GCN融合标签共现关系
        # self.graph_embedding = nn.Embedding(self.graph_nodes_num, self.embedding_dim)
        self.gnn = DiagGNN(k_layer=opt.gnn_layer,node_dim=node_embed_dim,type_dim=node_type_embed_dim,
                           score_dim=50,edge_dim=node_type_embed_dim,n_edges_type=50,
                           use_kge_loss=True)

        self.cls_layer = nn.Linear(self.embedding_dim + 2 * node_embed_dim,self.embedding_dim)
        # self.gcn = nn.parameter.Parameter(torch.eye(self.class_num))
        self.cls_layer2 = nn.Linear(self.embedding_dim,self.class_num)

        ##  256 
        best_u = 256
        best_da = 256

        self.bi_lstm = nn.LSTM(opt.embedding_dim, best_u, 2,
                            bidirectional=True, batch_first=True)

        self.w = nn.Parameter(torch.FloatTensor(best_u * 2,best_da))
        nn.init.xavier_normal_(self.w)
        self.u = nn.Parameter(torch.FloatTensor(best_da,opt.class_num))
        nn.init.xavier_normal_(self.u)

        self.final1 = nn.Linear(best_u * 2, opt.embedding_dim)
        nn.init.xavier_normal_(self.final1.weight)

        self.logic_entity_mlp[0].apply(self.init_weights)
        self.logic_entity_mlp[2].apply(self.init_weights)
        self.node_type_w.apply(self.init_weights)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)
        except:
            self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        self.max_length = opt.max_length
    
    def add_graph_to_self(self,graph:BaseGraph):
        self.graph = graph

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self,sentence,
                     mask,
                     nodes,
                     node_mask,
                     node_types,
                     edges,
                     edge_types,
                     prompts,
                     labels=None):
        # [1 * (class_num+entity_num) * 768] * [batch_size, (class_num+entity_num),1]
        # -> [batch_size * (class_num+entity_num) * 768]
        text_vec = self.base_layer(sentence,mask) # [batch_size, hidden_size]
        ans = [] # 
        prompt_ans = []
        pool_ans = []
        mask_loss = None
        mask_cnt = 0.0
        kge_loss = None
        kge_cnt = 0.0

        for i,(_nodes,_node_mask,_node_types,_edges,_edge_types,_prompts) in enumerate(zip(
                                                nodes,node_mask,node_types,
                                                edges,edge_types,
                                                prompts)):
            if _nodes.shape[1] > 1:
                logic_node_idx = torch.where(_node_mask[:,1].bool())[0]
                kg_node_num = logic_node_idx[0]
            else:
                kg_node_num = _nodes.shape[0]
            
            emr_edge_num = _edges.size(1) - len(_prompts['edge_prompt'])
            node_embeds = self.entity_embed(_nodes) # [node_num,3]
            node_type_embeds = self.entity_type_embed(_node_types)
            logic_embeds= self.logic_entity_mlp(node_embeds[kg_node_num:])
            node_embeds = torch.cat((node_embeds[:kg_node_num],logic_embeds)) 
            node_embeds = node_embeds @ self.node_type_w.weight # 
            node_embeds = (node_embeds * _node_mask.unsqueeze(2)).sum(dim =1)
            
            plm_output = text_vec[i].unsqueeze(0)
            node_embeds = torch.cat((plm_output,node_embeds),dim = 0)

            node_embeds = self.node_proj(node_embeds)
            
            if self.training:
                gnn_nodes, alpha, node_score, _kge_loss = self.gnn(node_embeds,node_type_embeds,_edges,_edge_types)
                if _kge_loss is not None:
                    if kge_loss is None:
                        kge_loss = _kge_loss
                    else:
                        kge_loss += _kge_loss
                    kge_cnt += 1
            else:
                gnn_nodes, alpha,node_score =self.gnn(node_embeds,node_type_embeds,_edges,_edge_types)
            context_vec = gnn_nodes[0]
            label_vec = gnn_nodes[1:1+self.class_num]
            alpha = alpha.mean(dim = 1)

            prompt_ans.append(text_vec[i].unsqueeze(0).unsqueeze(0).repeat(1,self.class_num,1))
            
            ans.append(context_vec.unsqueeze(0).repeat(self.class_num,1))
            pool_ans.append(label_vec)

        ans = torch.stack(ans,dim = 0) # [batch_size,embed_dim]
        prompt_ans = torch.cat(prompt_ans,dim = 0) # [batch_size,embed_dim]
        pool_ans = torch.stack(pool_ans, dim = 0)
        pool = torch.cat((ans,pool_ans,prompt_ans),dim = 2) # [batch_size,class_num,embed_dim]
        pool = self.cls_layer(pool)
        ans = (self.cls_layer2.weight.unsqueeze(0) * pool).sum(dim = 2) + self.cls_layer2.bias.unsqueeze(0)
        
        if self.training:
            if (mask_loss is None or torch.isnan(mask_loss)) and kge_loss is None:
                return ans,None
            elif (mask_loss is None or torch.isnan(mask_loss)):
                return ans,kge_loss / kge_cnt # [batch_size, num_labels]
            elif kge_loss is None:
                return ans, 0.01 * mask_loss / mask_cnt
            else:
                return ans,kge_loss / kge_cnt + 0.01 * mask_loss / mask_cnt
        else:
            return ans
    