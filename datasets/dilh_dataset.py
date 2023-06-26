
import torch
import json
import torch.utils.data as data
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer,BertTokenizer
import os
import pickle as pkl

from utils.tools import get_age
from utils.distribute import is_rank_0
from graph import BaseGraph
from utils.tokenizer import Tokenizer

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'

class DILHDataset(data.Dataset):
    def __init__(self, filename,opt,graph:BaseGraph):
        super(DILHDataset, self).__init__()
        self.data_dir = filename
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id
        self.batch_size = opt.batch_size
        self.cache_path = opt.cache_path

        self.bert_path = opt.bert_path
        self.return_logic = opt.return_logic

        self.data = []
        if '.embeds' in self.bert_path:
            # word2vec版本
            self.tokenizer = Tokenizer.from_pretrained(self.bert_path)
            if opt.use_pretrain_embed_weight:
                opt.embedding_dim = self.tokenizer.embedding_dim()
        else:
            # 预训练模型版本
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            except:
                self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.max_length = opt.max_length
        self.graph = graph
        self._preprocess()

    def _preprocess(self):
        if is_rank_0():
            print("Loading data file...")

        with open(self.data_dir, 'r', encoding='UTF-8')as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def _get_text(self,dic):
        if '主诉' in dic:
            if '主诉' not in dic["主诉"]:
                dic['主诉'] = '主诉：'+dic['主诉']
            if '现病史' not in dic["现病史"]:
                dic['现病史'] = '现病史：'+dic['现病史']
            if '既往史' not in dic["既往史"]:
                dic['既往史'] = '既往史：'+dic['既往史']
            chief_complaint = '性别：' + dic['性别'] + '；年龄：'+ dic['年龄'] + '；'+ dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]
            raw_doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history
        else:
            raw_doc = dic['doc'].replace('[CLS]','').replace('[SEP]','')
        return raw_doc

    def _get_graph(self,raw_doc,dic):
        if 'id' in dic:
            _cache_file_name = f"{dic['id']}-{self.graph.max_hop}-{int(self.return_logic)}.pkl"
        else:
            _cache_file_name = f"{dic['emr_id']}-{self.graph.max_hop}-{int(self.return_logic)}.pkl"

        _cache_path = os.path.join(self.cache_path,_cache_file_name)
        if False:
            with open(_cache_path,'rb') as f:
                data = pkl.load(f)
            return data['nodes'],data['edges'],data['node_types'],data['edge_types'],data['edge_prompts']
        else:
            if 'ents' in dic:
                ents = [e.split('\t')[0] for e in dic['ents']]
                nodes, edges, node_types, edge_types, edge_prompts = self.graph.generate_graph_by_text(raw_doc,ents,self.return_logic)
            else:
                nodes, edges, node_types, edge_types, edge_prompts = self.graph.generate_graph_by_text(raw_doc,self.return_logic)
            # data = {}
            # data['nodes'] = nodes
            # data['edges'] = edges
            # data['node_types'] = node_types
            # data['edge_types'] = edge_types
            # data['edge_prompts'] = edge_prompts
            # with open(_cache_path,'wb') as f:
            #     pkl.dump(data,f)
            return nodes, edges, node_types, edge_types, edge_prompts

    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        
        nodes, edges, node_types, edge_types, edge_prompts = self._get_graph(raw_doc,dic)
        edges,edge_types,edge_prompts = self.graph.filter_edges(edges,edge_types,edge_prompts)

        prompts = {'raw_doc':raw_doc,'edge_prompt':edge_prompts}
        label_key = 'label' if 'label' in dic else 'labels'
        label = np.array([self.label_smooth_lambda \
                          if label not in dic[label_key] else 1-self.label_smooth_lambda \
                          for label in self.label2id])
        
        doc_idxes, doc_mask = doc['input_ids'],doc['attention_mask']

        new_nodes = []
        node_mask = []
        for node in nodes:
            if isinstance(node,int):# 图谱知识
                new_nodes.append([node])
                node_mask.append([1])
            else: # 逻辑知识
                new_nodes.append(list(node))
                node_mask.append([1 for _ in range(len(node))])

        edges = torch.tensor(edges, dtype=torch.long)
        node_types = torch.tensor(node_types, dtype=torch.long) # [node_num]
        edge_types = torch.tensor(edge_types, dtype=torch.long) # [edge_num]

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return doc_idxes, doc_mask, new_nodes,node_mask,node_types, edges,edge_types, prompts, label

    def collate_fn(self, X):
        X = list(zip(*X))
        doc_idxes, doc_mask,nodes,node_mask,node_types, edges,edge_types,prompts,labels= X 
        
        nodes = list(nodes)
        node_mask =list(node_mask)
        # nodes : [batch,num_node,node_len]
        for i in range(len(nodes)):
            max_len = max([len(n) for n in nodes[i]])
            for j in range(len(nodes[i])):
                nodes[i][j].extend([0 for _ in range(max_len-len(nodes[i][j]))])
                node_mask[i][j].extend([0 for _ in range(max_len-len(node_mask[i][j]))])
            nodes[i] = torch.LongTensor(nodes[i])
            node_mask[i] = torch.LongTensor(node_mask[i])

        idxs = [doc_idxes]
        masks = [doc_mask]
        for j,(idx,mask) in enumerate(zip(idxs,masks)):
            max_len = max([len(t) for t in idx])
            for i in range(len(idx)):
                idx[i].extend([0 for _ in range(max_len - len(idx[i]))])  # pad
                mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
            idxs[j] = torch.tensor(idx,dtype = torch.long)
            masks[j] = torch.tensor(mask,dtype = torch.long)
        labels = torch.cat(labels, 0)
        # nodes,node_mask, graph
        data = {'input_ids':idxs[0], 
                'attention_mask':masks[0],
                'nodes':nodes,
                'node_mask':node_mask,
                'nodes_type':node_types,
                'edges':edges,
                'edges_type':edge_types,
                'prompts':prompts,
                'label':labels}
        l1,l2 = torch.where(labels>0.5)
        label_dict = {}
        id2label = [label for label in self.label2id]
        for idx,d in zip(l1,l2):
            if int(idx) not in label_dict:
                label_dict[int(idx)] = []
            label_dict[int(idx)].append(id2label[d])
        
        return data