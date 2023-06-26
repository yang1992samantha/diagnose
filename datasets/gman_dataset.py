import torch
import json
import os
import torch.utils.data as data
import numpy as np

from transformers import AutoTokenizer
from utils.tokenizer import Tokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0

from graph import BaseGraph

class GMANDataset(data.Dataset):
    def __init__(self, filename, opt, graph:BaseGraph):
        super(GMANDataset, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.batch_size = opt.batch_size
        self.data = []
        if '.embeds' in self.bert_path:
            # word2vec 版本
            self.tokenizer = Tokenizer.from_pretrained(self.bert_path)
            if opt.use_pretrain_embed_weight:
                opt.embedding_dim = self.tokenizer.embedding_dim()
        else:
            # 预训练模型版本
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.max_length = opt.max_length
        self.graph = graph

        if is_rank_0():
            print("Loading data file...")

        with open(self.data_dir, 'r', encoding='UTF-8')as f:
            self.data = json.load(f)
        
        cache_graph_file = filename.replace('train.json','D_F_D_D.npz').replace('dev.json','D_F_D_D.npz').replace('test.json','D_F_D_D.npz')
        self.entity2id = self.graph.match_model.entity2id
        # if 'train' in filename and not os.path.exists(cache_graph_file):    
        if 'train' in filename:
            self.D_F = self._D_F_graph(opt)
            self.D_D = self._D_D_graph(opt)
            np.savez(cache_graph_file,D_F = self.D_F,D_D = self.D_D)
        else:
            self.D_F = np.load(cache_graph_file)['D_F']
            self.D_D = np.load(cache_graph_file)['D_D']

    def __len__(self):
        return len(self.data)

    def _D_D_graph(self,opt):
        labels = []
        parent_labels = []
        edges = []
        with open(opt.label_hierarchy_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                edge = line.split('##')
                label_name = edge[0]
                parent_label_name = edge[1]
                edges.append(edge)
                labels.append(label_name)
                if parent_label_name not in parent_labels:
                    parent_labels.append(parent_label_name)
        all_label_num = len(labels) + len(parent_labels)
        all_labels = labels + parent_labels
        D_D = np.zeros((all_label_num,all_label_num)).astype(np.float32)
        for edge in edges:
            D_D[all_labels.index(edge[0]),all_labels.index(edge[1])] += 1
            # D_D[all_labels.index(edge[1]),all_labels.index(edge[0])] += 1
        return D_D
    
    def _D_F_graph(self,opt):
        D_F = np.zeros((opt.all_label_num,opt.entity_num)).astype(np.float32)
        for item in self.data:
            doc = self._get_text(item)
            entity_names = self.graph.generate_entity_name_by_text(doc)
            label_key = 'label' if 'label' in item else 'labels'
            label_names = item[label_key]
            for label_name in label_names:
                if label_name not in self.label2id:
                    continue
                for entity_name in entity_names:
                    if entity_name not in self.entity2id:
                        continue
                    D_F[self.label2id[label_name],self.entity2id[entity_name]] += 1
        # 计算TF-IDF
        N = len(self.data)
        tf = D_F / (D_F.sum(axis = 1,keepdims=True) + 1e-6)
        idf = np.log( N / (1 + D_F.sum(axis = 1, keepdims=True) + 1e-6) )
        D_F = tf * idf
        D_F = D_F / (D_F.sum(axis = 1,keepdims=True) + 1e-6)
        return D_F

    def _get_text(self,dic):
        if '主诉' in dic:
            if '主诉' not in dic["主诉"]:
                dic['主诉'] = '主诉：'+dic['主诉']
            if '现病史' not in dic["现病史"]:
                dic['现病史'] = '现病史：'+dic['现病史']
            if '既往史' not in dic["既往史"]:
                dic['既往史'] = '既往史：'+dic['既往史']
            chief_complaint = '性别：' + dic['性别'] + '；年龄：'+dic['年龄'] + '；'+ dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]
            raw_doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history
        else:
            raw_doc = dic['doc'].replace('[CLS]','').replace('[SEP]','')
        return raw_doc
    
    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        if 'ents' in dic:
            ents = [e.split('\t')[0] for e in dic['ents']]
            entity_names = self.graph.generate_entity_name_by_text(raw_doc,ents)
        else:
            entity_names = self.graph.generate_entity_name_by_text(raw_doc)
        entity_ids = []
        for entity_name in entity_names:
            if entity_name not in self.entity2id:
                continue
            entity_ids.append(self.entity2id[entity_name])
        entity_mask = [1] * len(entity_ids)
        D_F = []
        for entity_id in entity_ids:
            D_F.append(self.D_F[:,entity_id])
        D_F = np.stack(D_F,axis = 1)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        doc_idxes, doc_mask = doc['input_ids'],doc['attention_mask']
        label_key = 'label' if 'label' in dic else 'labels'
        label = torch.tensor([self.label_smooth_lambda if label not in dic[label_key] else 1-self.label_smooth_lambda \
                                    for label in self.label2id])
        
        label = label.unsqueeze(0)

        return doc_idxes, doc_mask,entity_ids,entity_mask,D_F, label

    def collate_fn(self, X):
        X = list(zip(*X))
        doc_idxes, doc_mask,entity_ids,entity_mask,D_F, labels = X
        D_F = list(D_F)
        idxs = [doc_idxes,entity_ids]
        masks = [doc_mask,entity_mask]
        for j,(idx,mask) in enumerate(zip(idxs,masks)):
            max_len = max([len(t) for t in idx])
            for i in range(len(idx)):
                idx[i].extend([self.tokenizer.pad_token_id for _ in range(max_len - len(idx[i]))])  # pad
                mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
                if j == 1:
                    # D_num,F_num
                    D_num,F_num = D_F[i].shape
                    pad_arr = np.zeros((D_num,max_len - F_num))
                    D_F[i] = np.concatenate((D_F[i],pad_arr),axis = 1)
            idxs[j] = torch.tensor(idx,dtype = torch.long)
            masks[j] = torch.tensor(mask,dtype = torch.long)
        D_F = torch.tensor(np.array(D_F).astype(np.float32))

        labels = torch.cat(labels, 0)
        D_D = torch.tensor(self.D_D) # [label_num,entity_num]
        data = {'input_ids':idxs[0],'attention_mask':masks[0],'nodes':idxs[1],'node_mask':masks[1],'D_F':D_F,'D_D':D_D,'label':labels}

        return data

