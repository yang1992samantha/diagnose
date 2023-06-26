
import torch
import json
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl
from datetime import datetime
from transformers import AutoTokenizer
import json

from utils.tools import get_age
from utils.distribute import is_rank_0
from graph import BaseGraph
from utils.tokenizer import Tokenizer
from tqdm import tqdm


PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'

class MSLANDataset(data.Dataset):
    def __init__(self, filename,opt,graph:BaseGraph):
        super(MSLANDataset, self).__init__()
        self.data_dir = filename
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id
        self.batch_size = opt.batch_size
        self.threshold = 100

        self.bert_path = opt.bert_path
        self.graph = graph
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

        if is_rank_0():
            print("Loading data file...")

        with open(self.data_dir, 'r', encoding='UTF-8') as f:
            self.data = json.load(f)

        cache_graph_file = filename.replace('train.json','A.pkl').replace('dev.json','A.pkl').replace('test.json','A.pkl')
        self.entity2id = self.graph.match_model.entity2id
        if 'train' in filename and not os.path.exists(cache_graph_file):    
        # if 'train' in filename:
            self.A = self._A_graph(opt)
            with open(cache_graph_file,'wb') as f:
                pkl.dump(self.A, f)
        else:
            with open(cache_graph_file,'rb') as f:
                self.A = pkl.load(f)

    def __len__(self):
        return len(self.data)
    
    def _A_graph(self,opt):
        all_num = opt.class_num + opt.entity_num
        A = {}
        print('生成A矩阵')
        for item in tqdm(self.data):
            doc = self._get_text(item)
            entity_names = self.graph.generate_entity_name_by_text(doc)
            label_key = 'label' if 'label' in item else 'labels'
            label_names = item[label_key]
            # 标签-实体
            for label_name in label_names:
                if label_name not in self.label2id:
                    continue
                for entity_name in entity_names:
                    if entity_name not in self.entity2id:
                        continue
                    label_id = self.label2id[label_name]
                    entity_id = self.entity2id[entity_name] + opt.class_num
                    key = (label_id,entity_id)
                    if key not in A:
                        A[key] = 0
                    A[key] += 1

            # 实体-实体
            for entity_name1 in entity_names:
                if entity_name1 not in self.entity2id:
                    continue
                for entity_name2 in entity_names:
                    if entity_name2 not in self.entity2id:
                        continue
                    # if entity_name1 == entity_name2:
                    #     continue
                    
                    entity_id1 = self.entity2id[entity_name1] + opt.class_num
                    entity_id2 = self.entity2id[entity_name2] + opt.class_num
                    key = (entity_id1,entity_id2)
                    if key not in A:
                        A[key] = 0
                    A[key] += 1
        return A

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

    def __getitem__(self, idx):
        dic = self.data[idx]        
        raw_doc = self._get_text(dic)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        if 'ents' in dic:
            ents = [e.split('\t')[0] for e in dic['ents']]
            nodes,edges = self.graph.generate_graph_by_co_occurrence(self.A, raw_doc,ner_entities=ents,threshold=self.threshold)
        else:
            nodes,edges = self.graph.generate_graph_by_co_occurrence(self.A, raw_doc,threshold=self.threshold)
        
        if 'labels' in dic:
            label = np.array([self.label_smooth_lambda \
                            if label not in dic['labels'] else 1-self.label_smooth_lambda \
                              for label in self.label2id])
        else:
            label = np.array([self.label_smooth_lambda \
                            if label not in dic['label'] else 1-self.label_smooth_lambda \
                              for label in self.label2id])
        doc_idxes, doc_mask = doc['input_ids'],doc['attention_mask']
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return doc_idxes, doc_mask,nodes,edges, label

    def collate_fn(self, X):
        X = list(zip(*X))
        doc_idxes, doc_mask,nodes,edges, labels = X 
        
        nodes = list(nodes)
        edges = list(edges)
        # nodes : [batch,num_node,node_len]
        for i in range(len(nodes)):
            nodes[i] = torch.LongTensor(nodes[i])
            edges[i] = torch.LongTensor(edges[i])

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
                'edges':edges,
                'label':labels}

        return data

