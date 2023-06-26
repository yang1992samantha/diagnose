import torch
import json
import torch.utils.data as data
import numpy as np

from transformers import AutoTokenizer
from utils.tokenizer import Tokenizer
import json

from graph import BaseGraph

from utils.tools import get_age
from utils.distribute import is_rank_0

class KG2TextDataset(data.Dataset):
    def __init__(self, filename, opt,graph:BaseGraph):
        super(KG2TextDataset, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.batch_size = opt.batch_size
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
    
    def __getitem__(self, idx):
        dic = self.data[idx]
        raw_doc = self._get_text(dic)
        doc = self.tokenizer(raw_doc,max_length=self.max_length,truncation=True)
        doc_idxes, doc_mask = doc['input_ids'],doc['attention_mask']
        if 'ents' in dic:
            ents = [e.split('\t')[0] for e in dic['ents']]
            kg_text = self.graph.generate_kg_text_by_text(raw_doc,ents)
        else:
            kg_text = self.graph.generate_kg_text_by_text(raw_doc)
        
        kg_text_doc = self.tokenizer(kg_text,max_length=self.max_length,truncation=True)
        kg_doc_idxes, kg_doc_mask = kg_text_doc['input_ids'],kg_text_doc['attention_mask']
        
        if 'label' in dic:
            label = torch.tensor([self.label_smooth_lambda if label not in dic['label'] else 1-self.label_smooth_lambda \
                                    for label in self.label2id])
        else:
            label = torch.tensor([self.label_smooth_lambda if label not in dic['labels'] else 1-self.label_smooth_lambda \
                                    for label in self.label2id])
        label = label.unsqueeze(0)

        return doc_idxes, doc_mask,kg_doc_idxes,kg_doc_mask, label

    def collate_fn(self, X):
        X = list(zip(*X))
        doc_idxes, doc_mask,kg_doc_idxes,kg_doc_mask, labels = X

        idxs = [doc_idxes,kg_doc_idxes]
        masks = [doc_mask,kg_doc_mask]
        for j,(idx,mask) in enumerate(zip(idxs,masks)):
            max_len = max([len(t) for t in idx])
            for i in range(len(idx)):
                idx[i].extend([self.tokenizer.pad_token_id for _ in range(max_len - len(idx[i]))])  # pad
                mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
            idxs[j] = torch.tensor(idx,dtype = torch.long)
            masks[j] = torch.tensor(mask,dtype = torch.long)

        labels = torch.cat(labels, 0)
        data = {'input_ids':idxs[0],'attention_mask':masks[0],
                'kg_input_ids':idxs[1],'kg_attention_mask':masks[1],
                'label':labels}

        return data

