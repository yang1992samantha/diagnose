# -*- coding:utf-8 -*-
import sys

sys.path.append('/home/lixin/diagnose2/diagnoisev3')
from graph import MIMICGraph,Chinese42Graph
from transformers import BertModel
from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class GenerateEmbedding:
    def __init__(self,bert_path):
        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        if torch.cuda.is_available():
            self.bert = self.bert.cuda()
    def generate(self,entity):
        tokens = self.tokenizer(entity,return_tensors = 'pt')
        with torch.no_grad():
            if torch.cuda.is_available():
                tokens['input_ids'] = tokens['input_ids'].cuda()
                tokens['attention_mask'] = tokens['attention_mask'].cuda()
            output = self.bert(tokens['input_ids'],tokens['attention_mask']).last_hidden_state[0,0]
        output = output.cpu().numpy()
        output = (output - output.mean()) / (output.std() +1e-6)
        output[output > 5] = 5
        output[output < -5] = -5
        return output
    
    def similarity(self,vec1,vec2):
        """
            计算余弦相似度
        """
        vec1 = (vec1-np.mean(vec1)) / np.std(vec1)
        vec2 = (vec2-np.mean(vec2)) / np.std(vec2)
        return np.sum(vec1 * vec2) / (np.sqrt(np.sum(np.power(vec1,2))) + np.sqrt(np.sum(np.power(vec2,2))))

if __name__ == '__main__':
    DATA_DIR = '/home/lixin/diagnose2/data/'
    # kg_graph_path = DATA_DIR + 'mimic-3/triples.txt'
    # logic_graph_path = DATA_DIR + 'mimic-3/anss.txt'
    # entity_path = DATA_DIR + 'mimic-3/entities.txt'
    # emr2kg_path = DATA_DIR + 'mimic-3/emr2kg.pkl'
    # label_path = DATA_DIR + 'mimic-3/label_name.txt'
    # train_path = DATA_DIR + 'mimic-3/train.json'
    # dev_path = DATA_DIR + 'mimic-3/dev.json'
    # test_path = DATA_DIR + 'mimic-3/test.json'
    # save_path = DATA_DIR + 'mimic-3/entity_embed.npz'
    # bert_path = 'yikuan8/Clinical-Longformer'
    # graph = MIMICGraph(
    #     kg_graph_path=kg_graph_path,
    #     logic_graph_path=logic_graph_path,
    #     entity_path=entity_path,
    #     emr2kg_path=emr2kg_path,
    #     label_path=label_path,
    #     logic_node_num=8
    # )
    kg_graph_path = DATA_DIR + 'electronic-medical-record-42/triplesv2.txt'
    logic_graph_path = DATA_DIR + 'electronic-medical-record-42/anss.txt'
    entity_path = DATA_DIR + 'electronic-medical-record-42/entities.txt'
    emr2kg_path = DATA_DIR + 'electronic-medical-record-42/emr2kg.pkl'
    label_path = DATA_DIR + 'electronic-medical-record-42/label2id.txt'
    train_path = DATA_DIR + 'electronic-medical-record-42/train.json'
    dev_path = DATA_DIR + 'electronic-medical-record-42/dev.json'
    test_path = DATA_DIR + 'electronic-medical-record-42/test.json'
    bert_path = 'bert-base-chinese'
    save_path = DATA_DIR + 'electronic-medical-record-42/entity_embed.npz'
    graph = Chinese42Graph(
        kg_graph_path=kg_graph_path,
        logic_graph_path=logic_graph_path,
        entity_path=entity_path,
        label_path=label_path,
        logic_node_num=2
    )

    gm = GenerateEmbedding(bert_path)

    disease1 = gm.generate('骨折')
    disease2 = gm.generate('骨折')
    disease3 = gm.generate('88')
    sim1 = gm.similarity(disease1,disease2)
    sim2 = gm.similarity(disease1,disease3)
    print()
    entity2id = graph.match_model.entity2id
    entity_embed = []
    for entity in tqdm(entity2id):
        _entity_embed = gm.generate(entity)
        entity_embed.append(_entity_embed)

    print()
    entity_embed = np.array(entity_embed)
    np.savez(save_path,entity_embed=entity_embed)
