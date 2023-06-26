import os
import json
import numpy as np
import datetime
from parsers import HOME_DIR

class Chinese42Config(object):

    def __init__(self, args):

        # self.bert_path = 'nghuyong/ernie-health-zh'
        self.bert_path = 'ValkyriaLenneth/longformer_zh'
        # self.bert_path = 'bert-base-chinese'
        # self.bert_path = 'IDEA-CCNL/Erlangshen-Longformer-110M'
        # self.bert_path = HOME_DIR + 'data/electronic-medical-record-42/embeds/128_0_10_cb_5n_5w.embeds'
        self.data_path = HOME_DIR + "data/electronic-medical-record-42"
        self.train_path = os.path.join(self.data_path, f"train{args.data_version}.json")
        self.dev_path = os.path.join(self.data_path, f"dev{args.data_version}.json")
        self.test_path = os.path.join(self.data_path, f"test{args.data_version}.json")
        # self.test_path = os.path.join(self.data_path, f"tiny.json")
        self.label_idx_path = os.path.join(self.data_path, "label2id.txt")
        self.pretrain_entity_path = os.path.join(self.data_path, "entity_embed.npz")
        self.cache_path = os.path.join(self.data_path,'cache')

        self.logic_graph_path = os.path.join(self.data_path, "anss.txt")
        self.kg_graph_path = os.path.join(self.data_path, "triplesv2.txt")
        self.entity_path = os.path.join(self.data_path, "entities.txt")
        self.label_hierarchy_path = os.path.join(self.data_path, "label_hierarchy.txt")

        self.max_length = 500
        self.label_smooth_lambda = 0.02
        self.bert_lr = 1e-5
        self.other_lr = 2e-4
        self.batch_size = 6
        self.logic_node_num = 2
        self.entity_embedding_dim= 100
        self.gnn_layer = 2
        self.num_hop = 2
        self.accumulation_steps = 1

        self.return_logic = True
        self.sample_type = 'attention'

        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))
        # self.save_model_path = '/home/lixin/diagnose2/logs/checkpoints/DILHv2_seed_1_2023-06-14-10-23-04.pth'
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

        self.label2id = {}
        with open(self.label_idx_path, "r", encoding="UTF-8") as f:
            for line in f:
                lin = line.strip().split()
                self.label2id[lin[0]] = len(self.label2id)
        self.class_num = len(self.label2id)
        self.entity_num = 0
        labels = []
        parent_labels = []
        with open(self.label_hierarchy_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                edge = line.split('##')
                label_name = edge[0]
                parent_label_name = edge[1]
                labels.append(label_name)
                if parent_label_name not in parent_labels:
                    parent_labels.append(parent_label_name)
        all_label_num = len(labels) + len(parent_labels)
        self.all_label_num = all_label_num
    def __str__(self):
        ans = "====================Configuration====================\n"
        for key, value in self.__dict__.items():
            if key in ['label2id','entity2id']:
                continue
            ans += key + ":" + (value if type(value) == str else str(value)) + "\n"
        ans += "====================Configuration====================\n"

        return ans