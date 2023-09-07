from .base import BaseGraph
from .base_match import TrieTreeMatch
import pickle as pkl
from tqdm import tqdm

class MIMICMatch(TrieTreeMatch):
    def __init__(self, entity_file_path):
        super().__init__(entity_file_path)
    
    def _remove_neg_entities(self,entities:list,sentence:str) -> list:
        """
        去除否定实体,mimic无需这个操作
        """
        return entities

class MIMICGraph(BaseGraph):
    
    def __init__(self,kg_graph_path, logic_graph_path, entity_path, emr2kg_path, label_path, max_hop=3, logic_node_num = 8) -> None:
        """
        初始化一些数据
        Args:
            graph_path:str
            内部格式如下
            Rotigotine..drug	contraindication	hypertensive disorder..disease
            Rotigotine..drug	contraindication	hypertension..disease
            Fosinopril..drug	indication	hypertensive disorder..disease
            Fosinopril..drug	indication	hypertension..disease
            Estradiol valerate..drug	contraindication	hypertensive 

            entity_path:str
            内部格式如下
            beclabuvir..drug
            simeprevir..drug
            vaniprevir..drug
            asunaprevir..drug
            telaprevir..drug
            boceprevir..drug

            emr2kg_path: {emr_entity:kg_entity,...} dict 实体链接结果

            label_path:str
            malignant hypertension
            vein
            congestive heart failure
            atrial fibrillation (disease)

        Return:
            None
        """
        # super().__init__(graph_path, entity_path, emr2kg_path, label_path, max_hop)

        # h2t = {} # headid:[tail_id1,tail_id2,..]
        # t2h = {} # 
        # ht2rel = {} # {(h,t):relation(str),...}
        # rel2id = {} # {'伴随':0,...}

        self.match_model = MIMICMatch(entity_path)
        self.max_hop = max_hop
        self.logic_node_num = logic_node_num
        with open(emr2kg_path,'rb') as f:
            self.emr2kg = pkl.load(f)
        self.entityname2type = self.match_model.entityname2type
        self.id2entity = self.match_model.id2entity
        self.h2t_prune_cache_path = kg_graph_path.replace('triples.txt','h2t_prune.pkl')
        self.split_sign = {'simple':' ','reverse':' reverse ',
                           'diLsy_logic':[' accompany ',' is ',' clinical manifestations.'],
                           'diZsy_logic':[' accompany ',' is ','diagnostic basis.'],
                           'logic_reverse':[' has symptoms of ',' accompany ']}
        self.emr_name = 'Electronic medical record '
        self.emr_co_name = ' appear '
        self.yes_token ='yes'
        self.no_token = 'no'
        self.reverse_word = ' reverse '
        """
        1. 知识图谱
        """
        self._read_kg_file(kg_graph_path)

        """
        2. 逻辑图
        """
        self._read_logic_file(logic_graph_path)

        """
        分类节点
        """
        self._read_label_file(label_path)

        self._pruning_h2t()
        print('实体数量:',self.match_model.entity_size)
        print('二元关系数量:',len(self.ht2rel))
        print('逻辑超边数量:',self.logic_num)
        print('二元关系平均节点度:',len(self.ht2rel)/self.match_model.entity_size)
        print('超边中最大节点数:',self.max_logic_num)

    def generate_entity_name_by_text(self,sentence,ner_entities=None):

        origin_entity_names = self.match_model.find_entities(sentence)
        if ner_entities is not None:
            for entity_name in ner_entities:
                if entity_name not in self.emr2kg:
                    continue
                entity_name = self.emr2kg[entity_name]
                entity_name = self.match_model._process_entity_name(entity_name)
                origin_entity_names.append(entity_name)
        return origin_entity_names

def function(nodes,edges,nodes_type,edges_type,edges_prompt,graph):
    node_set = set()
    kg_node_num = 0
    for node in nodes:
        if type(node) is int:
            node_set.add(node)
            kg_node_num += 1
        else:
            for _node in node:
                node_set.add(_node)

    logic_type_id = 2 * len(graph.rel2id) + 2
    logic_count = edges_type.count(logic_type_id)

    _max_logic_num = 0
    for node in nodes:
        if type(node) is int:
            continue
        _max_logic_num = max(_max_logic_num,len(node))

    nodes_num = len(node_set)
    edges_num = len(edges_prompt)//2-logic_count
    logic_edges_num = logic_count
    ave_degree = edges_num / kg_node_num
    max_logic_num = _max_logic_num
    return nodes_num,edges_num,logic_edges_num,ave_degree,max_logic_num

import json
if __name__ == '__main__':
    kg_graph_path = '/home/lixin/work/diagnoisev3/data/mimic-3/triples.txt'
    logic_graph_path = '/home/lixin/work/diagnoisev3/data/mimic-3/anss.txt'
    entity_path = '/home/lixin/work/diagnoisev3/data/mimic-3/entities.txt'
    emr2kg_path = '/home/lixin/work/diagnoisev3/data/mimic-3/emr2kg.pkl'
    label_path = '/home/lixin/work/diagnoisev3/data/mimic-3/label_name.txt'
    train_path = '/home/lixin/work/diagnoisev3/data/mimic-3/train.json'
    dev_path = '/home/lixin/work/diagnoisev3/data/mimic-3/dev.json'
    test_path = '/home/lixin/work/diagnoisev3/data/mimic-3/test.json'
    label_path2 = '/home/lixin/work/diagnoisev3/data/mimic-3/label2id.txt'
    graph = MIMICGraph(
        kg_graph_path=kg_graph_path,
        logic_graph_path=logic_graph_path,
        entity_path=entity_path,
        emr2kg_path=emr2kg_path,
        label_path=label_path,
        logic_node_num=8,
        max_hop=4
    )
    labels = []
    with open(label_path2,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)

    nodes_num = []
    edges_num = []
    logic_edges_num = []
    ave_degree = []
    max_logic_num = []
    ave_path = []
    ave_acc = []
    ave_path_num =[]

    with open(test_path,'r',encoding='utf-8') as f:
        test = json.load(f)
    
    for item in tqdm(test):
        ents = []
        for ent in item['ents']:
            ents.append(ent.split('\t')[0])
        
        labels_id = [graph.label_nodes[labels.index(label)] for label in item['labels']]
        nodes,edges,nodes_type,edges_type,edges_prompt,_ave_path,_ave_acc,path_num = graph.generate_graph_by_text(item['doc'],ner_entities=ents,labels_id=labels_id)
        
        _nodes_num,_edge_num,_logic_edges_num,_ave_degree,_max_logic_num = function(nodes,edges,nodes_type,edges_type,edges_prompt,graph)
        nodes_num.append(_nodes_num)
        edges_num.append(_edge_num)
        logic_edges_num.append(_logic_edges_num)
        ave_degree.append(_ave_degree)
        max_logic_num.append(_max_logic_num)
        ave_path.append(_ave_path)
        ave_acc.append(_ave_acc)
        ave_path_num.append(path_num)
        print(len(nodes_num))
        if len(nodes_num) % 5 == 0:
            print(sum(nodes_num)/len(nodes_num))
            print(sum(edges_num)/len(edges_num))
            print(sum(logic_edges_num)/len(logic_edges_num))
            print(sum(ave_degree)/len(ave_degree))
            print(sum(max_logic_num)/len(max_logic_num))
            print(sum(ave_path)/len(ave_path))
            print(sum(ave_acc)/len(ave_acc))
            print(sum(ave_path_num)/len(ave_path_num))


    # with open(dev_path,'r',encoding='utf-8') as f:
    #     test = json.load(f)
    # for item in tqdm(test):
    #     ents = []
    #     for ent in item['ents']:
    #         ents.append(ent.split('\t')[0])
    #     labels_id = [graph.label_nodes[labels.index(label)] for label in item['labels']]
    #     nodes,edges,nodes_type,edges_type,edges_prompt,_ave_path,_ave_acc,path_num = graph.generate_graph_by_text(item['doc'],ner_entities=ents,labels_id=labels_id)
        
    #     _nodes_num,_edge_num,_logic_edges_num,_ave_degree,_max_logic_num = function(nodes,edges,nodes_type,edges_type,edges_prompt,graph)
    #     nodes_num.append(_nodes_num)
    #     edges_num.append(_edge_num)
    #     logic_edges_num.append(_logic_edges_num)
    #     ave_degree.append(_ave_degree)
    #     max_logic_num.append(_max_logic_num)
    #     ave_path.append(_ave_path)
    #     ave_acc.append(_ave_acc)
    #     ave_path_num.append(path_num)

    # with open(train_path,'r',encoding='utf-8') as f:
    #     test = json.load(f)
    # for item in tqdm(test):
    #     ents = []
    #     for ent in item['ents']:
    #         ents.append(ent.split('\t')[0])
    #     labels_id = [graph.label_nodes[labels.index(label)] for label in item['labels']]
    #     nodes,edges,nodes_type,edges_type,edges_prompt,_ave_path,_ave_acc,path_num = graph.generate_graph_by_text(item['doc'],ner_entities=ents,labels_id=labels_id)
        
    #     _nodes_num,_edge_num,_logic_edges_num,_ave_degree,_max_logic_num = function(nodes,edges,nodes_type,edges_type,edges_prompt,graph)
    #     nodes_num.append(_nodes_num)
    #     edges_num.append(_edge_num)
    #     logic_edges_num.append(_logic_edges_num)
    #     ave_degree.append(_ave_degree)
    #     max_logic_num.append(_max_logic_num)
    #     ave_path.append(_ave_path)
    #     ave_acc.append(_ave_acc)
    #     ave_path_num.append(path_num)

    print(sum(nodes_num)/len(nodes_num))
    print(sum(edges_num)/len(edges_num))
    print(sum(logic_edges_num)/len(logic_edges_num))
    print(sum(ave_degree)/len(ave_degree))
    print(sum(max_logic_num)/len(max_logic_num))
    print(sum(ave_path)/len(ave_path))
    print(sum(ave_acc)/len(ave_acc))
    print(sum(ave_path_num)/len(ave_path_num))