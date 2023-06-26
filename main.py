# -*- coding: utf-8 -*-
from importlib import import_module

import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import copy
from parsers import get_args
import configs
import datasets
import trainers
from trainers import BaseTrainer
from utils.loss import getLossFunction

import transformers
from strategies import NaiveStrategy,DDPStrategy

import wandb

# os.environ["WANDB_MODE"] = "offline" # wandb离线
# os.environ["WANDB_MODE"] = "dryrun"
import random

from utils.distribute import is_rank_0
from graph import Chinese42Graph,MIMICGraph


if __name__ == '__main__':
    # 参数加载
    args = get_args()
    Config = eval('configs.' + args.config)
    Dataset = eval('datasets.' + args.dataset)
    Trainer:BaseTrainer = eval('trainers.' + args.trainer)
    opt = Config(args)

    # 分布式
    gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) 
    if gpus > 1:
        # 单机多卡
        strategy = DDPStrategy()
    else:
        strategy = NaiveStrategy()

    # wandb
    if is_rank_0() and opt.use_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="disease-project",
            # name='test',
            name=opt.model_name,
            # track hyperparameters and run metadata
            config=opt
        )
    if is_rank_0():
        print(opt)

    # 加载数据集
    if opt.dataset in ['DILHDataset','GMANDataset','MSLANDataset','KG2TextDataset']:
        if '42' in opt.kg_graph_path:
            graph = Chinese42Graph(kg_graph_path=opt.kg_graph_path,
                                   logic_graph_path=opt.logic_graph_path,
                                   entity_path=opt.entity_path,
                                   label_path=opt.label_idx_path,
                                   max_hop=opt.num_hop+1,
                                   logic_node_num=opt.logic_node_num)
        else:
            graph = MIMICGraph(kg_graph_path=opt.kg_graph_path,
                               logic_graph_path=opt.logic_graph_path,
                               entity_path=opt.entity_path,
                               emr2kg_path=opt.emr2kg_path,
                               label_path=opt.label_name_path,
                               max_hop=opt.num_hop+1,
                               logic_node_num=opt.logic_node_num)
        opt.entity_num = graph.entity_num()
        setattr(opt,'label_nodes',graph.label_nodes)
        train_dataset = Dataset(opt.train_path, opt, graph)
        dev_dataset = Dataset(opt.dev_path, opt, graph)
        test_dataset = Dataset(opt.test_path, opt, graph)
    else:
        train_dataset = Dataset(opt.train_path, opt)
        dev_dataset = Dataset(opt.dev_path, opt)
        test_dataset = Dataset(opt.test_path, opt)
    all_datasets = [train_dataset,dev_dataset,test_dataset]

    # 引入模型
    model = import_module('models.' + opt.model_name).Model(opt)
    if hasattr(model,'add_graph_to_self'):
        model.add_graph_to_self(graph)

    # 损失
    loss_fn = getLossFunction(name = opt.loss_fn)

    # 训练模型
    trainer = Trainer(model,opt,all_datasets,loss_fn,strategy)
    if not opt.test_only:
        trainer.fit()
    else:
        trainer.inference()
    
    # wandb
    if is_rank_0() and opt.use_wandb:
        # start a new wandb run to track this script
        wandb.finish()
    print("==============Finish==============")
