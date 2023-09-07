from typing import Optional, Tuple, Union

import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import math
import copy

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Size,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
# from torch_geometric.utils.sparse import set_sparse_value

class NavieGNN(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        in_channels_type: int,
        in_channels_edge: int,
        out_channels: int,
        out_channels_type: int,
        out_channels_score: int,
        out_channels_edge: int,
        heads: int = 2,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.in_channels_type = in_channels_type
        self.in_channels_edge = in_channels_edge

        self.out_channels = out_channels
        self.out_channels_type = out_channels_type
        self.out_channels_score = out_channels_score
        self.out_channels_edge = out_channels_edge

        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:

        self.lin = Linear(in_channels, heads * out_channels,
                                bias=False, weight_initializer='glorot')
        self.lin_type = Linear(in_channels_type, heads * out_channels_type,
                                bias=False, weight_initializer='glorot')
        self.lin_score = Linear(1, heads * out_channels_score,
                                bias=False, weight_initializer='glorot')
        self.lin_edge = Linear(in_channels_edge,heads * out_channels_edge,
                               bias=False, weight_initializer='glorot')
        
        self.Wq = Linear(heads*(out_channels + out_channels_type + out_channels_score), heads * out_channels,
                        bias=False, weight_initializer='glorot')
        self.Wk = Linear(heads*(out_channels + out_channels_type + out_channels_score + out_channels_edge),heads * out_channels,
                         bias=False, weight_initializer='glorot')

        self.Proj = Linear(heads*(out_channels + 2 * out_channels_type + 2 * out_channels_score + out_channels_edge),
                           heads * out_channels)


        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        self.lin.reset_parameters()
        self.lin_type.reset_parameters()
        self.lin_score.reset_parameters()
        self.lin_edge.reset_parameters()

        zeros(self.bias)

    def forward(self, x: Tensor,x_type: Tensor,x_score:Tensor, 
                edge_index: Tensor,edge_attr: Tensor = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels
        C_type = self.out_channels_type
        C_score = self.out_channels_score
        C_edge = self.out_channels_edge

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:

        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        num_nodes = x.shape[0]
        x = self.lin(x).view(-1, H, C) #[N,H,C]
        x_type = self.lin_type(x_type).view(-1, H, C_type)
        x_score = self.lin_score(x_score).view(-1, H, C_score)

        edge_attr = self.lin_edge(edge_attr).view(-1,H,C_edge)

        alpha = self.edge_updater(edge_index, x=x, x_type=x_type,x_score=x_score, edge_attr=edge_attr, num_nodes=num_nodes)

        # alpha融合
        out = self.propagate(edge_index,alpha=alpha,x=x,x_type=x_type,x_score=x_score,edge_attr=edge_attr)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            # alpha = alpha.mean(dim = 1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out,(edge_index,alpha)
        else:
            return out

    def edge_update(self, x_i,x_j,x_type_i,x_type_j,x_score_i,x_score_j,edge_attr,num_nodes,index) -> Tensor:
        """
        这个函数和edge_updater对应，把点的信息 node[N,H] -> node_i(目标点),node_j(源点) [E,H] ,不加i,j的参数保持不变。
        """
        E,H,D = x_i.shape
        source_features = torch.cat((x_j,x_type_j,x_score_j),dim = -1).view(E,-1)
        target_features = torch.cat((x_i,x_type_i,x_score_i,edge_attr),dim = -1).view(E,-1)
       
        source_features = self.Wq(source_features).view(E,H,-1)
        target_features = self.Wk(target_features).view(E,H,-1)

        alpha = (source_features * target_features).sum(dim = -1) / math.sqrt(D + 1e-6)

        alpha = softmax(alpha,index=index,num_nodes=num_nodes)

        return alpha

    def message(self,alpha,x_j,x_type_i,x_type_j,x_score_i,x_score_j,edge_attr,) -> Tensor:
        """
        这个函数和propagate对应，msg融合注意力
        """
        E,H = alpha.shape
        raw_msg = torch.cat((x_j,x_type_i,x_score_i,x_type_j,x_score_j,edge_attr),dim = -1).view(E,-1)
        msg = torch.relu(self.Proj(raw_msg)).view(E,H,-1)
        return alpha.unsqueeze(-1) * msg

class DiagGNN(nn.Module):
    def __init__(self,k_layer,node_dim,type_dim,score_dim,edge_dim,n_edges_type,use_kge_loss=False):
        super(DiagGNN, self).__init__()
        
        self.k_layer = k_layer
        self.use_kge_loss = use_kge_loss
        self.edge_embed = nn.Embedding(n_edges_type,edge_dim)
        gnn = NavieGNN(in_channels=node_dim,
                       in_channels_type=type_dim,
                       in_channels_edge=edge_dim,
                       out_channels=node_dim,
                       out_channels_type=type_dim,
                       out_channels_edge=edge_dim,
                       out_channels_score=score_dim)
        
        self.gnns = nn.ModuleList(
            [copy.deepcopy(gnn) for _ in range(k_layer)])
        self.edge_score = nn.Parameter(torch.randn(n_edges_type,node_dim,node_dim))
        for i in range(n_edges_type):
            nn.init.xavier_uniform_(self.edge_score[i])

    def _nodes_score(self,nodes,edges,edges_type):
        # 计算node_score
        n_node = nodes.size(0)-1 # 去掉emr
        head_node = edges[0][::2][:n_node]
        edge_type = edges_type[::2][:n_node]
        tail_node = edges[1][::2][:n_node]
        head_node_embed = nodes[head_node].unsqueeze(2) # n_node, dim
        rel_linear = self.edge_score[edge_type] # n_node, dim, dim
        tail_node_embed = nodes[tail_node].unsqueeze(2) # n_node, dim
        head_score = (rel_linear @ head_node_embed).squeeze(2) # 头实体 n_node,dim
        # tail_score = (rel_linear @ tail_node_embed).squeeze(2) # 尾实体 n_node,dim
        tail_score = tail_node_embed.squeeze(2)
        # n_node 
        nodes_score = (head_score * tail_score).sum(dim = 1) / (1e-6+torch.norm(head_score,dim = 1)
                                                                    * torch.norm(tail_score,dim = 1))
        nodes_score = torch.cat((torch.ones(1,device=nodes_score.device),nodes_score),dim=0)
        nodes_score = nodes_score.unsqueeze(1) # [N,1]
        return nodes_score

    def _kge_loss(self,nodes,edges,edges_type):
        
        edge_num = int(edges.size(1))
        if edge_num > 100:
            choice_idx = random.sample(list(range(edge_num)),100)
        else:
            choice_idx = list(range(edge_num))

        choose_head = edges[0][choice_idx]
        choose_rel_linear = edges_type[choice_idx]
        choose_tail = edges[1][choice_idx]

        head_node_embed = nodes[choose_head].unsqueeze(2) # n_node, dim
        rel_linear = self.edge_score[choose_rel_linear] # n_node, dim, dim
        tail_node_embed = nodes[choose_tail].unsqueeze(2) # n_node, dim

        head_score = (rel_linear @ head_node_embed).squeeze(2) # 头实体 n_node,dim
        # tail_score = (rel_linear @ tail_node_embed).squeeze(2) # 尾实体 n_node,dim
        tail_score = tail_node_embed.squeeze(2)
        # n_node
        node_score = (head_score * tail_score).sum(dim = 1) / (1e-6+torch.norm(head_score,dim = 1)
                                                                * torch.norm(tail_score,dim = 1))
        # 反例   
        random_idx = [random.randint(0,int(nodes.size(0))-1) for _ in range(len(choice_idx))]
        tail_node_embed_neg = nodes[random_idx].unsqueeze(2)
        # tail_score_neg = (rel_linear @ tail_node_embed_neg).squeeze(2) # 尾实体 n_node,dim
        tail_score_neg = tail_node_embed_neg.squeeze(2) # 尾实体 n_node,dim
        # n_node
        node_score_neg = (head_score * tail_score_neg).sum(dim = 1) / (1e-6+torch.norm(head_score,dim = 1)
                                                                * torch.norm(tail_score_neg,dim = 1))
        score = node_score_neg - node_score
        loss = torch.exp(score).mean()
        return loss

    def forward(self,nodes,nodes_type,edges,edges_type):
        """
        进行卷积
        Args:
            nodes: [N,H]
            nodes_type: [N,H]
            edges: [2,E]
            edges_type: [E]
        return:
            nodes: [N,H]
            loss: None or tensor
            alpha: [E]
        """

        edge_attr = self.edge_embed(edges_type) # dim
        
        kge_loss = None
        for i,gnn in enumerate(self.gnns):
            nodes_score = self._nodes_score(nodes,edges,edges_type)
            if i < self.k_layer-1:
                pre_nodes = nodes.clone()
                nodes = torch.relu(nodes)
                nodes = gnn(nodes,nodes_type,nodes_score,edges,edge_attr)
                nodes += pre_nodes
            else:
                nodes = torch.relu(nodes)
                nodes,(_,alpha) = gnn(nodes,nodes_type,nodes_score,edges,edge_attr,return_attention_weights=True)
                # alpha : [E]
            if self.use_kge_loss and self.training:
                _kge_loss = self._kge_loss(nodes,edges,edges_type)
                if kge_loss is None:
                    kge_loss = _kge_loss
                else:
                    kge_loss += _kge_loss
        if kge_loss is not None:
            kge_loss /= self.k_layer
        
        if self.training:
            return nodes,alpha,nodes_score.squeeze(1),kge_loss
        else:
            return nodes,alpha,nodes_score.squeeze(1)


if __name__ == '__main__':
    
    embed_dim = 200
    type_embed_dim = 100

    edge_index = torch.randint(0,99,size=(2,5000)).cuda(1)
    x = torch.randn(300,embed_dim).cuda(1)
    x_type = torch.randn(300,type_embed_dim).cuda(1)
    x_score = torch.randn(300,1).cuda(1)
    # edge_attr = torch.randn(5000,768).cuda(1)
    edge_type = torch.randint(0,10,size =(5000,)).cuda(1)
    # gat = NavieGNN(in_channels=768,in_channels_type=768,
    #               out_channels=768,out_channels_type=100,
    #               out_channels_score=100,edge_dim=768,
    #               in_channels_edge=768,out_channels_edge=768,concat=False).cuda(1)
    gat = DiagGNN(k_layer=3,node_dim=embed_dim,type_dim=type_embed_dim,
                  score_dim=100,edge_dim=type_embed_dim,n_edges_type=50,
                  use_kge_loss=True).cuda(1)
    optimizer = torch.optim.AdamW(gat.parameters(),lr = 0.1,weight_decay=0.01)
    out = gat(x,x_type,edge_index,edge_type)
    loss = out[2]
    loss.backward()
    optimizer.step()
    out = gat(x,x_type,edge_index,edge_type)
    loss = out[0].sum()
    loss.backward()
    optimizer.step()
    out = gat(x,x_type,edge_index,edge_type)
    loss = out[0].sum()
    loss.backward()
    optimizer.step()
    out = gat(x,x_type,edge_index,edge_type)
    loss = out[0].sum()
    loss.backward()
    optimizer.step()
    print()