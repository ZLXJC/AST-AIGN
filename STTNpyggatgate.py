# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

#from One_hot_encoder import One_hot_encoder
import numpy as np
import gcnnetSTTNs as NEW_GCN
from torch_geometric.nn import GCN
from Basic_Gnn import GAT


class STTNSNet(nn.Module):
    def __init__(self, adj, in_channels, embed_size, time_num,
                 num_layers, T_dim, output_T_dim, heads, dropout, forward_expansion):


        self.num_layers = num_layers
        super(STTNSNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.transformer = Transformer(embed_size, heads, adj, time_num, dropout, forward_expansion)
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv3 = nn.Conv2d(embed_size, in_channels, 1)

    def forward(self, x,device):#x是输入的data batch 还没有输入device
       
        data=x["flow_x"].to(device)
        edge_index = x["edge_index"][0].to(device)
        edge_attr =  x["edge_attr"][0].to(device)
        data_conv = data.permute(0,3,1,2) 
        data_conv = self.conv1(data_conv) 
        data_trans = data_conv.permute(0,2, 3, 1)  
        data_trans = self.transformer(data_trans, data_trans, data_trans, self.num_layers,device,edge_index,edge_attr) 
        data_cov2 = data_trans.permute(0, 2, 1, 3)  
        data_cov2 = self.conv2(data_cov2) 
        data_cov3 = data_cov2.permute(0, 3, 2, 1)  
        data_cov3 = self.conv3(data_cov3)  
        data_final = data_cov3.permute(0,2,3,1)
        return data_final


class Transformer(nn.Module):
    def __init__(self,embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(Transformer, self).__init__()
        self.sttnblock = STTNSNetBlock(embed_size, heads, adj, time_num, dropout, forward_expansion)

    def forward(self, query, key, value,  num_layers,device,edge_index,edge_attr):
        q, k, v = query, key, value
        for i in range(num_layers):
            out = self.sttnblock(q, k, v,device,edge_index,edge_attr)
            q, k, v = out, out, out
        return out


class STTNSNetBlock(nn.Module):
    def __init__(self,embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(STTNSNetBlock, self).__init__()
        self.SpatialTansformer = STransformer(embed_size, heads, adj, dropout, forward_expansion)
        self.TemporalTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.liner_first = nn.Linear(embed_size,embed_size)
        self.liner_second = nn.Linear(embed_size,embed_size)

    def forward(self, query, key, value,device,edge_index,edge_attr):
      
        # out1 = self.norm1(self.SpatialTansformer(query, key, value,device,edge_index,edge_attr) + query)
       
        # out2 = self.dropout(self.norm2(self.TemporalTransformer(out1, out1, out1,device) + out1))
        out1 = self.liner_first(self.norm1(self.SpatialTansformer(query, key, value,device,edge_index,edge_attr) + query))
       
        out2 = self.liner_second(self.norm2(self.TemporalTransformer(query, key, value,device)+query))

        GateT = torch.sigmoid(out1)
        GateS = torch.sigmoid(out2)
        out3 = GateT * out1 + GateS *out2
        

        # return out2
        return out3

import torch.nn.functional as F


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(STransformer, self).__init__()
        self.adj_matrix = adj
        self.D_S = adj
        self.D_S = nn.Parameter(adj)

   

        self.embed_linear = nn.Linear(adj.shape[0], embed_size)
        self.attention = SSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # self.gcn = NEW_GCN.GCN(embed_size, embed_size * 2, embed_size)
        # self.gcn = GCN(in_channels=embed_size,hidden_channels=embed_size * 2,num_layers=2,out_channels=embed_size,add_self_loops=False)
        self.gcn = GAT(in_channels=embed_size, hidden_channels=embed_size * 2, out_channels=embed_size, num_layers=2,jk='max',add_self_loops=False)
        self.norm_adj = nn.InstanceNorm2d(1)

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value,device,edge_index,edge_attr):
        edge_index = edge_index.long()
     
        B= query.shape[0]
        N= query.shape[1]
        T= query.shape[2]
        C= query.shape[3]

        D_S = self.embed_linear((self.D_S.to(device)))
        D_S = D_S.expand(T, N, C)
        D_S = D_S.permute(1, 0, 2)
        D_S=D_S.unsqueeze(0)
      
        # GCN 部分
        X_G = torch.Tensor(query.shape[0],query.shape[1], 0, query.shape[3])
        for t in range(query.shape[2]):
            # o = self.gcn(query[:,:, t, :], self.adj_matrix.to(device))
            o = self.gcn(query[:, :, t, :],edge_index).unsqueeze(2)
            X_G=X_G.to(device)
            X_G = torch.cat((o, X_G), dim=2)
        
        query = query + D_S
        
        value = value + D_S
        key = key + D_S
        attn = self.attention(value, key, query)  
        M_s = self.dropout(self.norm1(attn + query))
        feedforward = self.feed_forward(M_s)
        U_s = self.dropout(self.norm2(feedforward + M_s))
        # 融合
        g = torch.sigmoid(self.fs(U_s) + self.fg(X_G))
        out = g * U_s + (1 - g) * X_G
        return out



class SSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.values = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.queries = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.keys = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        B,N, T, C = query.shape
        query = query.reshape(B,N, T, self.heads, self.per_dim)
        keys = keys.reshape(B,N, T, self.heads, self.per_dim)
        values = values.reshape(B,N, T, self.heads, self.per_dim)
      
        queries = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)
       
        attn = torch.einsum("bqthd, bkthd->bqkth", (queries, keys)) 
        attention = torch.softmax(attn / (self.embed_size ** (1 / 2)), dim=1)
        out = torch.einsum("bqkth,bkthd->bqthd", (attention, values))
        out = out.reshape(B,N, T, self.heads * self.per_dim) 
        out = self.fc(out)

        return out


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
       
        self.time_num = time_num
        
        self.temporal_embedding = nn.Embedding(time_num, embed_size) 

        self.attention = TSelfattention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value,device):
       
        B, N, T, C = query.shape

       
      
        D_T = self.temporal_embedding(torch.arange(0, T).to(device)) 
       
        D_T = D_T.expand(N, T, C)
        D_T = D_T.unsqueeze(0)
       
        x = D_T + query
        attention = self.attention(x, x, x)
        M_t = self.dropout(self.norm1(attention + x))
        feedforward = self.feed_forward(M_t)
        U_t = self.dropout(self.norm2(M_t + feedforward))
        out = U_t + x + M_t
        return out


class TSelfattention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfattention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = self.embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
       
        B,N, T, C = query.shape

     
        keys = key.reshape(B,N, T, self.heads, self.per_dim)
        queries=keys
        values=keys
      

        keys = self.keys(keys)
        values=keys
        queries=keys
       
        attnscore = torch.einsum("bnqhd, bnkhd->bnqkh", (queries, keys)) 
        attention = torch.softmax(attnscore / (self.embed_size ** (1/2)), dim=2)

        out = torch.einsum("bnqkh, bnkhd->bnqhd", (attention, values))
        out = out.reshape(B,N, T, self.embed_size)
        out = self.fc(out)

        return out
