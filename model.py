import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCNEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super(GCNEmbedding, self).__init__()
        assert num_layers >= 2, "num_layers should be at least 2"

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x  # 返回的是最终节点嵌入，不做 softmax


class GraphSAGEEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.2):
        super(GraphSAGEEmbedding, self).__init__()
        assert num_layers >= 2, "num_layers should be at least 2"

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.layers.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x



class GATEmbedding(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.2, heads=3):
        super(GATEmbedding, self).__init__()
        assert num_layers >= 2, "num_layers should be at least 2"

        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.heads = heads

        # 第一层：输入维度 -> hidden_dim
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))

        # 中间层（维度保持一致）
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))

        # 最后一层：hidden_dim -> out_dim（不再 concat）
        self.layers.append(GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x







class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super(Predictor, self).__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)  # 不做 softmax，交给 loss 函数处理
