import torch
import torch.nn as nn
from torch_geometric.data import Data
from model import GCNEmbedding, Predictor, GraphSAGEEmbedding, GATEmbedding


class GCNClient(nn.Module):
    def __init__(self, in_dim, gnn_hidden_dim, gnn_out_dim, pred_hidden_dim, out_dim,
                 gnn_layers=2, pred_layers=2, gnn_dropout=0.2, pred_dropout=0.5):
        super(GCNClient, self).__init__()
        self.embedding_model = GCNEmbedding(
            in_dim=in_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout
        )

        self.predictor = Predictor(
            in_dim=gnn_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=out_dim,
            num_layers=pred_layers,
            dropout=pred_dropout
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None

    def forward(self, data: Data):
        """
        输入:
            data: PyG 图对象，包含 x, edge_index, train_mask, target_labels
        输出:
            logits: 所有节点的预测值
            loss: 仅对 train_mask 节点计算的 BCE 多标签损失
        """
        x = data.x
        edge_index = data.edge_index

        node_embed = self.embedding_model(x, edge_index)
        logits = self.predictor(node_embed)

        # 仅使用 train_mask 中的节点计算损失
        if hasattr(data, 'train_mask') and hasattr(data, 'target_labels'):
            mask = data.train_mask
            labels = data.target_labels
            loss = self.loss_fn(logits[mask], labels[mask])
        else:
            raise ValueError("data 需要包含 `train_mask` 和 `target_labels` 字段")

        return logits, loss

    def get_node_embedding(self, data: Data):
        return self.embedding_model(data.x, data.edge_index)

    def create_optimizer(self, lr=1e-3, weight_decay=1e-5):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class GraphSAGEClient(nn.Module):
    def __init__(self, in_dim, gnn_hidden_dim, gnn_out_dim, pred_hidden_dim, out_dim,
                 gnn_layers=2, pred_layers=2, gnn_dropout=0.2, pred_dropout=0.5):
        super(GraphSAGEClient, self).__init__()
        self.embedding_model = GraphSAGEEmbedding(
            in_dim=in_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout
        )

        self.predictor = Predictor(
            in_dim=gnn_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=out_dim,
            num_layers=pred_layers,
            dropout=pred_dropout
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None

    def forward(self, data: Data):
        """
        输入:
            data: PyG 图对象，包含 x, edge_index, train_mask, target_labels
        输出:
            logits: 所有节点的预测值
            loss: 仅对 train_mask 节点计算的 BCE 多标签损失
        """
        x = data.x
        edge_index = data.edge_index

        node_embed = self.embedding_model(x, edge_index)
        logits = self.predictor(node_embed)

        # 仅使用 train_mask 中的节点计算损失
        if hasattr(data, 'train_mask') and hasattr(data, 'target_labels'):
            mask = data.train_mask
            labels = data.target_labels
            loss = self.loss_fn(logits[mask], labels[mask])
        else:
            raise ValueError("data 需要包含 `train_mask` 和 `target_labels` 字段")

        return logits, loss

    def get_node_embedding(self, data: Data):
        return self.embedding_model(data.x, data.edge_index)

    def create_optimizer(self, lr=1e-3, weight_decay=1e-5):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)



class GATClient(nn.Module):
    def __init__(self, in_dim, gnn_hidden_dim, gnn_out_dim, pred_hidden_dim, out_dim,
                 gnn_layers=2, pred_layers=2, gnn_dropout=0.2, pred_dropout=0.5, heads=1):
        super(GATClient, self).__init__()
        self.embedding_model = GATEmbedding(
            in_dim=in_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout,
            heads=heads
        )

        self.predictor = Predictor(
            in_dim=gnn_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=out_dim,
            num_layers=pred_layers,
            dropout=pred_dropout
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = None

    def forward(self, data: Data):
        """
        输入:
            data: PyG 图对象，包含 x, edge_index, train_mask, target_labels
        输出:
            logits: 所有节点的预测值
            loss: 仅对 train_mask 节点计算的 BCE 多标签损失
        """
        x = data.x
        edge_index = data.edge_index

        node_embed = self.embedding_model(x, edge_index)
        logits = self.predictor(node_embed)

        if hasattr(data, 'train_mask') and hasattr(data, 'target_labels'):
            mask = data.train_mask
            labels = data.target_labels
            loss = self.loss_fn(logits[mask], labels[mask])
        else:
            raise ValueError("data 需要包含 `train_mask` 和 `target_labels` 字段")

        return logits, loss

    def get_node_embedding(self, data: Data):
        return self.embedding_model(data.x, data.edge_index)

    def create_optimizer(self, lr=1e-3, weight_decay=1e-5):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


