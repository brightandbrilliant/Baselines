import argparse
import torch
from torch_geometric.data import Data
from client import GCNClient, GraphSAGEClient, GATClient  # 确保路径正确
import os


def train(client, data, device, epochs=2000, save_path='./Check_GAT'):
    data = data.to(device)
    client.to(device)
    client.train()

    for epoch in range(epochs):
        client.optimizer.zero_grad()
        logits, loss = client(data)
        loss.backward()
        client.optimizer.step()

        print(f"[Epoch {epoch+1:03d}] Loss: {loss.item():.4f}")

        # 保存模型
        if save_path and (epoch + 1) % 10 == 0:
            torch.save(client.state_dict(), os.path.join(save_path, f'gat_epoch_{epoch+1}.pth'))


def main():
    parser = argparse.ArgumentParser(description="Train GCN baseline on single client")
    parser.add_argument('--data_path', type=str, default='./Parsed_dataset/BlogCatalog/client0.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./Check_GAT')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Using device: {args.device}")

    # 加载数据
    data = torch.load(args.data_path)

    if not hasattr(data, 'train_mask') or not hasattr(data, 'target_labels'):
        raise ValueError("数据缺失 train_mask 或 target_labels")

    in_dim = data.x.shape[1]
    out_dim = data.target_labels.shape[1]

    # 创建 GCNClient 模型
    client = GATClient(
        in_dim=in_dim,
        gnn_hidden_dim=64,
        gnn_out_dim=128,
        pred_hidden_dim=128,
        out_dim=out_dim,
        gnn_layers=3,
        pred_layers=3,
        gnn_dropout=0.4,
        pred_dropout=0.4,
        heads=3
    )

    client.create_optimizer(lr=1e-3, weight_decay=1e-5)

    # 开始训练
    train(client, data, args.device, epochs=args.epochs, save_path=args.save_dir)


if __name__ == '__main__':
    main()
