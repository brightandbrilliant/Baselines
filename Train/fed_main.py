import argparse
import torch
from client import GCNClient
from Train.FedAVGTrainer import FedAvgTrainer
import os

def build_gcn_clients(n_clients, feature_dim, out_dim, device='cpu'):
    clients = {}
    for cid in range(n_clients):
        client = GCNClient(
            in_dim=feature_dim,
            gnn_hidden_dim=32,
            gnn_out_dim=64,
            pred_hidden_dim=32,
            out_dim=out_dim,
            gnn_layers=2,
            pred_layers=2,
            gnn_dropout=0.2,
            pred_dropout=0.5
        ).to(device)
        client.create_optimizer(lr=1e-3, weight_decay=5e-4)
        clients[cid] = client
    return clients


def main():
    parser = argparse.ArgumentParser(description="FedAvg GCN Trainer")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_clients', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=39)
    parser.add_argument('--local_steps', type=int, default=10)
    parser.add_argument('--total_rounds', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='Check_GCN_FedAvg')
    parser.add_argument('--data_dir', type=str, default='./Parsed_dataset/BlogCatalog')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Step 1：读取每个 Client 的数据
    train_data_dict = {}
    out_dims = set()
    for cid in range(args.n_clients):
        path = os.path.join(args.data_dir, f'client{cid}.pt')
        data = torch.load(path)

        if not hasattr(data, 'train_mask') or not hasattr(data, 'target_labels'):
            raise ValueError(f"client{cid}.pt 缺少必要字段：`train_mask` 或 `target_labels`")

        out_dims.add(data.target_labels.shape[1])
        train_data_dict[cid] = data

    # Step 2：统一检查所有 client 的标签维度一致
    if len(out_dims) != 1:
        raise ValueError(f"所有 client 的 target_labels 应该具有相同维度，但发现多个不同维度：{out_dims}")
    out_dim = out_dims.pop()

    # Step 3：构建 GCN Clients
    clients = build_gcn_clients(
        n_clients=args.n_clients,
        feature_dim=args.feature_dim,
        out_dim=out_dim,
        device=args.device
    )

    # Step 4：构建 FedAvg Trainer
    trainer = FedAvgTrainer(
        clients=clients,
        train_loaders=train_data_dict,  # 注意不是 loader，是 Data 对象
        device=args.device,
        local_steps=args.local_steps,
        total_rounds=args.total_rounds,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every
    )

    # Step 5：开始训练
    trainer.train()


if __name__ == '__main__':
    main()
