import torch
from client import GCNClient, GraphSAGEClient, GATClient  # 确保路径正确
from sklearn.metrics import precision_score, recall_score
import argparse
import os


def load_model(checkpoint_path, model_args, device):
    model = GATClient(**model_args)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, data, device, threshold=0.1):
    data = data.to(device)

    with torch.no_grad():
        logits, _ = model(data)
        probs = torch.sigmoid(logits)

    # 使用 test_mask 选出测试节点
    test_mask = data.test_mask
    preds = (probs[test_mask] > threshold).float().cpu().numpy()
    labels = data.target_labels[test_mask].cpu().numpy()

    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)
    return precision, recall


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAT model")
    parser.add_argument('--data_path', type=str, default='./Parsed_dataset/BlogCatalog/client0.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='./Check_SAGE')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    data = torch.load(args.data_path)

    if not hasattr(data, 'test_mask') or not hasattr(data, 'target_labels'):
        raise ValueError("数据缺失 test_mask 或 target_labels")

    in_dim = data.x.shape[1]
    out_dim = data.target_labels.shape[1]

    model_args = dict(
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

    # 遍历所有保存的 checkpoint 进行评估
    for i in range(1, 21):
        checkpoint_path = os.path.join(f'Check_GAT/gat_epoch_{100 * i}.pth')
        model = load_model(checkpoint_path, model_args, args.device)
        precision, recall = evaluate(model, data, args.device)
        print(f"[{checkpoint_path}] Precision: {precision:.4f}, Recall: {recall:.4f}")


if __name__ == '__main__':
    main()
