import torch
from client import GCNClient, GraphSAGEClient, GATClient  # 确保路径正确
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
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

    test_mask = data.test_mask
    preds = (probs[test_mask] > threshold).float().cpu().numpy()
    labels = data.target_labels[test_mask].cpu().numpy()

    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)
    f1 = f1_score(labels, preds, average='micro', zero_division=0)

    # FPR计算（按列/标签统计）
    fpr_list = []
    for i in range(labels.shape[1]):
        cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp = cm[0, 0], cm[0, 1]
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0
        fpr_list.append(fpr)
    fpr_macro = sum(fpr_list) / len(fpr_list)

    return precision, recall, f1, fpr_macro


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAT model")
    parser.add_argument('--data_path', type=str, default='./Parsed_dataset/BlogCatalog/client0.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='./Check_GAT')
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
    for i in range(1, 11):
        checkpoint_path = os.path.join(f'Check_GAT/gat_epoch_{10 * i}.pth')
        model = load_model(checkpoint_path, model_args, args.device)
        precision, recall, f1, fpr = evaluate(model, data, args.device)
        print(f"[{checkpoint_path}] Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1:{f1:.4f}, FPR:{fpr:.4f}")


if __name__ == '__main__':
    main()
