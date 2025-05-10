import os
import torch
import copy

class FedAvgTrainer:
    def __init__(self, clients, train_loaders, device, local_steps, total_rounds,
                 checkpoint_dir='Checkpoints_Baseline', save_every=1):
        self.clients = clients  # dict: {cid: GCNClient instance}
        self.train_loaders = train_loaders  # dict: {cid: DataLoader}
        self.device = device
        self.local_steps = local_steps
        self.total_rounds = total_rounds
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        for round in range(self.total_rounds):
            print(f"\n=== FedAvg Round {round + 1} ===")
            local_weights = []

            for cid, client in self.clients.items():
                client.to(self.device)
                client.train()

                # 初始化 optimizer（如未手动调用 create_optimizer）
                if client.optimizer is None:
                    client.create_optimizer(lr=0.001, weight_decay=5e-4)

                data = self.train_loaders[cid]  # 这里直接是 Data，不是 loader！
                data = data.to(self.device)

                for _ in range(self.local_steps):
                    try:
                        logits, loss = client.forward(data)
                    except Exception as e:
                        print(f"[Error] Client {cid} forward error: {e}")
                        continue

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[Warning] Client {cid} produced NaN/Inf loss. Skipping this step.")
                        continue

                    client.optimizer.zero_grad()
                    loss.backward()
                    client.optimizer.step()

                local_weights.append(copy.deepcopy(client.state_dict()))

            # 聚合参数
            global_weights = self.average_weights(local_weights)

            # 更新每个 client
            for client in self.clients.values():
                self.safe_load_state_dict(client, global_weights)

            # 保存模型
            if (round + 1) % self.save_every == 0:
                self.save_checkpoint(global_weights, round + 1)

    def average_weights(self, local_weights):
        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights)):
                if avg_weights[key].shape == local_weights[i][key].shape:
                    avg_weights[key] += local_weights[i][key]
            avg_weights[key] /= len(local_weights)
        return avg_weights

    def save_checkpoint(self, state_dict, round_num):
        path = os.path.join(self.checkpoint_dir, f'baseline_global_model_round_{round_num}.pt')
        torch.save(state_dict, path)
        print(f"Checkpoint saved at round {round_num}: {path}")

    def safe_load_state_dict(self, model, state_dict):
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"[Warning] Skipped loading param {k} due to shape mismatch.")
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)


