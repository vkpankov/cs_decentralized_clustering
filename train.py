import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import build_model
from dataset import GridAgentsDataset
from loss import hungarian_loss

parser = argparse.ArgumentParser(
    description="Train a model with configurable parameters."
)
parser.add_argument("--clusters_num", type=int, default=10, help="Number of clusters")
parser.add_argument("--agents_num", type=int, default=1000, help="Number of agents")
parser.add_argument("--grid_size", type=int, default=128, help="Grid size")
parser.add_argument(
    "--comp_size",
    type=int,
    nargs=2,
    default=(12, 12),
    help="Compressed size (height, width)",
)
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="checkpoints",
    help="Directory to save checkpoints",
)
args = parser.parse_args()


def get_scheduler(optimizer, warmup_steps=1000, total_steps=10000):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


ds = GridAgentsDataset(
    grid_size=args.grid_size, agents_num=args.agents_num, max_clusters=args.clusters_num
)
train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)

rec_model = build_model(
    compressed_size=tuple(args.comp_size), clusters_number=args.clusters_num
)
rec_model = rec_model.to("cuda")

optimizer = torch.optim.AdamW(rec_model.parameters(), lr=args.lr)
scheduler = get_scheduler(
    optimizer, warmup_steps=1000, total_steps=len(train_loader) * args.epochs
)
criterion = nn.SmoothL1Loss()

rec_model.train()
for epoch in range(args.epochs):
    full_train_loss = 0.0
    train_loss = 0.0
    cnt = 0
    acc = 0
    for data, target_img, target in tqdm(train_loader):
        data, target_img = data.to("cuda"), target_img.to("cuda")
        target = target.to("cuda")

        optimizer.zero_grad()
        output = rec_model(data)

        with torch.multiprocessing.Pool(10) as pool:
            loss = (
                hungarian_loss(
                    output,
                    target[:, :, 1:3].swapaxes(-1, -2),
                    thread_pool=pool,
                )
                .unsqueeze(0)
                .mean()
            )
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        acc += 1

        if cnt % 100 == 0:
            avg_loss = train_loss / acc if acc != 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Iteration {cnt}: Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
            )
            train_loss = 0
            acc = 0
            checkpoint_path = f"{args.checkpoint_dir}/last_c{args.clusters_num}_p{args.comp_size[0]}x{args.comp_size[1]}.pt"
            torch.save(rec_model.state_dict(), checkpoint_path)
        cnt += 1

    full_train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch + 1}, Training Loss: {full_train_loss:.4f}")

print("Training and validation complete.")
