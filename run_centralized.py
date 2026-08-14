from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.centralized import train_centralized
from src.data import build_test_loader, load_mnist
from src.model import SimpleCNN
from src.utils import ensure_dir, resolve_device, set_seed


def main():
    p = argparse.ArgumentParser(description="Centralized MNIST baseline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_ds, test_ds = load_mnist("./data")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = build_test_loader(test_ds)

    history = train_centralized(
        SimpleCNN(), train_loader, test_loader, device, args.epochs, args.lr
    )

    out = ensure_dir("./results/centralized")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(7, 4))
    plt.plot([x["epoch"] for x in history], [x["test_accuracy"] for x in history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Centralized MNIST baseline")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(Path(out) / "accuracy.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
