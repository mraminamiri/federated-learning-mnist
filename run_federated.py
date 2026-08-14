from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import FLConfig
from src.data import (
    build_client_loaders,
    build_test_loader,
    class_distribution,
    dirichlet_partition,
    iid_partition,
    load_mnist,
)
from src.federated import evaluate, fedavg, sample_clients, train_local
from src.model import SimpleCNN
from src.utils import ensure_dir, resolve_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Educational FedAvg on MNIST")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--num-clients", type=int, default=10)
    p.add_argument("--clients-per-round", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def save_history(history, output_dir: Path):
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    rounds = [r["round"] for r in history]
    acc = [r["test_accuracy"] for r in history]
    loss = [r["test_loss"] for r in history]

    plt.figure(figsize=(7, 4))
    plt.plot(rounds, acc, marker="o")
    plt.xlabel("Communication round")
    plt.ylabel("Test accuracy")
    plt.title("FedAvg on MNIST")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(rounds, loss, marker="o")
    plt.xlabel("Communication round")
    plt.ylabel("Test loss")
    plt.title("FedAvg on MNIST")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=160)
    plt.close()


def main():
    args = parse_args()
    cfg = FLConfig(
        seed=args.seed,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        partition=args.partition,
        dirichlet_alpha=args.alpha,
        device=args.device,
    )

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    train_ds, test_ds = load_mnist(cfg.data_dir)
    targets = train_ds.targets.numpy()

    if cfg.partition == "iid":
        partitions = iid_partition(len(train_ds), cfg.num_clients, cfg.seed)
    else:
        partitions = dirichlet_partition(
            targets,
            cfg.num_clients,
            cfg.dirichlet_alpha,
            cfg.seed,
            cfg.min_client_samples,
        )

    client_loaders = build_client_loaders(train_ds, partitions, cfg.batch_size)
    test_loader = build_test_loader(test_ds)

    run_name = (
        f"fedavg_{cfg.partition}_clients{cfg.num_clients}_rounds{cfg.rounds}"
        + (f"_alpha{cfg.dirichlet_alpha}" if cfg.partition == "dirichlet" else "")
    )
    output_dir = ensure_dir(str(Path(cfg.results_dir) / run_name))
    save_json(vars(cfg), str(output_dir / "config.json"))
    save_json(class_distribution(targets, partitions), str(output_dir / "client_class_distribution.json"))

    model = SimpleCNN()
    rng = random.Random(cfg.seed)
    history = []

    initial = evaluate(model, test_loader, device)
    print(f"Round 00 | test_loss={initial['loss']:.4f} | test_acc={initial['accuracy']:.4f}")

    for rnd in range(1, cfg.rounds + 1):
        selected = sample_clients(cfg.num_clients, cfg.clients_per_round, rng)
        local_updates = []
        weighted_local_loss = 0.0
        total_local_examples = 0

        for cid in selected:
            state, n, local_loss = train_local(
                model,
                client_loaders[cid],
                device,
                cfg.local_epochs,
                cfg.learning_rate,
            )
            local_updates.append((state, n))
            weighted_local_loss += local_loss * n
            total_local_examples += n

        model.load_state_dict(fedavg(local_updates))
        metrics = evaluate(model, test_loader, device)
        row = {
            "round": rnd,
            "selected_clients": ";".join(map(str, selected)),
            "mean_local_loss": weighted_local_loss / max(total_local_examples, 1),
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
        }
        history.append(row)
        print(
            f"Round {rnd:02d} | local_loss={row['mean_local_loss']:.4f} | "
            f"test_loss={row['test_loss']:.4f} | test_acc={row['test_accuracy']:.4f}"
        )

    save_history(history, output_dir)
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
