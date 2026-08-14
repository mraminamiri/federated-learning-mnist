# Federated Learning from Scratch: FedAvg on MNIST

A compact, educational implementation of **Federated Averaging (FedAvg)** using PyTorch. The repository is designed to make the core mechanics of federated learning visible instead of hiding them behind a framework.

The project compares:

- **IID client data**: each client receives an approximately representative random split.
- **Non-IID client data**: label distributions are generated with a Dirichlet partition.
- **Centralized baseline**: the same CNN trained on the full training set.

## Why this project?

Federated learning trains a shared model across multiple clients while keeping each client's raw training data local. This repository focuses on the classic synchronous FedAvg pipeline:

1. The server initializes a global model.
2. A subset of clients receives the current global weights.
3. Each selected client trains locally.
4. Clients return model parameters, not raw data.
5. The server computes a sample-size-weighted average of client parameters.
6. The updated global model is evaluated and the process repeats.

For client \(k\) with \(n_k\) local samples, FedAvg performs:

\[
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} w_{t+1}^{(k)}
\]

## Project structure

```text
federated-learning-mnist/
├── run_federated.py           # Main FL experiment
├── run_centralized.py         # Centralized reference baseline
├── src/
│   ├── config.py
│   ├── data.py                # MNIST + IID/Dirichlet partitioning
│   ├── federated.py           # Local training, FedAvg, evaluation
│   ├── centralized.py
│   ├── model.py               # Small CNN
│   └── utils.py
├── tests/
│   └── test_partition_and_fedavg.py
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 1. Run the IID federated experiment

```bash
python run_federated.py \
  --partition iid \
  --num-clients 10 \
  --clients-per-round 10 \
  --rounds 10 \
  --local-epochs 1
```

## 2. Run the Non-IID experiment

```bash
python run_federated.py \
  --partition dirichlet \
  --alpha 0.3 \
  --num-clients 10 \
  --clients-per-round 10 \
  --rounds 10 \
  --local-epochs 1
```

Lower `--alpha` creates stronger label skew. Try `1.0`, `0.5`, `0.3`, and `0.1` to see how statistical heterogeneity affects FedAvg.

## 3. Run the centralized baseline

```bash
python run_centralized.py --epochs 10
```

## Outputs

Each federated run stores:

- `config.json`
- `client_class_distribution.json`
- `metrics.csv`
- `accuracy.png`
- `loss.png`

under a separate folder in `results/`.

No result values are committed in advance; run the experiments locally so the plots reflect your actual environment and configuration.

## What to inspect

A useful first experiment is to keep every setting fixed and change only the data partition:

| Experiment | Partition | Dirichlet alpha | Question |
|---|---|---:|---|
| A | IID | — | How quickly does FedAvg converge when clients are statistically similar? |
| B | Non-IID | 1.0 | What happens under mild label skew? |
| C | Non-IID | 0.3 | How much does stronger heterogeneity hurt convergence? |
| D | Non-IID | 0.1 | Does the global model become unstable or converge more slowly? |

## Implementation notes

### Local training

Every selected client starts each communication round from the **same global model**, then performs local SGD on only its own partition.

### Aggregation

The server weights each client's update by its number of local examples. This is the standard FedAvg weighting rule and avoids treating a client with 100 samples as equivalent to one with 10,000 samples.

### Non-IID simulation

The Dirichlet partition creates different class mixtures across clients. This is a simple way to reproduce one of the central challenges in federated learning: **statistical heterogeneity**.

## Tests

```bash
pytest -q
```

The tests verify that:

- IID partitioning covers each sample exactly once.
- FedAvg performs the correct sample-size-weighted parameter average.

## Suggested learning path

This repository is deliberately a baseline. Good next extensions are:

1. Partial client participation and client dropout.
2. Per-client accuracy and fairness metrics.
3. FedProx for heterogeneous clients.
4. Differential privacy with gradient clipping and noise.
5. Secure aggregation concepts.
6. Personalized federated learning.
7. Communication-cost tracking.
8. Real-world sensitive/tabular datasets.

## Reproducibility

Experiments expose a random seed and write the full run configuration to disk. Exact numerical results can still vary slightly by hardware, PyTorch version, and accelerator backend.

## Author

**Mohammad Amin Amiri**  
Computer Science | Machine Learning | Federated Learning

## License

MIT
