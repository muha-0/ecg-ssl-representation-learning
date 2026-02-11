
# The Impact of Temporal Context Length and Encoding Strategies on Self-Supervised ECG Representation Learning

Official implementation of the experiments presented in:

**Ahmed Sameh, Ramzi Al-Sharawi, Yogatheesan Varatharajah**
*The Impact of Temporal Context Length and Encoding Strategies on Self-Supervised ECG Representation Learning*

This repository provides a unified framework for **self-supervised representation learning on ambulatory ECG** using the Icentia11k dataset. The code reproduces the controlled study comparing:

* **Temporal context:** short (16 s) vs long (10 min) input horizons
* **Encoding strategy:** continuous CNN patch embeddings vs discretized VQ tokens
* Shared **Transformer backbone and training protocol**

Representations are evaluated through:

* Downstream **AFib/AFL vs Normal rhythm classification**
* **Patient-level retrieval (Recall@k)** to assess cross-session consistency
* **t-SNE visualization** of learned representations
* **Convergence diagnostics** across SSL checkpoints

---

# Repository Structure

* `train/train_ssl.py` — Self-supervised pretraining (contrastive patient-pair learning).
* `eval/eval.py` — Supervised evaluation (linear probing or fine-tuning).
* `eval/clustering.py` — Patient-level retrieval metrics and t-SNE visualization.
* `eval/convergence.py` — Retrieval convergence vs training step.
* `models/` — Tokenizers and Transformer encoder (`ECGEncoder`).
* `data/` — Datasets, preprocessing, augmentation, splits, collate functions.
* `codebooks/` — Offline K-means VQ codebooks.
* `checkpoints/` — Model weights used in the paper.
* `docs/` — Figures and visualizations.

---

# Setup

## Environment

Activate your environment (example):

```bash
conda activate ecg_ssl
```

## Dataset

All scripts assume a local install of **Icentia11k**:

```
/data/ahmed/icentia11k
```

Override with `--root` in any CLI.

## VQ Codebooks

Required for VQ tokenizer:

* 10-min: `codebooks/icentia_codebook_256x160.pt`
* 16-sec: `codebooks/icentia_codebook_256x160_16secs.pt`

---

# 1. Self-Supervised Pretraining

`train/train_ssl.py`

Trains an ECG encoder using **contrastive patient-pair learning (InfoNCE)**.

Supports:

* 10-minute or 16-second windows
* CNN (continuous) or VQ (discretized) tokenization
* WFDB or preprocessed memmap backend

## Key arguments

* `--backend {wfdb,npy}`
* `--window_sec {600,16}`
* `--tokenizer {cnn,vq}`
* `--aug_preset {10min,16s}`
* `--exp_dir` save checkpoints
* `--max_steps` training length
* `--codebook_path` required for VQ

## Example — 10-min SSL (CNN)

```bash
python -m train.train_ssl \
  --backend wfdb \
  --aug_preset 10min \
  --window_sec 600 \
  --tokenizer cnn \
  --exp_dir checkpoints/CNN_tokenizer_10mins
```

## Example — preprocess once (for npy backend)

```bash
python -m train.train_ssl \
  --backend npy \
  --processed_root /data/ahmed/icentia_processed \
  --preprocess
```

## Example — 16-sec SSL (VQ)

```bash
python -m train.train_ssl \
  --backend npy \
  --processed_root /data/ahmed/icentia_processed \
  --aug_preset 16s \
  --window_sec 16 \
  --tokenizer vq \
  --codebook_path codebooks/icentia_codebook_256x160_16secs.pt \
  --exp_dir checkpoints/VQ_tokenizer_16secs
```

---

# 2. Supervised Evaluation

`eval/eval.py`

Unified CLI for AFib/AFL vs Normal rhythm classification.

Supports:

* Window length: `10m` or `16s`
* Tokenizer: `cnn` or `vq`
* Initialization: random or pretrained encoder
* Training mode: **linear probe** or **end-to-end fine-tune**
* Optional 10-minute aggregation for 16-sec encoders

## Key arguments

* `--window {10m,16s}`
* `--mode {probe,finetune}`
* `--tokenizer {cnn,vq}`
* `--ckpt` pretrained encoder (optional)
* `--codebook` required for VQ
* `--save` save trained model

## Example — 10-min CNN, linear probe

```bash
python -m eval.eval \
  --window 10m \
  --tokenizer cnn \
  --mode probe \
  --ckpt checkpoints/CNN_tokenizer_10mins/checkpoint_50000.pth
```

## Example — 16-sec VQ, fine-tune

```bash
python -m eval.eval \
  --window 16s \
  --tokenizer vq \
  --mode finetune \
  --codebook codebooks/icentia_codebook_256x160_16secs.pt
```

## Example — 16-sec CNN with 10-min aggregation

```bash
python -m eval.eval \
  --window 16s \
  --tokenizer cnn \
  --mode probe \
  --ckpt checkpoints/CNN_tokenizer_16secs/checkpoint_50000.pth \
  --eval10m-agg
```

---

# 3. Patient Retrieval & Representation Visualization

`eval/clustering.py`

Computes:

* **Retrieval metrics (Recall@k)** on ~550 patients
* **t-SNE visualization** on a smaller subset

## Key arguments

* `--variant {10m_cnn,10m_vq,16s_cnn,16s_vq}`
* `--ckpt` encoder checkpoint
* `--fig_dir` save t-SNE PDF
* `--no_show` for headless servers

## Example

```bash
python -m eval.clustering \
  --variant 10m_cnn \
  --ckpt checkpoints/CNN_tokenizer_10mins/checkpoint_50000.pth \
  --fig_dir docs \
  --no_show
```

---

# 4. Convergence Diagnostics

`eval/convergence.py`

Evaluates retrieval **Recall@k vs training step** across checkpoints using a deterministic probe set.

Outputs:

* PDF curves
* CSV table of metrics

## Example

```bash
python -m eval.convergence \
  --variant 10m_cnn \
  --exp_dir checkpoints/CNN_tokenizer_10mins \
  --out_dir artifacts/convergence \
  --no_show
```

---

# Notes

* VQ models require a codebook for both training and evaluation.
* Use `--no_show` when running on remote servers.
* All splits are **patient-level** and deterministic for reproducibility.
* To use backend npy, you must preprocess first.
---

# Citation

```bibtex
@article{sameh2026ecgssl,
  title={The Impact of Temporal Context Length and Encoding Strategies on Self-Supervised ECG Representation Learning},
  author={Sameh, Ahmed and Al-Sharawi, Ramzi and Varatharajah, Yogatheesan},
  year={2026}
}
```

---

# Data

The Icentia11k dataset is publicly available. This repository does **not** redistribute the data. Please obtain it separately and set `--root` accordingly.

---

# Contact

For questions, please open a GitHub issue or contact the authors.
