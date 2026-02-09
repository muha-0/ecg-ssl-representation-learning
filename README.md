# Icentia11k ECG SSL Encoder + Rhythm Evaluation

This repo trains self-supervised ECG encoders on **Icentia11k**, then evaluates them for AF vs Normal rhythm classification, and provides clustering + convergence diagnostics.

## Project layout

- `train/train_ssl.py` — SSL training (SimCLR-style pairs) for 10-min or 16-sec windows.
- `eval/eval.py` — supervised evaluation (linear probe or end-to-end fine-tuning), supports 10m and 16s.
- `eval/clustering.py` — patient clustering diagnostics (t-SNE + retrieval-style clustering metrics).
- `eval/convergence.py` — checkpoint sweep showing retrieval Recall@K convergence over training.
- `codebooks/` — offline VQ codebooks for KMeans VQ tokenizer.
- `data/` — datasets, preprocessing, splits, augmentation, collates.

---

## Setup

### 1) Environment
Activate your environment (example):
```bash
conda activate eeg_challenge
```

### 2) Data root
By default, scripts assume:
- Icentia11k root: `/data/ahmed/icentia11k`

You can override with `--root ...` in each CLI.

### 3) Codebooks (for VQ tokenizer)
You need a codebook file when using `tokenizer=vq`:
- 10-min: `codebooks/icentia_codebook_256x160.pt`
- 16-sec: `codebooks/icentia_codebook_256x160_16secs.pt`

---

## 1) SSL Training (train/train_ssl.py)

Trains an `ECGEncoder` using patient-pair SSL windows.

### Common options
- `--backend {wfdb,npy}`:
  - `wfdb`: read directly from WFDB each batch.
  - `npy`: read from preprocessed memmaps (faster).
- `--window_sec`:
  - `600` for 10-min
  - `16` for 16-sec
- `--tokenizer {vq,cnn}`:
  - `vq`: discretized KMeans-VQ patch tokenizer (requires `--codebook_path` or config default)
  - `cnn`: continuous Conv patch tokenizer
- `--aug_preset {10min,16s}` controls augmentation recipe.
- `--exp_dir` saves checkpoints and logs.
- `--max_steps` controls training length (default `50000`).

### Example: 10-min SSL, VQ tokenizer, WFDB backend
```bash
python -m train.train_ssl \
  --backend wfdb \
  --aug_preset 10min \
  --window_sec 600 \
  --tokenizer vq \
  --exp_dir checkpoints/VQ_tokenizer_10mins
```

### Example: preprocess once for npy backend
```bash
python -m train.train_ssl \
  --backend npy \
  --processed_root /data/ahmed/icentia_processed \
  --preprocess
```

### Example: 16-sec SSL, VQ tokenizer, npy backend
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

## 2) Supervised Evaluation (eval/eval.py)

Unified CLI for supervised rhythm classification evaluation:
- window length: `10m` or `16s`
- tokenizer: `cnn` or `vq`
- init: random or load SSL encoder checkpoint
- mode: linear probe (frozen encoder) or fine-tune end-to-end
- optional: for 16s models, evaluate aggregated performance on 10-min windows (`--eval10m-agg`)

### Key arguments
- `--window {10m,16s}` (or numeric seconds like `"600"`)
- `--mode {probe,finetune}`
- `--tokenizer {cnn,vq}`
- `--ckpt PATH` (optional, omit for random init)
- `--codebook PATH` (required if `--tokenizer vq`)
- `--save PATH` (optional, saves encoder+head state dicts)
- `--use-amp {true,false}` (default true)

### Example: 10-min, CNN, linear probe, load pretrained encoder
```bash
python -m eval.eval \
  --window 10m \
  --tokenizer cnn \
  --mode probe \
  --ckpt checkpoints/CNN_tokenizer_10mins/checkpoint_50000.pth \
  --save checkpoints/linear_probe_CNN_tokenizer_10mins/model.pth
```

### Example: 16-sec, VQ, random init, fine-tune end-to-end
```bash
python -m eval.eval \
  --window 16s \
  --tokenizer vq \
  --mode finetune \
  --codebook codebooks/icentia_codebook_256x160_16secs.pt \
  --save checkpoints/fine_tuned_VQ_tokenizer_random_16secs/model.pth
```

### Example: 16-sec, CNN, probe, plus 10-min aggregated evaluation
```bash
python -m eval.eval \
  --window 16s \
  --tokenizer cnn \
  --mode probe \
  --ckpt checkpoints/CNN_tokenizer_16secs/checkpoint_50000.pth \
  --eval10m-agg
```

---

## 3) Patient Clustering Diagnostics (eval/clustering.py)

Computes:
1) Retrieval-style clustering metrics (Recall@K + silhouette) on a larger subset (`--metrics_patients`, default 550)
2) t-SNE plot on a smaller subset (`--tsne_patients`, default 40)

### Key arguments
- `--variant {10m_cnn,10m_vq,16s_cnn,16s_vq}`
- `--ckpt PATH` (encoder checkpoint to evaluate)
- `--split {train,val,test,ssl}` (default test)
- `--fig_dir DIR` (optional, saves `tsne_<variant>_<split>.pdf`)
- `--no_show` (server-safe)

### Example: 10-min CNN clustering
```bash
python -m eval.clustering \
  --variant 10m_cnn \
  --ckpt checkpoints/CNN_tokenizer_10mins/checkpoint_50000.pth \
  --fig_dir docs \
  --no_show
```

### Example: 16-sec VQ clustering
```bash
python -m eval.clustering \
  --variant 16s_vq \
  --ckpt checkpoints/VQ_tokenizer_16secs/checkpoint_50000.pth \
  --fig_dir docs \
  --no_show
```

---

## 4) Convergence Sweep (eval/convergence.py)

Sweeps a directory of checkpoints (`checkpoint_*.pth`) and evaluates retrieval Recall@K on a **deterministic probe set** (same patients/windows across checkpoints), then plots Recall@K vs step.

Outputs:
- PDF plot(s) per metric if `--out_dir` is set
- CSV table of results if `--out_dir` is set

### Key arguments
- `--variant {10m_cnn,10m_vq,16s_cnn,16s_vq}`
- `--exp_dir DIR` directory containing `checkpoint_*.pth`
- `--out_dir DIR` (optional) save PDF(s) + CSV
- `--ks "1,5,10"` metrics to compute
- `--max_ckpts N` optionally limit number of checkpoints
- `--n_patients` deterministic probe patients (default 550)
- `--k_windows` windows per patient (default 3)
- `--no_show` for servers

### Example: 10-min CNN sweep
```bash
python -m eval.convergence \
  --variant 10m_cnn \
  --exp_dir checkpoints/CNN_tokenizer_10mins \
  --out_dir artifacts/convergence \
  --split test \
  --no_show
```

### Example: 10-min VQ sweep (limit checkpoints)
```bash
python -m eval.convergence \
  --variant 10m_vq \
  --exp_dir checkpoints/VQ_tokenizer_10mins \
  --max_ckpts 25 \
  --out_dir artifacts/convergence \
  --no_show
```

---

## Notes / Tips

- **VQ tokenizers require a codebook**:
  - training uses `--codebook_path` in SSL
  - eval uses `--codebook` in supervised CLI
  - clustering/convergence use `--codebook_10m` and `--codebook_16s` (defaults already point to `codebooks/`)

- For headless servers:
  - add `--no_show` to `eval.clustering` and `eval.convergence`
  - for matplotlib backends, saving plots via `--fig_dir` / `--out_dir` is the safest path

- Default split is `test` across evaluation/diagnostics scripts unless overridden.

---

## Repo tree (max depth 2)

```text
.
./eval
./models
./train
./data
./codebooks
./checkpoints
./docs
```
