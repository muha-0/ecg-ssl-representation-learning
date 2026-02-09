# eval/convergence.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from data.split import split_patients
from data.utils import list_patients
from data.convergence_dataset import DeterministicPatientWindows
from data.collate import collate_patient_windows

from models.tokenizers import ConvPatchTokenizer, KMeansVQPatchTokenizer
from models.encoder import ECGEncoder
from codebooks.build_codebook import load_offline_codebook

warnings.filterwarnings("ignore", message="enable_nested_tensor")


def _pick_patients(split_name: str, root: Path):
    patients = list_patients(root)
    ssl_patients, train_patients, val_patients, test_patients = split_patients(patients)

    if split_name == "train":
        return train_patients
    if split_name == "val":
        return val_patients
    if split_name == "test":
        return test_patients
    if split_name == "ssl":
        return ssl_patients
    raise ValueError(f"Unknown split: {split_name}")


def _subset_patients(patient_list, n: int, seed: int):
    if n <= 0:
        raise ValueError("n must be > 0")
    if n > len(patient_list):
        raise ValueError(f"Requested n={n} patients, but split only has {len(patient_list)} patients.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(patient_list), size=n, replace=False)
    return [patient_list[i] for i in idx]


def _list_checkpoints(exp_dir: Path):
    """
    Expects checkpoint_1000.pth, checkpoint_2000.pth, ...
    Returns list of (step:int, path:Path) sorted by step.
    """
    cks = sorted(exp_dir.glob("checkpoint_*.pth"))

    def step_of(p: Path) -> int:
        m = re.search(r"checkpoint_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    out = [(step_of(p), p) for p in cks]
    out = [x for x in out if x[0] >= 0]
    out.sort(key=lambda x: x[0])
    return out


def _build_model(
    variant: str,
    fs: int,
    patch_len: int,
    codebook_10m: Path,
    codebook_16s: Path,
):
    """
    Returns:
      make_model_fn() -> ECGEncoder
      window_samples (int)
    """
    if variant.startswith("10m"):
        window_samples = fs * (10 * 60)
        max_len = 1024
        vq_codebook = codebook_10m
    elif variant.startswith("16s"):
        window_samples = fs * 16
        max_len = (window_samples // patch_len) + 2
        vq_codebook = codebook_16s
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if variant.endswith("_cnn"):
        def make_model():
            tok = ConvPatchTokenizer(d_model=256, patch_len=patch_len)
            return ECGEncoder(tokenizer=tok, d_model=256, max_len=max_len, use_cls=False)
        return make_model, window_samples

    if variant.endswith("_vq"):
        centroids, _meta = load_offline_codebook(str(vq_codebook))

        def make_model():
            tok = KMeansVQPatchTokenizer(
                d_model=256,
                centroids=centroids,
                patch_len=patch_len,
                enforce_window_zscore=False,
                use_embedding_table=True,
                return_ids=False,
            )
            return ECGEncoder(tokenizer=tok, d_model=256, max_len=max_len, use_cls=False)

        return make_model, window_samples

    raise ValueError(f"Unknown variant: {variant}")


def recall_at_k(emb: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)):
    """
    emb: [N, D]
    labels: [N] integer class labels
    """
    emb_t = torch.from_numpy(emb)
    emb_t = F.normalize(emb_t, dim=1)
    sim = (emb_t @ emb_t.T).cpu().numpy()
    np.fill_diagonal(sim, -1e9)

    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hit = (labels[topk] == labels[:, None]).any(axis=1).mean()
        out[f"R@{k}"] = float(hit)
    return out


@torch.no_grad()
def embed_probe_set(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    H, pid = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        h, _ = model(x, return_h=True)
        H.append(h.detach().cpu().numpy())
        pid.extend(batch["patient_ids"])
    H = np.concatenate(H, axis=0)
    return H, np.asarray(pid)


def parse_args():
    p = argparse.ArgumentParser(description="Convergence sweep: retrieval Recall@K over checkpoints.")

    p.add_argument("--variant", required=True, choices=["10m_cnn", "10m_vq", "16s_cnn", "16s_vq"])
    p.add_argument("--exp_dir", type=Path, required=True, help="Directory containing checkpoint_*.pth files.")
    p.add_argument("--root", type=Path, default=Path("/data/ahmed/icentia11k"))
    p.add_argument("--split", default="test", choices=["train", "val", "test", "ssl"])

    p.add_argument("--n_patients", type=int, default=550, help="Number of patients in deterministic probe set.")
    p.add_argument("--k_windows", type=int, default=3, help="Deterministic windows per patient.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ks", type=str, default="1,5,10", help="Comma-separated list, e.g. '1,5,10'.")

    p.add_argument("--max_ckpts", type=int, default=None, help="Optionally limit how many checkpoints to evaluate.")
    p.add_argument("--out_dir", type=Path, default=None, help="If set, saves CSV + PDF plots there.")
    p.add_argument("--no_show", action="store_true", help="Do not call plt.show().")

    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--patch_len", type=int, default=160)
    p.add_argument("--codebook_10m", type=Path, default=Path("codebooks/icentia_codebook_256x160.pt"))
    p.add_argument("--codebook_16s", type=Path, default=Path("codebooks/icentia_codebook_256x160_16secs.pt"))

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip() != "")
    if len(ks) == 0:
        raise ValueError("--ks must contain at least one integer, e.g. 1,5,10")

    ckpts = _list_checkpoints(args.exp_dir)
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoint_*.pth files found in: {args.exp_dir}")
    if args.max_ckpts is not None:
        ckpts = ckpts[: args.max_ckpts]

    make_model, window_samples = _build_model(
        variant=args.variant,
        fs=args.fs,
        patch_len=args.patch_len,
        codebook_10m=args.codebook_10m,
        codebook_16s=args.codebook_16s,
    )

    # Build deterministic probe set
    patient_list = _pick_patients(args.split, args.root)
    probe_patients = _subset_patients(patient_list, args.n_patients, seed=args.seed)

    probe_ds = DeterministicPatientWindows(
        patient_dirs=probe_patients,
        window_samples=window_samples,
        fs=args.fs,
        seed=args.seed,
        k_windows=args.k_windows,
        cache_size=64,
    )
    probe_loader = DataLoader(
        probe_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_patient_windows,
    )

    rows = []
    for step, path in tqdm(ckpts, desc=f"Sweeping {args.variant}"):
        model = make_model().to(device)
        payload = torch.load(str(path), map_location=device)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model.load_state_dict(state, strict=True)

        H, pid = embed_probe_set(model, probe_loader, device)

        # Map patient_id -> integer label
        uniq = sorted(set(pid.tolist()))
        pid2i = {p: i for i, p in enumerate(uniq)}
        lab = np.array([pid2i[p] for p in pid], dtype=np.int64)

        r = recall_at_k(H, lab, ks=ks)
        rows.append({"step": int(step), **r})

    df = pd.DataFrame(rows).sort_values("step")
    print("\nLast checkpoint:")
    print(df.tail(1).to_string(index=False))

    # Plot each metric
    for k in ks:
        key = f"R@{k}"
        plt.figure()
        plt.plot(df["step"], df[key], label=f"{args.variant} {key}")
        plt.xlabel("Step")
        plt.ylabel(key)
        plt.grid(alpha=0.3)
        plt.legend()

        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_pdf = args.out_dir / f"convergence_{args.variant}_{args.split}_{key.replace('@','at')}.pdf"
            plt.savefig(out_pdf, bbox_inches="tight")
            print(f"[OK] Saved: {out_pdf}")

        if not args.no_show:
            plt.show()
        else:
            plt.close()

    # Save CSV (optional, not JSON)
    if args.out_dir is not None:
        out_csv = args.out_dir / f"convergence_{args.variant}_{args.split}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")

    """
    Example usage:

    10 min CNN sweep:
    python -m eval.convergence \
  --variant 10m_cnn \
  --exp_dir checkpoints/CNN_tokenizer_10mins \
  --out_dir artifacts/convergence \
  --split test

    10 min VQ sweep, limit checkpoints:
    python -m eval.convergence \
  --variant 10m_vq \
  --exp_dir checkpoints/VQ_tokenizer_10mins \
  --max_ckpts 25 \
  --out_dir artifacts/convergence \
  --no_show

    16-sec CNN sweep:
    python -m eval.convergence \
  --variant 16s_cnn \
  --exp_dir checkpoints/CNN_tokenizer_16secs \
  --out_dir artifacts/convergence
    """
if __name__ == "__main__":
    main()
