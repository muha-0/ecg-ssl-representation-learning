# eval/clustering.py
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch

from data.split import split_patients
from data.utils import list_patients
from data.clustering_dataset import IcentiaPatientWindowsDataset
from data.collate import collate_patient_windows

from models.tokenizers import KMeansVQPatchTokenizer, ConvPatchTokenizer
from models.encoder import ECGEncoder
from codebooks.build_codebook import load_offline_codebook

from .metrics import compute_metrics
from .plots import plot_tsne

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


def _load_encoder_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location=device)

    # Handle common checkpoint formats:
    # 1) {"model_state_dict": ...}
    # 2) raw state_dict
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        step = payload.get("step", "unknown")
    else:
        state = payload
        step = "unknown"

    model.load_state_dict(state, strict=True)
    return step


def _build_model_and_window_args(
    variant: str,
    fs: int,
    patch_len: int,
    codebook_10m: Path,
    codebook_16s: Path,
):
    """
    Returns:
      model (ECGEncoder), window_samples (int)
    """
    if variant.startswith("10m"):
        window_sec = 10 * 60
        window_samples = fs * window_sec
        max_len = 1024
        vq_codebook = codebook_10m
    elif variant.startswith("16s"):
        window_sec = 16
        window_samples = fs * window_sec
        max_len = (window_samples // patch_len) + 2
        vq_codebook = codebook_16s
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if variant.endswith("_cnn"):
        tok = ConvPatchTokenizer(d_model=256, patch_len=patch_len)
    elif variant.endswith("_vq"):
        centroids, meta = load_offline_codebook(str(vq_codebook))
        tok = KMeansVQPatchTokenizer(
            d_model=256,
            centroids=centroids,
            patch_len=patch_len,
            enforce_window_zscore=False,
            use_embedding_table=True,
            return_ids=False,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = ECGEncoder(
        tokenizer=tok,
        d_model=256,
        max_len=max_len,
        use_cls=False,
    )

    return model, window_samples


def _make_loader(
    patient_dirs,
    window_samples: int,
    fs: int,
    seed: int,
    k_windows: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    ds = IcentiaPatientWindowsDataset(
        patient_dirs=patient_dirs,
        window_samples=window_samples,
        fs=fs,
        seed=seed,
        cache_size=64,
        k_windows=k_windows,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_patient_windows,
    )
    return loader


@torch.no_grad()
def _embed_windows(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    all_h = []
    all_pid = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        h, _ = model(x, return_h=True)  # [M, D]
        all_h.append(h.detach().cpu().numpy())
        all_pid.extend(batch["patient_ids"])

    H = np.concatenate(all_h, axis=0)
    pid = np.array(all_pid)
    return H, pid


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute clustering metrics and t-SNE plots for an encoder (10m/16s, CNN/VQ)."
    )

    p.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["10m_cnn", "10m_vq", "16s_cnn", "16s_vq"],
        help="Which model/tokenizer configuration to use.",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint path to load encoder weights from.",
    )

    p.add_argument("--root", type=Path, default=Path("/data/ahmed/icentia11k"))
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "ssl"])

    p.add_argument("--tsne_patients", type=int, default=40)
    p.add_argument("--metrics_patients", type=int, default=550)

    p.add_argument("--k_windows", type=int, default=5, help="Windows per patient.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument("--perplexity", type=int, default=15)

    p.add_argument(
        "--fig_dir",
        type=Path,
        default=None,
        help="Optional directory to save the t-SNE figure. If omitted, it will just display.",
    )
    p.add_argument(
        "--no_show",
        action="store_true",
        help="Do not call plt.show() in plotting (useful on servers).",
    )

    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--patch_len", type=int, default=160)

    p.add_argument("--codebook_10m", type=Path, default=Path("codebooks/icentia_codebook_256x160.pt"))
    p.add_argument("--codebook_16s", type=Path, default=Path("codebooks/icentia_codebook_256x160_16secs.pt"))

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    patient_list = _pick_patients(args.split, args.root)

    # Build model + window length from variant
    model, window_samples = _build_model_and_window_args(
        variant=args.variant,
        fs=args.fs,
        patch_len=args.patch_len,
        codebook_10m=args.codebook_10m,
        codebook_16s=args.codebook_16s,
    )
    model.to(device)

    step = _load_encoder_checkpoint(model, args.ckpt, device=device)
    print(f"[OK] Loaded {args.variant} checkpoint: {args.ckpt} (step={step})")

    # -------------------------
    # Metrics subset (default 550)
    # -------------------------
    metrics_patients = _subset_patients(patient_list, args.metrics_patients, seed=args.seed)
    metrics_loader = _make_loader(
        patient_dirs=metrics_patients,
        window_samples=window_samples,
        fs=args.fs,
        seed=args.seed,
        k_windows=args.k_windows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    Hm, pidm = _embed_windows(model, metrics_loader, device)
    metrics_out = compute_metrics(Hm, pidm)

    print("\n--- Metrics ---")
    for k, v in metrics_out.items():
        print(f"{k:12s}: {v:.4f}")

    # -------------------------
    # t-SNE subset (default 40)
    # -------------------------
    tsne_patients = _subset_patients(patient_list, args.tsne_patients, seed=args.seed + 1)
    tsne_loader = _make_loader(
        patient_dirs=tsne_patients,
        window_samples=window_samples,
        fs=args.fs,
        seed=args.seed + 1,
        k_windows=args.k_windows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Optional saving
    fig_dir = args.fig_dir
    out_path = None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / f"tsne_{args.variant}_{args.split}.pdf"

    title_map = {
        "10m_cnn": "Continuous CNN Encoder (10-min)",
        "10m_vq":  "Discretized VQ Encoder (10-min)",
        "16s_cnn": "Continuous CNN Encoder (16-sec)",
        "16s_vq":  "Discretized VQ Encoder (16-sec)",
    }

    # plot_tsne should accept out_path=None gracefully; if your implementation requires a path,
    # keep fig_dir and provide out_path.
    plot_tsne(
        model=model,
        loader=tsne_loader,
        device=device,
        title=title_map[args.variant],
        out_path=str(out_path) if out_path is not None else None,
        seed=args.tsne_seed,
        perplexity=args.perplexity,
        show=(not args.no_show),
    )

    if out_path is not None:
        print(f"[OK] Saved t-SNE figure: {out_path}")

    # Run like:
    # python -m eval.clustering --variant 10m_cnn --ckpt checkpoints/CNN_tokenizer_10mins/checkpoint_50000.pth
    # python -m eval.clustering --variant 16s_vq  --ckpt checkpoints/VQ_tokenizer_16secs/checkpoint_50000.pth --fig_dir docs/cluster


if __name__ == "__main__":
    main()
