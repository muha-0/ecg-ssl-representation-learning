from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train.config import SSLConfig
from data.utils import list_patients, list_record_bases
from data.split import split_patients
from data.pair_dataset import IcentiaPatientPairDataset
from data.collate import collate_patient_pairs
from data.seed import seed_worker
from data.augment import make_augment
from data.preprocess import preprocess_to_memmap

from codebooks.build_codebook import load_offline_codebook  # (build stays optional)
from models.tokenizers import KMeansVQPatchTokenizer, ConvPatchTokenizer
from models.encoder import ECGEncoder

from train.engine import train_ssl
from eval.plots import plot_loss
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor")

def parse_args():
    p = argparse.ArgumentParser()

    # config overrides
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--backend", type=str, choices=["wfdb", "npy"], default=None)
    p.add_argument("--processed_root", type=str, default=None)

    p.add_argument("--window_sec", type=int, default=None)
    p.add_argument("--fs", type=int, default=None)
    p.add_argument("--patch_len", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--aug_preset", type=str, choices=["10min", "16s"], default=None)

    p.add_argument("--tokenizer", type=str, choices=["vq", "cnn"], default="vq")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--use_cls", action="store_true")

    p.add_argument("--codebook_path", type=str, default=None)

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--exp_dir", type=str, default=None)

    # utilities
    p.add_argument("--preprocess", action="store_true")  # for backend=npy setup, then exit

    return p.parse_args()


def apply_overrides(cfg: SSLConfig, args) -> SSLConfig:
    # Paths
    if args.root is not None:
        cfg.root = Path(args.root)

    if args.backend is not None:
        cfg.backend = args.backend

    if args.processed_root is not None:
        cfg.processed_root = Path(args.processed_root)

    # Core params
    if args.fs is not None:
        cfg.fs = int(args.fs)
    if args.window_sec is not None:
        cfg.window_sec = int(args.window_sec)
    if args.patch_len is not None:
        cfg.patch_len = int(args.patch_len)

    # Loader params
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    # Aug preset
    if args.aug_preset is not None:
        cfg.aug_preset = args.aug_preset

    # Codebook path
    if args.codebook_path is not None:
        cfg.codebook_path = args.codebook_path

    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(SSLConfig(), args)

    # backend sanity
    if cfg.backend == "npy" and cfg.processed_root is None:
        raise ValueError("backend='npy' requires cfg.processed_root (or pass --processed_root).")

    ROOT = cfg.root
    PROCESSED_ROOT = cfg.processed_root if cfg.backend == "npy" else None

    print("===== Fetching patient list and splitting =====")
    patients = list_patients(ROOT)
    print("num patients:", len(patients))
    print("example:", patients[0], "records:", len(list_record_bases(patients[0])))

    ssl_patients, train_patients, val_patients, test_patients = split_patients(patients, seed=cfg.seed)
    print(
        f"SSL={len(ssl_patients)} | Train={len(train_patients)} | "
        f"Val={len(val_patients)} | Test={len(test_patients)}"
    )
    print("===== Done =====\n")

    # One-time preprocessing for npy backend
    if args.preprocess:
        if PROCESSED_ROOT is None:
            raise ValueError("--preprocess only makes sense with backend='npy' and processed_root set.")
        preprocess_to_memmap(
            patient_dirs=patients,
            root=ROOT,
            save_root=PROCESSED_ROOT,
            fs=cfg.fs,
        )
        print("Preprocessing done. Exiting.")
        return

    print("===== Building dataset and dataloader =====")
    augment = make_augment(cfg.aug_preset, fs=cfg.fs)

    ssl_ds = IcentiaPatientPairDataset(
        patient_dirs=ssl_patients,
        window_samples=cfg.window_samples,
        fs=cfg.fs,
        augment=augment,
        seed=123,
        cache_size=64,
        root=ROOT,
        processed_root=str(PROCESSED_ROOT) if PROCESSED_ROOT else None,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    ssl_loader = DataLoader(
        ssl_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=collate_patient_pairs,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(f"window_sec={cfg.window_sec} | window_samples={cfg.window_samples} | max_len={cfg.max_len}")
    print("===== Done =====\n")

    # ---- Tokenizer + model ----
    if args.tokenizer == "vq":
        print("===== Loading codebook =====")
        centroids, meta = load_offline_codebook(cfg.codebook_path)
        print(f"Centroids: {tuple(centroids.shape)} | meta.window_samples={meta.get('window_samples')}")
        print("===== Done =====\n")

        tok = KMeansVQPatchTokenizer(
            d_model=args.d_model,
            centroids=centroids,
            patch_len=cfg.patch_len,
            enforce_window_zscore=False,
            use_embedding_table=True,
            return_ids=False,
        )
    else:
        tok = ConvPatchTokenizer(d_model=args.d_model, patch_len=cfg.patch_len)

    model = ECGEncoder(
        tokenizer=tok,
        d_model=args.d_model,
        max_len=cfg.max_len,
        use_cls=args.use_cls,
    )

    exp_dir = args.exp_dir
    if exp_dir:
        os.makedirs(exp_dir, exist_ok=True)

    print("===== Training =====")
    stats = train_ssl(
        model,
        ssl_loader,
        epochs=args.epochs,
        max_steps=args.max_steps,
        exp_dir=exp_dir,
    )
    print("===== Done =====\n")

    plot_loss(stats["losses"], title=f"{args.tokenizer.upper()} | {cfg.aug_preset} | {cfg.window_sec}s")

# python -m train.train_ssl --backend wfdb --aug_preset 10min --window_sec 600 --tokenizer vq
# python -m train.train_ssl --backend npy --processed_root /data/ahmed/icentia_processed --preprocess
# python -m train.train_ssl --backend npy --processed_root /data/ahmed/icentia_processed --aug_preset 16s --window_sec 16 --tokenizer vq --codebook_path codebooks/icentia_codebook_256x160_16secs.pt

if __name__ == "__main__":
    main()