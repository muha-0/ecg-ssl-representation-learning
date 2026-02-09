
"""
Unified CLI for supervised evaluation on Icentia11k.

Supports:
- window length: 10 minutes OR 16 seconds
- training mode: linear probe (frozen encoder) OR fine-tune end-to-end
- init: random weights OR load encoder checkpoint path
- tokenizer: CNN patch tokenizer OR VQ tokenizer (needs codebook path)
- optional saving of trained weights (encoder + head/probe) to a user-provided path

Run examples:

# 10-min, CNN, load pretrained encoder, linear probe
python -m eval.eval --window 10m --tokenizer cnn --mode probe --ckpt checkpoints/CNN_tokenizer_10mins/checkpoint_50000.pth --save checkpoints/linear_probe_CNN_tokenizer_10mins/model.pth

# 16-sec, VQ, random init, fine-tune
python -m eval.eval --window 16s --tokenizer vq --mode finetune --codebook codebooks/icentia_codebook_256x160_16secs.pt --save checkpoints/fine_tuned_VQ_tokenizer_16secs/model.pth

# 16-sec, CNN, pretrained init, probe, and also report 10-min aggregated (fairness-style) metric
python -m eval.eval --window 16s --tokenizer cnn --mode probe --ckpt checkpoints/CNN_tokenizer_16secs/checkpoint_50000.pth --eval10m-agg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.utils import list_patients, list_record_bases
from data.split import split_patients
from data.seed import seed_worker

from data.window_dataset import IcentiaRhythmWindowDataset
from data.tenmin_eval_dataset import IcentiaTenMinForSixteenSecEval
from data.collate import collate_tenmin

from models.tokenizers import KMeansVQPatchTokenizer, ConvPatchTokenizer
from models.encoder import ECGEncoder
from codebooks.build_codebook import load_offline_codebook

from .plots import plot_probe_history
from .linear_probe import train_linear_probe
from .utils import eval_on_loader, count_pos, eval_16sec_model_on_10min_windows 
from train.fine_tune import train_end2end_classifier
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor")

# ---------------------------
# helpers
# ---------------------------

def _parse_window_arg(window: str, fs: int) -> tuple[int, int]:
    """
    window: "10m" or "16s" or raw seconds e.g. "600"
    returns (window_sec, window_samples)
    """
    w = window.strip().lower()
    if w in ("10m", "10min", "10mins"):
        sec = 10 * 60
    elif w in ("16s", "16sec", "16secs"):
        sec = 16
    else:
        # try numeric seconds
        sec = int(float(w))
    return sec, int(sec * fs)


def _infer_max_len(window_samples: int, patch_len: int, window_kind: str) -> int:
    """
    - 10m used 1024
    - 16s used (WINDOW_SAMPLES // patch_len) + 2
    """
    wk = window_kind.strip().lower()
    if wk in ("10m", "10min", "10mins"):
        return 1024
    # for 16s (and any custom), do the safer computed length
    return (window_samples // patch_len) + 2


def _build_tokenizer(
    tokenizer_kind: str,
    d_model: int,
    patch_len: int,
    codebook_path: str | None,
) -> torch.nn.Module:
    tk = tokenizer_kind.lower()
    if tk == "cnn":
        return ConvPatchTokenizer(d_model=d_model, patch_len=patch_len)

    if tk == "vq":
        if codebook_path is None:
            raise ValueError("--codebook is required when --tokenizer vq")
        centroids, _meta = load_offline_codebook(codebook_path)
        return KMeansVQPatchTokenizer(
            d_model=d_model,
            centroids=centroids,
            patch_len=patch_len,
            enforce_window_zscore=False,
            use_embedding_table=True,
            return_ids=False,
        )

    raise ValueError(f"Unknown tokenizer kind: {tokenizer_kind}")


def _maybe_load_encoder_ckpt(encoder: torch.nn.Module, ckpt_path: str | None, device: torch.device) -> None:
    if ckpt_path is None:
        print("Random Weights (no --ckpt provided)")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)  # allow raw state_dict or training dict
    encoder.load_state_dict(state, strict=True)
    step = ckpt.get("step", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"Loaded encoder checkpoint: {ckpt_path} (step={step})")


def _save_checkpoint(save_path: str, encoder: torch.nn.Module, head: torch.nn.Module | None, extra: dict | None = None) -> None:
    sp = Path(save_path)
    sp.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "encoder_state_dict": encoder.state_dict(),
    }
    if head is not None:
        payload["head_state_dict"] = head.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, str(sp))
    print(f"Saved checkpoint to: {sp}")


# ---------------------------
# main pipeline
# ---------------------------

def main():
    p = argparse.ArgumentParser("ECG eval (probe/finetune; 10m/16s; random/ckpt)")

    p.add_argument("--root", type=str, default="/data/ahmed/icentia11k", help="Icentia11k root directory")
    p.add_argument("--fs", type=int, default=250)

    # core axes
    p.add_argument("--window", type=str, required=True, help='Window: "10m" or "16s" (or seconds like "600")')
    p.add_argument("--mode", type=str, choices=["probe", "finetune"], required=True, help="Training mode")
    p.add_argument("--tokenizer", type=str, choices=["cnn", "vq"], required=True, help="Tokenizer type")
    p.add_argument("--ckpt", type=str, default=None, help="Encoder checkpoint path to load (omit for random init)")
    p.add_argument("--save", type=str, default=None, help="Where to save trained weights (encoder+head). Optional.")

    # vq needs codebook
    p.add_argument("--codebook", type=str, default=None, help="Codebook path (required for tokenizer=vq)")

    # training hyperparams
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)

    p.add_argument("--lr", type=float, default=1e-3, help="Probe LR (mode=probe)")
    p.add_argument("--lr-enc", type=float, default=1e-4, help="Encoder LR (mode=finetune)")
    p.add_argument("--lr-head", type=float, default=1e-3, help="Head LR (mode=finetune)")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--early-stop-metric", type=str, default="AUPRC", choices=["AUPRC", "AUROC", "loss"])
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ("yes", "true", "t", "1", "y"):
            return True
        if v in ("no", "false", "f", "0", "n"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")
    p.add_argument("--use-amp", type=str2bool, default=True)


    # labeling / sampling
    p.add_argument("--occ-thresh", type=float, default=0.05)
    p.add_argument("--min-covered", type=float, default=0.95)
    p.add_argument("--p-af-train", type=float, default=0.5)

    # evaluation
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--eval10m-agg", action="store_true",
                   help="For 16s runs: also evaluate 16s model aggregated on 10-min windows (mean p over 37 chunks).")

    args = p.parse_args()

    # constants
    ROOT = Path(args.root)
    FS = int(args.fs)

    window_sec, window_samples = _parse_window_arg(args.window, FS)
    patch_len = 160
    d_model = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # patients + split
    patients = list_patients(ROOT)
    print("num patients:", len(patients))
    if len(patients) > 0:
        print("example:", patients[0], "records:", len(list_record_bases(patients[0])))

    ssl_patients, train_patients, val_patients, test_patients = split_patients(patients)
    print("split sizes:", len(ssl_patients), len(train_patients), len(val_patients), len(test_patients))

    # build tokenizer + encoder
    tok = _build_tokenizer(
        tokenizer_kind=args.tokenizer,
        d_model=d_model,
        patch_len=patch_len,
        codebook_path=args.codebook,
    )

    max_len = _infer_max_len(window_samples=window_samples, patch_len=patch_len, window_kind=args.window)

    encoder = ECGEncoder(
        tokenizer=tok,
        d_model=d_model,
        max_len=max_len,
        use_cls=False,
    ).to(device)

    _maybe_load_encoder_ckpt(encoder, args.ckpt, device)

    # ---------------------------
    # build datasets + loaders based on 10m vs 16s
    # ---------------------------

    bs = int(args.batch_size)
    nw = int(args.num_workers)

    if window_sec == 10 * 60:
        # 10m window-level datasets (train/val/test all same type)
        sup_train_ds = IcentiaRhythmWindowDataset(
            patient_dirs=train_patients,
            window_samples=window_samples,
            fs=FS,
            seed=202,
            p_af=float(args.p_af_train),  # balanced for train
            occ_thresh=float(args.occ_thresh),
            min_covered=float(args.min_covered),
            augment=None,
            deterministic=False,
            k_per_patient=1,
        )
        sup_val_ds = IcentiaRhythmWindowDataset(
            patient_dirs=val_patients,
            window_samples=window_samples,
            fs=FS,
            seed=303,
            p_af=None,  # natural prevalence for eval
            occ_thresh=float(args.occ_thresh),
            min_covered=float(args.min_covered),
            augment=None,
            deterministic=True,
            k_per_patient=2,
        )
        sup_test_ds = IcentiaRhythmWindowDataset(
            patient_dirs=test_patients,
            window_samples=window_samples,
            fs=FS,
            seed=404,
            p_af=None,
            occ_thresh=float(args.occ_thresh),
            min_covered=float(args.min_covered),
            augment=None,
            deterministic=True,
            k_per_patient=2,
        )

        train_loader = DataLoader(
            sup_train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4
        )
        val_loader = DataLoader(
            sup_val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4
        )
        test_loader = DataLoader(
            sup_test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4
        )

        print("VAL:", count_pos(val_loader))
        print("TEST:", count_pos(test_loader))

        # ---------------------------
        # train: probe or finetune
        # ---------------------------
        if args.mode == "probe":
            probe, history = train_linear_probe(
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=int(args.epochs),
                lr=float(args.lr),
                wd=float(args.wd),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                early_stop_metric=args.early_stop_metric,
            )
            head = probe
        else:
            encoder, head, history = train_end2end_classifier(
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=int(args.epochs),
                lr_enc=float(args.lr_enc),
                lr_head=float(args.lr_head),
                wd=float(args.wd),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                early_stop_metric=args.early_stop_metric,
                use_amp=bool(args.use_amp),
            )

        plot_probe_history(history)

        # ---------------------------
        # test eval (window-level)
        # ---------------------------
        test_metrics = eval_on_loader(encoder, head, test_loader, device, threshold=float(args.threshold))

        print("\n=== TEST METRICS (window-level) ===")
        for k in ["n", "pos_rate", "loss", "AUROC", "AUPRC"]:
            print(f"{k:10s}: {test_metrics[k]}")
        print("\n=== Confusion @ {:.3f} ===".format(float(args.threshold)))
        print(f"TP={test_metrics['TP']}  FP={test_metrics['FP']}")
        print(f"TN={test_metrics['TN']}  FN={test_metrics['FN']}")

        # save if requested
        if args.save:
            _save_checkpoint(
                args.save,
                encoder=encoder,
                head=head,
                extra={
                    "window_sec": int(window_sec),
                    "tokenizer": args.tokenizer,
                    "mode": args.mode,
                    "init_ckpt": args.ckpt,
                },
            )

    else:
        # 16s case:
        # 1) train_loader uses 16s windows (balanced)
        # 2) val/test window-level loaders use 16s windows (natural prevalence)
        # 3) optional: 10-min aggregation evaluation uses TenMinForSixteenSecEval + collate_tenmin
        if window_sec != 16:
            raise ValueError(f"For now only 10m or 16s are supported. Got window_sec={window_sec}")

        # 16s window dataset for training
        sup_train16_ds = IcentiaRhythmWindowDataset(
            patient_dirs=train_patients,
            window_samples=window_samples,  # 16s samples
            fs=FS,
            seed=202,
            p_af=float(args.p_af_train),
            occ_thresh=float(args.occ_thresh),
            min_covered=float(args.min_covered),
            augment=None,
            deterministic=False,
            k_per_patient=1,
        )

        # 16s window dataset for window-level eval (val/test)
        sup_val16_ds = IcentiaRhythmWindowDataset(
            patient_dirs=val_patients,
            window_samples=window_samples,
            fs=FS,
            seed=303,
            p_af=None,
            occ_thresh=float(args.occ_thresh),
            min_covered=float(args.min_covered),
            augment=None,
            deterministic=True,
            k_per_patient=2,
        )
        sup_test16_ds = IcentiaRhythmWindowDataset(
            patient_dirs=test_patients,
            window_samples=window_samples,
            fs=FS,
            seed=404,
            p_af=None,
            occ_thresh=float(args.occ_thresh),
            min_covered=float(args.min_covered),
            augment=None,
            deterministic=True,
            k_per_patient=2,
        )

        train_loader = DataLoader(
            sup_train16_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4
        )
        val16_loader = DataLoader(
            sup_val16_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4
        )
        test16_loader = DataLoader(
            sup_test16_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True,
            worker_init_fn=seed_worker, persistent_workers=True, prefetch_factor=4
        )

        print("VAL16:", count_pos(val16_loader))
        print("TEST16:", count_pos(test16_loader))

        # optional: build 10-min aggregated evaluation loader (labels at 10-min, inputs are 37x16s chunks)
        tenmin_test_loader = None
        if args.eval10m_agg:
            WIN10_SAMPLES = FS * 592
            WIN16_SAMPLES = FS * 16

            tenmin_test_ds = IcentiaTenMinForSixteenSecEval(
                patient_dirs=test_patients,
                fs=FS,
                win10_samples=WIN10_SAMPLES,
                win16_samples=WIN16_SAMPLES,
                occ_thresh=float(args.occ_thresh),
                min_covered=float(args.min_covered),
                seed=404,
                cache_size=64,
                deterministic=True,
                k_per_patient=2,
            )

            tenmin_test_loader = DataLoader(
                tenmin_test_ds,
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                pin_memory=True,
                collate_fn=collate_tenmin,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                prefetch_factor=4,
            )
            print("TEST10(agg):", count_pos(tenmin_test_loader))

        # ---------------------------
        # train: probe or finetune (val uses 16s window-level)
        # ---------------------------
        if args.mode == "probe":
            probe, history = train_linear_probe(
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val16_loader,
                device=device,
                epochs=int(args.epochs),
                lr=float(args.lr),
                wd=float(args.wd),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                early_stop_metric=args.early_stop_metric,
            )
            head = probe
        else:
            encoder, head, history = train_end2end_classifier(
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val16_loader,
                device=device,
                epochs=int(args.epochs),
                lr_enc=float(args.lr_enc),
                lr_head=float(args.lr_head),
                wd=float(args.wd),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                early_stop_metric=args.early_stop_metric,
                use_amp=bool(args.use_amp),
            )

        plot_probe_history(history)

        # ---------------------------
        # test eval: 16s window-level
        # ---------------------------
        test_metrics_16 = eval_on_loader(encoder, head, test16_loader, device, threshold=float(args.threshold))

        print("\n=== TEST METRICS (16s window-level) ===")
        for k in ["n", "pos_rate", "loss", "AUROC", "AUPRC"]:
            print(f"{k:10s}: {test_metrics_16[k]}")
        print("\n=== Confusion @ {:.3f} ===".format(float(args.threshold)))
        print(f"TP={test_metrics_16['TP']}  FP={test_metrics_16['FP']}")
        print(f"TN={test_metrics_16['TN']}  FN={test_metrics_16['FN']}")

        # optional: 10-min aggregation
        if tenmin_test_loader is not None:
            agg = eval_16sec_model_on_10min_windows(encoder, head, tenmin_test_loader, device)
            print("\n=== TEST METRICS (10-min aggregated from 16s; mean p) ===")
            for k in ["n", "pos_rate", "AUROC", "AUPRC"]:
                print(f"{k:10s}: {agg[k]}")

        # save if requested
        if args.save:
            extra = {
                "window_sec": int(window_sec),
                "tokenizer": args.tokenizer,
                "mode": args.mode,
                "init_ckpt": args.ckpt,
            }
            if tenmin_test_loader is not None:
                extra["eval10m_agg"] = True
            _save_checkpoint(args.save, encoder=encoder, head=head, extra=extra)


if __name__ == "__main__":
    main()