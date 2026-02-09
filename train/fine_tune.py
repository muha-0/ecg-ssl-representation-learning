import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.probe import LinearProbe
from eval.utils import eval_probe_with_loss

def train_end2end_classifier(
    encoder,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr_enc=1e-4,          # encoder LR (smaller)
    lr_head=1e-3,         # head LR (bigger)
    wd=1e-4,
    patience=3,
    min_delta=1e-4,
    early_stop_metric="AUPRC",
    use_amp=True,
):
    assert early_stop_metric in ("AUPRC", "AUROC", "loss")
    encoder = encoder.to(device)

    # infer dim for head
    xb = next(iter(train_loader))["x"].to(device).float()
    with torch.no_grad():
        h, _ = encoder(xb, return_h=True)
    in_dim = h.shape[-1]

    head = LinearProbe(in_dim=in_dim, n_classes=2).to(device)

    # Unfreeze encoder
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True

    # param groups (different LRs)
    opt = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": lr_enc},
            {"params": head.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_AUROC": [],
        "train_AUPRC": [],
        "val_AUROC": [],
        "val_AUPRC": [],
    }

    # best tracking
    if early_stop_metric == "loss":
        best_val = float("inf")
        better = lambda cur, best: cur <= best - min_delta
    else:
        best_val = -float("inf")
        better = lambda cur, best: cur >= best + min_delta

    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        encoder.train()
        head.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"E2E Epoch {ep}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                h, _ = encoder(x, return_h=True)     # NO no_grad
                logits = head(h)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # Eval (encoder+head)
        train_metrics = eval_probe_with_loss(encoder, head, train_loader, device)
        val_metrics   = eval_probe_with_loss(encoder, head, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_AUROC"].append(train_metrics["AUROC"])
        history["train_AUPRC"].append(train_metrics["AUPRC"])
        history["val_AUROC"].append(val_metrics["AUROC"])
        history["val_AUPRC"].append(val_metrics["AUPRC"])

        cur = val_metrics[early_stop_metric]
        cur_improved = (not np.isnan(cur)) and better(cur, best_val)

        if cur_improved:
            best_val = cur
            best_state = {
                "encoder": copy.deepcopy(encoder.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
            best_epoch = ep
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"epoch={ep:02d} "
            f"train_loss={avg_train_loss:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train AUROC={train_metrics['AUROC']:.4f} AUPRC={train_metrics['AUPRC']:.4f} | "
            f"val AUROC={val_metrics['AUROC']:.4f} AUPRC={val_metrics['AUPRC']:.4f} | "
            f"best {early_stop_metric}={best_val:.4f} (epoch {best_epoch}) "
            f"patience={bad_epochs}/{patience}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {ep}. Best epoch was {best_epoch}.")
            break

    # restore best
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])

    return encoder, head, history