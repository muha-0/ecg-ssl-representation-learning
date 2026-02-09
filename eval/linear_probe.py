import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
from models.probe import LinearProbe
from .utils import eval_probe_with_loss

def train_linear_probe(
    encoder,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-3,
    wd=1e-4,
    patience=3,
    min_delta=1e-4,
    early_stop_metric="AUPRC",  # "AUPRC" is best for imbalanced AF
):
    """
    Early stopping:
      stops if val metric doesn't improve by >= min_delta for `patience` epochs.

    Returns:
      probe (restored to best epoch), history
    """
    assert early_stop_metric in ("AUPRC", "AUROC", "loss")
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # infer dim
    xb = next(iter(train_loader))["x"].to(device).float()
    with torch.no_grad():
        h, _ = encoder(xb, return_h=True)
    in_dim = h.shape[-1]

    probe = LinearProbe(in_dim=in_dim, n_classes=2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_AUROC": [],
        "train_AUPRC": [],
        "val_AUROC": [],
        "val_AUPRC": [],
    }

    # Tracking best
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
        probe.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True)

            with torch.no_grad():
                h, _ = encoder(x, return_h=True)

            logits = probe(h)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # Eval (with loss)
        train_metrics = eval_probe_with_loss(encoder, probe, train_loader, device)
        val_metrics   = eval_probe_with_loss(encoder, probe, val_loader, device)

        history["train_loss"].append(avg_train_loss)          # training-loop loss
        history["val_loss"].append(val_metrics["loss"])       # proper eval loss
        history["train_AUROC"].append(train_metrics["AUROC"])
        history["train_AUPRC"].append(train_metrics["AUPRC"])
        history["val_AUROC"].append(val_metrics["AUROC"])
        history["val_AUPRC"].append(val_metrics["AUPRC"])

        # Early stopping check
        cur = val_metrics[early_stop_metric]
        if np.isnan(cur):
            # If metric is nan, don't early stop on it; just keep going
            cur_improved = False
        else:
            cur_improved = better(cur, best_val)

        if cur_improved:
            best_val = cur
            best_state = copy.deepcopy(probe.state_dict())
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
            print(f"Early stopping triggered at epoch {ep}. Best epoch was {best_epoch}.")
            break

    # Restore best
    if best_state is not None:
        probe.load_state_dict(best_state)

    return probe, history