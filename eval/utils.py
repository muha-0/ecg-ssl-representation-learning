from .metrics import auc_roc, average_precision
import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def eval_probe_with_loss(encoder, probe, loader, device):
    """
    Computes:
      - mean CE loss
      - AUROC, AUPRC (based on prob of class 1)
    """
    encoder.eval()
    probe.eval()

    ys, scores = [], []
    losses = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True)

        h, _ = encoder(x, return_h=True)
        logits = probe(h)

        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())

        prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
        ys.extend(y.detach().cpu().numpy().tolist())
        scores.extend(prob)

    return {
        "loss": float(np.mean(losses)) if len(losses) else float("nan"),
        "AUROC": auc_roc(ys, scores),
        "AUPRC": average_precision(ys, scores),
    }

@torch.no_grad()
def eval_on_loader(encoder, probe, loader, device, threshold=0.5):
    encoder.eval()
    probe.eval()

    ys = []
    scores = []
    losses = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True)

        h, _ = encoder(x, return_h=True)
        logits = probe(h)

        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))

        prob1 = torch.softmax(logits, dim=-1)[:, 1]
        ys.append(y.detach().cpu().numpy())
        scores.append(prob1.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_score = np.concatenate(scores).astype(np.float64)

    # If you want sklearn metrics, uncomment:
    # from sklearn.metrics import roc_auc_score, average_precision_score
    # auroc = roc_auc_score(y_true, y_score)
    # auprc = average_precision_score(y_true, y_score)

    # Otherwise reuse your implementations already defined:
    auroc = auc_roc(y_true, y_score)
    auprc = average_precision(y_true, y_score)

    # Confusion matrix at threshold
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    out = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "pos_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    return out

@torch.no_grad()
def eval_16sec_model_on_10min_windows(
    encoder16,
    head16,            # LinearProbe or your fine-tuned head
    loader10,
    device,
):
    """
    Returns AUROC/AUPRC at window level for the 16s model using mean(p) aggregation.
    """
    encoder16.eval()
    head16.eval()

    all_y = []
    all_score = []

    for batch in loader10:
        x16 = batch["x"].to(device).float()   # [B, 37, T16]
        y10 = batch["y"].to(device)          # [B]

        B, N, T = x16.shape
        xflat = x16.view(B * N, T)             # [B*37, T16]

        h, _ = encoder16(xflat, return_h=True)     # h: [B*37, D]
        logits = head16(h)                          # [B*37, 2]
        p = torch.softmax(logits, dim=-1)[:, 1]     # [B*37]

        p = p.view(B, N)                            # [B, 37]
        p_mean = p.mean(dim=1)                      # [B]  <<==== your chosen aggregation

        all_y.append(y10.detach().cpu().numpy())
        all_score.append(p_mean.detach().cpu().numpy())

    y_true = np.concatenate(all_y).astype(np.int64)
    y_score = np.concatenate(all_score).astype(np.float64)

    return {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "AUROC": float(auc_roc(y_true, y_score)),
        "AUPRC": float(average_precision(y_true, y_score)),
    }

def count_pos(loader):
    n = 0; p = 0
    for b in loader:
        y = b["y"].cpu().numpy()
        n += len(y)
        p += int((y == 1).sum())
    return n, p, p / max(1, n)