import matplotlib.pyplot as plt
import numpy as np
from .t_sne import get_tsne_embeddings

def plot_loss(losses, title="SSL loss", save_path="loss_curve.png", show=False):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("InfoNCE loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def plot_probe_history(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))

    # ---- Loss ----
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss (loop)")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Loss vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()

    # ---- AUROC ----
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_AUROC"], marker="o", label="Train AUROC")
    plt.plot(epochs, history["val_AUROC"], marker="o", label="Val AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("AUROC vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()

    # ---- AUPRC ----
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_AUPRC"], marker="o", label="Train AUPRC")
    plt.plot(epochs, history["val_AUPRC"], marker="o", label="Val AUPRC")
    plt.xlabel("Epoch")
    plt.ylabel("AUPRC")
    plt.title("AUPRC vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_tsne(model, loader, device, title, out_path,
                     seed=42, perplexity=15, s=45, alpha=0.8, show=True):
    Z, L, uniq_p = get_tsne_embeddings(model, loader, device, seed=seed, perplexity=perplexity)

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

    sc = ax.scatter(
        Z[:, 0], Z[:, 1],
        c=L,
        cmap="tab20",
        s=s,
        alpha=alpha,
        edgecolors="none",
    )

    ax.set_title(title, fontweight="normal", pad=10)
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.grid(True, alpha=0.15, linestyle="--")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()