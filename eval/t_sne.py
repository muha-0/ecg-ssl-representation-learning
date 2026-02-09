from sklearn.manifold import TSNE
import numpy as np
import torch
def get_tsne_embeddings(model, loader, device, seed=42, perplexity=15):
    model.eval()
    all_h, all_pid = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            h, _ = model(x, return_h=True)
            all_h.append(h.detach().cpu().numpy())
            all_pid.extend(batch["patient_ids"])

    H = np.concatenate(all_h, axis=0)
    pid = np.array(all_pid)

    # Map patient_id -> integer labels
    uniq_p = sorted(set(pid.tolist()))
    pid2i = {p: i for i, p in enumerate(uniq_p)}
    labels = np.array([pid2i[p] for p in pid], dtype=int)

    # Safe perplexity: must be < n_samples
    n = H.shape[0]
    p = min(perplexity, max(2, (n - 1) // 3))
    p = min(p, n - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=p,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    z = tsne.fit_transform(H)
    return z, labels, uniq_p