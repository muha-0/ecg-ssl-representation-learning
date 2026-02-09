import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
import wfdb
import scipy.signal
from data.utils import list_record_bases

def build_offline_codebook_per_patient_window_z(
    patient_dirs,
    patch_len=160,
    n_tokens=256,
    patches_per_patient=36,
    window_samples=250 * 600,          # default = 10 minutes @250Hz; pass WINDOW_SAMPLES
    fs=250,
    seed=42,
    save_path="codebooks/icentia_codebook_256x160.pt",
    max_patients=None,                 # for quick debug runs
    bandpass=(0.5, 40.0),              # matches the pipeline
    max_record_pick_tries=10,
    max_window_tries_per_patient=80,
    max_patch_tries_per_window=2000,
    std_thresh=1e-4,
):
    """
    For each patient:
      - choose ONE random record
      - load + bandpass filter whole record ONCE
      - repeat until you have patches_per_patient patches:
          * sample a random WINDOW (length window_samples)
          * window z-score (mean/std over that window)
          * sample random PATCHES (length patch_len) within that window
          * skip degenerate patches: NaNs or std < std_thresh
    Then fit MiniBatchKMeans on all patches.
    """

    rng = np.random.default_rng(seed)
    patient_dirs = list(patient_dirs)
    if max_patients is not None:
        patient_dirs = patient_dirs[: int(max_patients)]

    # --- bandpass filter once ---
    lowcut, highcut = bandpass
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(4, [lowcut / nyq, highcut / nyq], btype="bandpass")

    patches_all = []
    n_pat_used = 0
    n_pat_skipped = 0

    for pi, pd in enumerate(patient_dirs):
        recs = list_record_bases(pd)
        if not recs:
            n_pat_skipped += 1
            continue

        # 1) choose ONE random record for this patient, load+filter once
        sig = None
        chosen_rec = None
        for _ in range(max_record_pick_tries):
            chosen_rec = recs[int(rng.integers(0, len(recs)))]
            try:
                x, _ = wfdb.rdsamp(str(chosen_rec))
                x = x[:, 0].astype(np.float32) if x.ndim == 2 else x.squeeze().astype(np.float32)
                x = scipy.signal.filtfilt(b, a, x).astype(np.float32)

                if len(x) < max(patch_len, window_samples):
                    continue
                if np.isnan(x).any() or float(np.std(x)) < std_thresh:
                    continue

                sig = x
                break
            except Exception:
                sig = None

        if sig is None:
            n_pat_skipped += 1
            continue

        sig_len = len(sig)
        if sig_len < window_samples:
            n_pat_skipped += 1
            continue

        # 2) collect patches_per_patient from windows within this same record
        pat_patches = []
        window_tries = 0

        while len(pat_patches) < patches_per_patient and window_tries < max_window_tries_per_patient:
            window_tries += 1

            # sample a random window
            ws = int(rng.integers(0, sig_len - window_samples))
            we = ws + window_samples
            win = sig[ws:we]

            if np.isnan(win).any():
                continue

            wstd = float(np.std(win))
            if wstd < std_thresh:
                continue

            # window z-score
            win = (win - float(np.mean(win))) / (wstd + 1e-6)
            win = win.astype(np.float32)

            # sample random patches inside this window
            # (we oversample attempts because we may skip degenerate patches)
            patch_tries = 0
            while (
                len(pat_patches) < patches_per_patient
                and patch_tries < max_patch_tries_per_window
            ):
                patch_tries += 1

                s = int(rng.integers(0, window_samples - patch_len))
                patch = win[s : s + patch_len]

                # skip degenerate patches
                if np.isnan(patch).any():
                    continue
                if float(np.std(patch)) < std_thresh:
                    continue

                pat_patches.append(patch)

        if len(pat_patches) < patches_per_patient:
            n_pat_skipped += 1
            continue

        patches_all.append(np.stack(pat_patches[:patches_per_patient], axis=0))  # [P, patch_len]
        n_pat_used += 1

        if (pi + 1) % 200 == 0:
            print(
                f"[{pi+1}/{len(patient_dirs)}] patients processed | "
                f"used={n_pat_used}, skipped={n_pat_skipped}, "
                f"patches={n_pat_used*patches_per_patient}"
            )

    if n_pat_used == 0:
        raise RuntimeError("No patients produced valid patches. Check paths / WFDB reads / thresholds.")

    X = np.concatenate(patches_all, axis=0)  # [n_pat_used*patches_per_patient, patch_len]
    print("Final patch matrix:", X.shape)

    km = MiniBatchKMeans(
        n_clusters=n_tokens,
        batch_size=4096,
        n_init=5,
        random_state=seed,
        reassignment_ratio=0.01,
    )
    km.fit(X)

    centroids = torch.from_numpy(km.cluster_centers_).float()

    payload = {
        "centroids": centroids,
        "meta": {
            "patch_len": int(patch_len),
            "n_tokens": int(n_tokens),
            "patches_per_patient": int(patches_per_patient),
            "window_samples": int(window_samples),
            "seed": int(seed),
            "fs": int(fs),
            "bandpass_low": float(lowcut),
            "bandpass_high": float(highcut),
            "std_thresh": float(std_thresh),
            "n_patients_requested": int(len(patient_dirs)),
            "n_patients_used": int(n_pat_used),
            "n_patients_skipped": int(n_pat_skipped),
            "n_patches_fit": int(X.shape[0]),
            "record_sampling": "one_random_record_per_patient",
            "sampling": "random_window_then_random_patch",
            "normalization": "window_zscore_then_patch",
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(payload, save_path)

    return centroids


def load_offline_codebook(save_path="codebooks/icentia_codebook_256x160.pt"):
    payload = torch.load(save_path, map_location="cpu")
    return payload["centroids"], payload.get("meta", {})