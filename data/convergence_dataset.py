# data/debug_datasets.py
from __future__ import annotations

from torch.utils.data import Dataset
from functools import lru_cache
import numpy as np
import torch
import scipy.signal
import wfdb

from data.utils import list_record_bases


class DeterministicPatientWindows(Dataset):
    """
    Debug/test fixture dataset:
    For each patient idx, deterministically picks K records + K windows, returns [K, T].
    Useful for convergence/repro checks (NOT for main evaluation).
    """

    def __init__(self, patient_dirs, window_samples, fs=250, seed=1234, k_windows=3, cache_size=64):
        self.patient_dirs = list(patient_dirs)
        self.window_samples = int(window_samples)
        self.fs = int(fs)
        self.seed = int(seed)
        self.k_windows = int(k_windows)

        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]
        self._cache_size = int(cache_size)
        self._init_record_cache()

    def __len__(self):
        return len(self.patient_dirs)

    def _init_record_cache(self):
        lowcut = 0.5
        highcut = 40.0
        nyq = 0.5 * self.fs
        b, a = scipy.signal.butter(4, [lowcut / nyq, highcut / nyq], btype="bandpass")

        @lru_cache(maxsize=self._cache_size)
        def _cached_load(rec_base_str: str):
            sig, _ = wfdb.rdsamp(rec_base_str)
            sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
            sig = scipy.signal.filtfilt(b, a, sig).astype(np.float32)
            return sig

        self._load_record_cached = _cached_load

    def _sample_window(self, sig, rng, max_tries=200):
        sig_len = len(sig)
        if sig_len <= self.window_samples:
            return None
        for _ in range(max_tries):
            s = int(rng.integers(0, sig_len - self.window_samples))
            e = s + self.window_samples
            x = sig[s:e]
            if np.isnan(x).any() or np.std(x) < 1e-4:
                continue
            x = (x - x.mean()) / (x.std() + 1e-6)
            return x.astype(np.float32)
        return None

    def __getitem__(self, idx):
        recs = self.patient_records[idx]
        pid = self.patient_dirs[idx].name

        rng_p = np.random.default_rng(self.seed + 1000003 * int(idx))

        if not recs:
            x0 = np.zeros((self.window_samples,), dtype=np.float32)
            xk = np.stack([x0] * self.k_windows, axis=0)
            return {"patient_id": pid, "x": torch.from_numpy(xk)}

        K = self.k_windows
        chosen = rng_p.choice(len(recs), size=K, replace=(len(recs) < K))

        xs = []
        for wi, j in enumerate(chosen):
            rng_w = np.random.default_rng(self.seed + 1000003 * int(idx) + 9176 * int(wi))
            rec_path = recs[int(j)]
            sig = self._load_record_cached(str(rec_path))
            x = self._sample_window(sig, rng_w)
            if x is None:
                x = np.zeros((self.window_samples,), dtype=np.float32)
            xs.append(torch.from_numpy(x))

        return {"patient_id": pid, "x": torch.stack(xs, dim=0)}  # [K, T]
