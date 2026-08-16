import torch
import numpy as np
import scipy.signal
import wfdb
from .utils import list_record_bases
from functools import lru_cache

class IcentiaPatientWindowsDataset(torch.utils.data.Dataset):
    def __init__(self, patient_dirs, window_samples, fs=250, seed=0, cache_size=64, k_windows=5):
        self.patient_dirs = list(patient_dirs)
        self.window_samples = int(window_samples)
        self.fs = fs
        self.k_windows = int(k_windows)
        

        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]

        self._cache_size = int(cache_size)
        self.seed = int(seed)
        self._init_record_cache()

    def __len__(self):
        return len(self.patient_dirs)

    def _init_record_cache(self):
        lowcut = 0.5
        highcut = 40.0
        nyq = 0.5 * self.fs
        b, a = scipy.signal.butter(4, [lowcut/nyq, highcut/nyq], btype="bandpass")

        @lru_cache(maxsize=self._cache_size)
        def _cached_load(rec_base_str: str):
            sig, _ = wfdb.rdsamp(rec_base_str)
            sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
            sig = scipy.signal.filtfilt(b, a, sig).astype(np.float32)
            return sig

        self._load_record_cached = _cached_load

    def _sample_window(self, sig, rng, max_tries=80):
        sig_len = len(sig)
        if sig_len <= self.window_samples:
            return None
        for _ in range(max_tries):
            s = int(rng.integers(0, sig_len - self.window_samples))
            e = s + self.window_samples
            x = sig[s:e]
            if np.isnan(x).any() or np.std(x) < 1e-4:
                continue
            # per-window z-score (match your augment preprocess)
            x = (x - x.mean()) / (x.std() + 1e-6)
            return x.astype(np.float32)
        return None

    def __getitem__(self, idx):
        recs = self.patient_records[idx]
        if not recs:
            ridx = idx  # deterministic fallback
            return self.__getitem__(ridx)

        K = self.k_windows
        # deterministic RNG per patient
        rng_p = np.random.default_rng(self.seed + 1000003 * int(idx))

        if len(recs) >= K:
            chosen = rng_p.choice(len(recs), size=K, replace=False)
        else:
            chosen = rng_p.choice(len(recs), size=K, replace=True)

        xs = []
        for wi, j in enumerate(chosen):
            rng_w = np.random.default_rng(self.seed + 1000003 * int(idx) + 9176 * int(wi))

            rec_path = recs[int(j)]
            sig = self._load_record_cached(str(rec_path))

            x = self._sample_window(sig, rng_w)   # pass rng_w
            if x is None:
                # deterministic fallback attempts
                ok = False
                for t in range(5):
                    alt = recs[int(rng_w.integers(0, len(recs)))]
                    sig2 = self._load_record_cached(str(alt))
                    x = self._sample_window(sig2, rng_w)
                    if x is not None:
                        ok = True
                        break
                if not ok:
                    break

            xs.append(torch.from_numpy(x))

        if len(xs) == 0:
            # deterministic fallback: return zeros or resample fixed
            x0 = np.zeros((self.window_samples,), dtype=np.float32)
            xs = [torch.from_numpy(x0)]

        while len(xs) < K:
            xs.append(xs[-1].clone())

        return {"patient_id": self.patient_dirs[idx].name, "x": torch.stack(xs, dim=0)}