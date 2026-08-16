import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.signal
import wfdb
from functools import lru_cache
from data.utils import list_record_bases
from .rhythm import build_rhythm_intervals_from_ann, label_window_strict

class IcentiaTenMinForSixteenSecEval(Dataset):
    """
    Produces examples at the 16-sec level, but returns the 37x16s chunks inside each 16-sec window.

    Label y_10m is computed with your SAME strict window labeling rule:
      - coverage >= min_covered
      - AF positive if AFIB or AFL occupancy > occ_thresh (e.g., 0.05)

    Output per item:
      {
        "x16": FloatTensor [NCHUNKS, T16]  (z-scored per 16s chunk)
        "y10": LongTensor scalar (0/1)
        "patient_id", "rec", "w_start"
      }
    """

    def __init__(
        self,
        patient_dirs,
        fs=250,
        win10_samples=250*600,
        win16_samples=250*16,
        occ_thresh=0.05,
        min_covered=0.95,
        seed=0,
        cache_size=64,
        deterministic=True,
        # sampling:
        k_per_patient=2,
        max_tries=120,
    ):
        self.patient_dirs = list(patient_dirs)
        self.fs = fs
        self.win10 = int(win10_samples)
        self.win16 = int(win16_samples)
        self.occ_thresh = float(occ_thresh)
        self.min_covered = float(min_covered)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.k_per_patient = int(k_per_patient)
        self.max_tries = int(max_tries)

        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]

        self._cache_size = int(cache_size)
        self._init_record_cache()

        # number of 16s chunks inside 10m if non-overlapping
        self.n_chunks = self.win10 // self.win16
        assert self.n_chunks > 0 and self.n_chunks * self.win16 == self.win10, \
            f"win10 ({self.win10}) must be divisible by win16 ({self.win16})."

    def __len__(self):
        return len(self.patient_dirs) * self.k_per_patient

    def _rng_for_patient_rep(self, patient_idx, rep):
        return np.random.default_rng(self.seed + 1000003 * int(patient_idx) + 9176 * int(rep))

    def _init_record_cache(self):
        # same filter settings you used in dataset
        lowcut = 0.5
        highcut = 40.0
        nyq = 0.5 * self.fs
        b, a = scipy.signal.butter(4, [lowcut/nyq, highcut/nyq], btype="bandpass")

        @lru_cache(maxsize=self._cache_size)
        def _cached_load(rec_base_str: str):
            sig, _ = wfdb.rdsamp(rec_base_str)
            sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
            sig = scipy.signal.filtfilt(b, a, sig).astype(np.float32)

            sig_len = len(sig)
            ann = wfdb.rdann(rec_base_str, extension="atr")
            intervals = build_rhythm_intervals_from_ann(ann, sig_len)
            return sig, intervals

        self._load_record_cached = _cached_load

    def _sample_10min_window(self, sig, intervals, rng):
        sig_len = len(sig)
        if sig_len <= self.win10:
            return None

        # sample a window start
        for _ in range(self.max_tries):
            w_start = int(rng.integers(0, sig_len - self.win10))
            w_end = w_start + self.win10

            lab = label_window_strict(
                intervals, w_start, w_end,
                thresh=self.occ_thresh,
                min_covered=self.min_covered
            )
            if lab is None:
                continue  # unlabeled, skip

            # window label: AF vs N
            y10 = 1 if (lab in ("AFIB", "AFL")) else 0
            x10 = sig[w_start:w_end]
            if np.isnan(x10).any() or np.std(x10) < 1e-4:
                continue

            return w_start, y10

        return None

    def __getitem__(self, idx):
        patient_idx = int(idx) // self.k_per_patient
        rep = int(idx) % self.k_per_patient
        rng = self._rng_for_patient_rep(patient_idx, rep) if self.deterministic else np.random.default_rng()

        recs = self.patient_records[patient_idx]
        if not recs:
            # fallback
            ridx = int(rng.integers(0, len(self)))
            return self.__getitem__(ridx)

        for _ in range(10):
            rec_path = recs[int(rng.integers(0, len(recs)))]
            sig, intervals = self._load_record_cached(str(rec_path))

            out = self._sample_10min_window(sig, intervals, rng)
            if out is None:
                continue
            w_start, y10 = out

            # split into 16s chunks (non-overlapping within the 10m window)
            chunks = []
            for j in range(self.n_chunks):
                s = w_start + j * self.win16
                e = s + self.win16
                x = sig[s:e].astype(np.float32)

                # per-chunk z-score (match supervised pipeline)
                x = (x - x.mean()) / (x.std() + 1e-6)
                chunks.append(x)

            x16 = np.stack(chunks, axis=0)  # [37, T16]

            return {
                "x": torch.from_numpy(x16),  # float32
                "y": torch.tensor(y10, dtype=torch.long),
                "patient_id": self.patient_dirs[patient_idx].name,
                "rec": rec_path.name,
                "w_start": int(w_start),
                "rep": int(rep),
            }

        # fallback: resample
        ridx = int(rng.integers(0, len(self)))
        return self.__getitem__(ridx)

