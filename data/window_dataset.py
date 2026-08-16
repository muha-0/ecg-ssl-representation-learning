from torch.utils.data import Dataset
from functools import lru_cache
import scipy.signal
import wfdb
import numpy as np
import torch
from data.rhythm import label_window_strict, build_rhythm_intervals_from_ann
from data.utils import list_record_bases

LABEL2ID_BIN = {"N": 0, "AF": 1}  # AF = AFIB or AFL

class IcentiaRhythmWindowDataset(Dataset):
    """
    Supervised dataset for rhythm classification:
    - Samples a single window per __getitem__
    - Labels using strict coverage to ignore OTHER/UNLABELED
    - Supports binary AF vs N by default (AF = AFIB or AFL)
    """

    def __init__(
        self,
        patient_dirs,
        window_samples,
        fs=250,
        seed=0,
        cache_size=64,
        # Labeling
        occ_thresh=0.05,
        min_covered=0.95,
        # Sampling / balance
        p_af=0.5,              # probability to sample an AF window (balanced training)
        max_tries=80,
        augment=None,          # for supervised, usually None or mild
        deterministic=False,
        k_per_patient: int = 1,
    ):
        self.patient_dirs = list(patient_dirs)
        self.window_samples = int(window_samples)
        self.fs = fs
        self.occ_thresh = float(occ_thresh)
        self.min_covered = float(min_covered)
        self.p_af = p_af
        self.max_tries = int(max_tries)
        self.augment = augment

        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.rng = np.random.default_rng(self.seed)

        self.k_per_patient = int(k_per_patient)
        assert self.k_per_patient >= 1

        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]

        self._cache_size = int(cache_size)
        self._init_record_cache()

    def __len__(self):
        # Expose K windows per patient
        return len(self.patient_dirs) * self.k_per_patient

    def _rng_for_patient_rep(self, patient_idx: int, rep: int):
        # Stable per patient and per replicate (rep=0..K-1)
        # Using large odd multipliers to reduce collisions.
        return np.random.default_rng(
            self.seed + 1000003 * int(patient_idx) + 9176 * int(rep)
        )
    def _init_record_cache(self):
        # Filter once per record
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


    def _sample_any_labelable_window(self, sig, intervals, rng):
        """Sample a random window; accept the first labelable window (AF or N)."""
        sig_len = len(sig)
        if sig_len <= self.window_samples:
            return None, None

        for _ in range(self.max_tries):
            w_start = int(rng.integers(0, sig_len - self.window_samples))
            w_end = w_start + self.window_samples
            chunk = sig[w_start:w_end]

            if np.isnan(chunk).any() or np.std(chunk) < 1e-4:
                continue

            lab = label_window_strict(
                intervals, w_start, w_end,
                thresh=self.occ_thresh,
                min_covered=self.min_covered
            )
            if lab is None:
                continue

            is_af = (lab == "AFIB") or (lab == "AFL")
            y = LABEL2ID_BIN["AF"] if is_af else LABEL2ID_BIN["N"]
            return chunk, y

        return None, None
    
    def _sample_window_with_target(self, sig, intervals, target_is_af: bool, rng):

        sig_len = len(sig)
        if sig_len <= self.window_samples:
            return None, None

        for _ in range(self.max_tries):
            w_start = int(rng.integers(0, sig_len - self.window_samples))
            w_end = w_start + self.window_samples
            chunk = sig[w_start:w_end]

            if np.isnan(chunk).any() or np.std(chunk) < 1e-4:
                continue

            lab = label_window_strict(
                intervals, w_start, w_end,
                thresh=self.occ_thresh,
                min_covered=self.min_covered
            )
            if lab is None:
                continue  # OTHER/UNLABELED -> skip

            is_af = (lab == "AFIB") or (lab == "AFL")
            if is_af != target_is_af:
                continue

            # Return chunk and binary label
            y = LABEL2ID_BIN["AF"] if is_af else LABEL2ID_BIN["N"]
            return chunk, y

        return None, None

    def __getitem__(self, idx):
        # Map flat idx -> patient index + replicate id
        patient_idx = int(idx) // self.k_per_patient
        rep = int(idx) % self.k_per_patient

        rng = self._rng_for_patient_rep(patient_idx, rep) if self.deterministic else self.rng

        recs = self.patient_records[patient_idx]
        if not recs:
            # fallback random patient
            ridx = int(rng.integers(0, len(self)))
            return self.__getitem__(ridx)

        for _ in range(10):
            rec_path = recs[int(rng.integers(0, len(recs)))]
            sig, intervals = self._load_record_cached(str(rec_path))

            if self.p_af is None:
                x_raw, y = self._sample_any_labelable_window(sig, intervals, rng)
            else:
                target_is_af = (rng.random() < float(self.p_af))
                x_raw, y = self._sample_window_with_target(sig, intervals, target_is_af, rng)

            if x_raw is None:
                continue

            x = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-6)
            x = x.astype(np.float32)

            return {
                "x": torch.from_numpy(x),
                "y": torch.tensor(y, dtype=torch.long),
                "patient_id": self.patient_dirs[patient_idx].name,  
                "rec": rec_path.name,
                "rep": rep,  # optional but useful for debugging
            }

        ridx = int(rng.integers(0, len(self)))
        return self.__getitem__(ridx)