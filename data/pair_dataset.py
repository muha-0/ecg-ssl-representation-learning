import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
import scipy.signal
from pathlib import Path
from functools import lru_cache
from .utils import list_record_bases
from .rhythm import build_rhythm_intervals_from_ann, label_window_by_occupancy

class IcentiaPatientPairDataset(Dataset):
    def __init__(self, patient_dirs, window_samples, root: Path, fs=250, augment=None, seed=0, cache_size=64, processed_root: str | None = None):
        self.patient_dirs = list(patient_dirs)
        self.window_samples = window_samples
        self.fs = fs
        self.augment = augment
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        # Pre-cache list of record paths per patient
        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]

        self._cache_size = int(cache_size)
        self.root = Path(root)
        self.processed_root = Path(processed_root) if processed_root else None
        self._init_record_cache()
        
    def __len__(self):
        return len(self.patient_dirs)

    def _init_record_cache(self):
        if self.processed_root is None:
            # Define filter parameters once to avoid re-calculation
            lowcut = 0.5
            highcut = 40.0
            nyq = 0.5 * self.fs
            low = lowcut / nyq
            high = highcut / nyq
            # We define b, a here so they are captured by the closure below
            b, a = scipy.signal.butter(4, [low, high], btype='bandpass')

            @lru_cache(maxsize=self._cache_size)
            def _cached_load(rec_base_str: str):
                # 1. Load the raw signal
                sig, fields = wfdb.rdsamp(rec_base_str)
                sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
                
                # 2. FILTER ONCE: Apply the bandpass filter to the entire 70-minute record
                # Doing this once per record saves massive CPU cycles compared to doing it per-window
                sig = scipy.signal.filtfilt(b, a, sig).astype(np.float32)
                
                sig_len = len(sig)
                # 3. Load annotations
                ann = wfdb.rdann(rec_base_str, extension="atr")
                intervals = build_rhythm_intervals_from_ann(ann, sig_len)
                
                return sig, intervals

            self._load_record_cached = _cached_load
        else:
            @lru_cache(maxsize=self._cache_size)
            def _cached_load(patient_folder_name, rec_name):
                # 1. Path to the processed .npy file
                parent_folder = patient_folder_name[:3] 
                npy_path = self.processed_root / parent_folder / patient_folder_name / f"{rec_name}_filtered.npy"
                rec_base_path = self.root / parent_folder / patient_folder_name / rec_name

                # 2. Load with MEMMAP (this is nearly instantaneous)
                sig = np.load(str(npy_path), mmap_mode='r')
                # 3. Load annotations
                ann = wfdb.rdann(str(rec_base_path), extension="atr")
                intervals = build_rhythm_intervals_from_ann(ann, len(sig))
                
                return sig, intervals

            self._load_record_cached = _cached_load

    def _sample_valid_window(self, sig, intervals, max_tries=50):
        sig_len = len(sig)
        if sig_len <= self.window_samples:
            return None, None, None

        for _ in range(max_tries):
            # Sample a random start time
            w_start = int(self.rng.integers(0, sig_len - self.window_samples))
            w_end = w_start + self.window_samples
            
            chunk = sig[w_start:w_end]

            # QC: reject windows with NaNs or flat-lines
            if np.isnan(chunk).any() or np.std(chunk) < 1e-4:
                continue

            lab = label_window_by_occupancy(intervals, w_start, w_end, thresh=0.10)
            if lab is not None:
                return w_start, w_end, lab

        return None, None, None

    def __getitem__(self, idx):
        if self.processed_root is None:
            recs = self.patient_records[idx]
            if not recs:
                raise RuntimeError(f"No records found for patient index {idx}")

            # If patient has >=2 records, force two different records.
            # Otherwise fall back to same record (can't do cross-record).
            if len(recs) >= 2:
                r1, r2 = self.rng.choice(len(recs), size=2, replace=False)
                rec_path1 = recs[int(r1)]
                rec_path2 = recs[int(r2)]
            else:
                rec_path1 = recs[0]
                rec_path2 = recs[0]

            # Load record 1
            sig1, intervals1 = self._load_record_cached(str(rec_path1))
        
            res1 = self._sample_valid_window(sig1, intervals1)
            if res1[0] is None:
                return self.__getitem__(self.rng.integers(0, len(self)))
            s1, e1, y1 = res1
            x_raw1 = sig1[s1:e1]

            # Load record 2 (different record)
            sig2, intervals2 = self._load_record_cached(str(rec_path2))
            res2 = self._sample_valid_window(sig2, intervals2)
            if res2[0] is None:
                # fallback: try a few alternate records to still enforce "different record"
                ok = False
                if len(recs) >= 2:
                    for _ in range(6):
                        alt = recs[int(self.rng.integers(0, len(recs)))]
                        if str(alt) == str(rec_path1):
                            continue
                        sig2, intervals2 = self._load_record_cached(str(alt))
                        res2 = self._sample_valid_window(sig2, intervals2)
                        if res2[0] is not None:
                            rec_path2 = alt
                            ok = True
                            break
                if not ok:
                    # last-resort: copy view1 (rare)
                    x_raw2 = x_raw1.copy()
                    y2 = y1
                else:
                    s2, e2, y2 = res2
                    x_raw2 = sig2[s2:e2]
            else:
                s2, e2, y2 = res2
                x_raw2 = sig2[s2:e2]

            # Apply augmentations
            if self.augment is not None:
                x1 = self.augment(x_raw1)
                x2 = self.augment(x_raw2)
            else:
                x1, x2 = x_raw1.copy(), x_raw2.copy()

            return {
                "patient_id": self.patient_dirs[idx].name,
                "rec1": rec_path1.name,
                "rec2": rec_path2.name,
                "x1": torch.from_numpy(x1),
                "x2": torch.from_numpy(x2),
                "y": y1,
            }
        else:
            patient_path = self.patient_dirs[idx]
            p_folder = patient_path.name # e.g. "p00001"
            recs = self.patient_records[idx]

            if len(recs) >= 2:
                r1, r2 = self.rng.choice(len(recs), size=2, replace=False)
                rec_path1 = recs[int(r1)]
                rec_path2 = recs[int(r2)]
            else:
                rec_path1 = recs[0]
                rec_path2 = recs[0]

            
            sig1, intervals1 = self._load_record_cached(p_folder, rec_path1.name)
            res1 = self._sample_valid_window(sig1, intervals1)
            if res1[0] is None:
                return self.__getitem__(self.rng.integers(0, len(self)))
            
            s1, e1, y1 = res1
            x_raw1 = np.array(sig1[s1:e1]) # Force conversion to standard numpy array

            sig2, intervals2 = self._load_record_cached(p_folder, rec_path2.name)
            res2 = self._sample_valid_window(sig2, intervals2)
            
            if res2[0] is None:
                x_raw2 = x_raw1.copy()
                y2 = y1
            else:
                s2, e2, y2 = res2
                x_raw2 = np.array(sig2[s2:e2]) # Force conversion to standard numpy array

            if self.augment is not None:
                x1 = self.augment(x_raw1)
                x2 = self.augment(x_raw2)
            else:
                x1, x2 = x_raw1.copy(), x_raw2.copy()

            return {
                "patient_id": p_folder,
                "rec1": rec_path1.name,
                "rec2": rec_path2.name,
                "x1": torch.from_numpy(x1),
                "x2": torch.from_numpy(x2),
                "y": y1,
            }
    