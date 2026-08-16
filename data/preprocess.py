import os
from pathlib import Path
import numpy as np
import wfdb
import scipy.signal
from tqdm import tqdm

from .utils import list_record_bases

def preprocess_to_memmap(patient_dirs, root, save_root, fs=250):
    root = Path(root)
    save_root = Path(save_root)
    os.makedirs(save_root, exist_ok=True)
    
    # Define filter
    lowcut, highcut = 0.5, 40.0
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(4, [lowcut/nyq, highcut/nyq], btype='bandpass')

    print(f"Preprocessing {len(patient_dirs)} patients...")
    for pd in tqdm(patient_dirs):
        # Create subfolders like p00/p00001
        rel_path = pd.relative_to(root)
        (save_root / rel_path).mkdir(parents=True, exist_ok=True)
        
        recs = list_record_bases(pd)
        for rec in recs:
            save_path = save_root / rel_path / f"{rec.name}_filtered.npy"
            
            # Skip if already processed
            if save_path.exists():
                continue
                
            try:
                # Load and Filter
                sig, _ = wfdb.rdsamp(str(rec))
                sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
                sig = scipy.signal.filtfilt(b, a, sig).astype(np.float32)
                
                # Save as raw numpy array
                np.save(str(save_path), sig)
            except Exception as e:
                print(f"Error processing {rec}: {e}")

# Run this once!
PROCESSED_ROOT = "/data/ahmed/icentia_processed"
# preprocess_to_memmap(patients, PROCESSED_ROOT)