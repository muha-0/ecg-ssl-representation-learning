from dataclasses import dataclass
from pathlib import Path

@dataclass
class SSLConfig:
    root: Path = Path("/data/ahmed/icentia11k")
    fs: int = 250
    window_sec: int = 600          # 10 min default
    patch_len: int = 160
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42

    # data backend
    backend: str = "wfdb"          # "wfdb" or "npy"
    processed_root: Path | None = None

    # augmentation preset
    aug_preset: str = "10min"      # "10min" or "16s"

    # codebook path
    codebook_path: str = "codebooks/icentia_codebook_256x160.pt"

    @property
    def window_samples(self) -> int:
        return self.fs * self.window_sec

    @property
    def max_len(self) -> int:
        return (self.window_samples // self.patch_len) + 2