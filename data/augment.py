import numpy as np

class ECGAugment:
    """
    Realistic ECG augmentations for contrastive / masked pretraining.
    Designed to avoid shortcut cues (e.g., hard zeros) and preserve morphology.
    """
    def __init__(
        self,
        fs: int = 250,

        # Gain / noise
        p_scale: float = 1.0,
        scale_min: float = 0.90,
        scale_max: float = 1.10,

        p_white_noise: float = 1.0,
        white_noise_std: float = 0.005,   # relative to signal units; tune after inspecting amplitude

        # Baseline wander (low-frequency drift)
        p_baseline: float = 0.7,
        baseline_amp: float = 0.10,       # as fraction of signal std
        baseline_f_lo: float = 0.05,      # Hz
        baseline_f_hi: float = 0.50,      # Hz

        # EMG-like noise (higher frequency)
        p_emg: float = 0.5,
        emg_std: float = 0.01,            # as fraction of signal std

        # Smooth masking (structured dropout)
        p_mask: float = 0.5,
        n_masks_min: int = 3,
        n_masks_max: int = 12,
        mask_len_min_sec: float = 0.10,
        mask_len_max_sec: float = 0.50,
        mask_depth_min: float = 0.7,      # 0.7 means keep 30% during mask
        mask_depth_max: float = 1.0,      # 1.0 means fully suppressed (but still smooth edges)

        # RNG
        seed: int | None = None,
    ):
        self.fs = fs

        self.p_scale = p_scale
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.p_white_noise = p_white_noise
        self.white_noise_std = white_noise_std

        self.p_baseline = p_baseline
        self.baseline_amp = baseline_amp
        self.baseline_f_lo = baseline_f_lo
        self.baseline_f_hi = baseline_f_hi

        self.p_emg = p_emg
        self.emg_std = emg_std

        self.p_mask = p_mask
        self.n_masks_min = n_masks_min
        self.n_masks_max = n_masks_max
        self.mask_len_min = int(mask_len_min_sec * fs)
        self.mask_len_max = int(mask_len_max_sec * fs)
        self.mask_depth_min = mask_depth_min
        self.mask_depth_max = mask_depth_max

        self.rng = np.random.default_rng(seed)
        

    def _as_2d(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (y[T,C], was_1d)."""
        if x.ndim == 1:
            return x[:, None], True
        return x, False

    def _smooth_mask(self, T: int, start: int, m: int, depth: float) -> np.ndarray:
        """
        Create multiplicative mask of length T with a smooth window on [start, start+m).
        depth in [0,1]: depth=1 -> fully suppressed region, depth=0 -> no suppression.
        We implement a raised-cosine fade.
        """
        mask = np.ones((T,), dtype=np.float32)
        end = min(T, start + m)
        m_eff = end - start
        if m_eff <= 1:
            return mask

        # raised cosine from 1 down to (1-depth) and back
        # window shape in [0,pi]: w = 0.5 - 0.5*cos(t) ranges 0->1
        t = np.linspace(0, np.pi, m_eff, dtype=np.float32)
        w = 0.5 - 0.5 * np.cos(t)  # 0..1
        # center is most suppressed; edges near 0 suppression
        # suppression factor: 1 - depth*w
        mask[start:end] = 1.0 - depth * w
        return mask
    
    def _preprocess(self, x):
        x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-6)
        return x.astype(np.float32)

        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, was_1d = self._as_2d(x.astype(np.float32, copy=True))
        y = self._preprocess(y)
        T, C = y.shape

        # robust scale reference
        sig_std = float(np.std(y)) + 1e-6

        # 1) gain scaling
        if self.rng.random() < self.p_scale:
            s = self.rng.uniform(self.scale_min, self.scale_max)
            y *= np.float32(s)

        # 2) baseline wander: sum of a few low-freq sinusoids
        if self.rng.random() < self.p_baseline:
            t = np.arange(T, dtype=np.float32) / np.float32(self.fs)
            n_comp = int(self.rng.integers(1, 4))
            drift = np.zeros((T,), dtype=np.float32)
            for _ in range(n_comp):
                f = self.rng.uniform(self.baseline_f_lo, self.baseline_f_hi)
                phase = self.rng.uniform(0, 2*np.pi)
                drift += np.sin(2*np.pi*f*t + phase).astype(np.float32)

            drift /= (np.max(np.abs(drift)) + 1e-6)
            amp = np.float32(self.baseline_amp * sig_std)  # in signal units
            y += drift[:, None] * amp

        # 3) EMG-like noise: high-frequency-ish via differenced white noise
        if self.rng.random() < self.p_emg:
            n = self.rng.normal(0.0, 1.0, size=(T, C)).astype(np.float32)
            n = np.concatenate([n[:1], np.diff(n, axis=0)], axis=0)  # emphasize high-freq
            y += n * np.float32(self.emg_std * sig_std)

        # 4) small white noise
        if self.rng.random() < self.p_white_noise:
            y += self.rng.normal(0.0, 1.0, size=y.shape).astype(np.float32) * np.float32(self.white_noise_std * sig_std)

        # 5) structured smooth masking (no hard zeros)
        if self.rng.random() < self.p_mask:
            n_masks = int(self.rng.integers(self.n_masks_min, self.n_masks_max + 1))
            for _ in range(n_masks):
                m = int(self.rng.integers(max(2, self.mask_len_min), max(3, self.mask_len_max + 1)))
                start = int(self.rng.integers(0, max(1, T - m)))
                depth = float(self.rng.uniform(self.mask_depth_min, self.mask_depth_max))
                mult = self._smooth_mask(T, start, m, depth)[:, None]
                y *= mult

        return y[:, 0] if was_1d else y
    
def make_augment(preset: str, fs: int = 250):
    if preset == "16s":
        return ECGAugment(fs=fs, n_masks_min=0, n_masks_max=1)
    elif preset == "10min":
        return ECGAugment(fs=fs, n_masks_min=3, n_masks_max=12)
    else:
        raise ValueError(f"Unknown aug preset: {preset}")
