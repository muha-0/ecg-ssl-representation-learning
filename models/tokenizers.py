import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Tokenizer A: Quantization Based
# -------------------------

class KMeansVQPatchTokenizer(nn.Module):
    """
    Vector-quantize each patch using a fixed KMeans codebook.

    Inputs:
      x: [B, T] window (typically already z-scored by pipeline, but we can enforce it)

    Output:
      tok: [B, L, d_model] token embeddings
      (optionally ids: [B, L])
    """
    def __init__(
        self,
        d_model: int,
        centroids: torch.Tensor,     # [K, patch_len]
        patch_len: int = 160,
        enforce_window_zscore: bool = True,
        use_embedding_table: bool = True,
        return_ids: bool = False,
    ):
        super().__init__()
        assert centroids.ndim == 2, "centroids must be [K, patch_len]"
        K, P = centroids.shape
        assert P == patch_len, f"centroids patch_len ({P}) != patch_len ({patch_len})"

        self.patch_len = int(patch_len)
        self.K = int(K)
        self.enforce_window_zscore = bool(enforce_window_zscore)
        self.use_embedding_table = bool(use_embedding_table)
        self.return_ids = bool(return_ids)

        # Register codebook as a buffer (no gradients)
        self.register_buffer("centroids", centroids.float().contiguous())  # [K, P]

        if self.use_embedding_table:
            # Token embedding lookup (learnable, like NLP)
            self.embed = nn.Embedding(self.K, d_model)
        else:
            # Use centroid waveform itself then project to d_model (non-learnable codebook, learnable projection)
            self.proj = nn.Linear(self.patch_len, d_model, bias=True)

    @staticmethod
    def _window_zscore(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, x: torch.Tensor):
        # x: [B, T]
        B, T = x.shape
        L = T // self.patch_len
        if L <= 0:
            raise ValueError(f"Input length T={T} smaller than patch_len={self.patch_len}")

        x = x[:, : L * self.patch_len]                      # [B, L*P]
        if self.enforce_window_zscore:
            x = self._window_zscore(x)

        patches = x.view(B, L, self.patch_len)              # [B, L, P]

        # Nearest-centroid assignment (Euclidean)
        # torch.cdist is convenient and reasonably fast for your sizes (B*L ~ 32*2*937)
        # distances: [B, L, K]
        d = torch.cdist(patches, self.centroids.unsqueeze(0), p=2)  # [B, L, K]
        ids = d.argmin(dim=-1)                                      # [B, L]

        if self.use_embedding_table:
            tok = self.embed(ids)                                   # [B, L, d_model]
        else:
            # Use centroid waveforms and project
            q = self.centroids[ids]                                 # [B, L, P]
            tok = self.proj(q)                                      # [B, L, d_model]

        if self.return_ids:
            return tok, ids
        return tok



# -------------------------
# Tokenizer B: Conv1d patch embedding
# -------------------------
class ConvPatchTokenizer(nn.Module):
    def __init__(self, d_model: int, patch_len: int = 160):
        super().__init__()
        self.patch_len = patch_len
        self.conv = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=patch_len)

    def forward(self, x):  # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        tok = self.conv(x)  # [B, d_model, L]
        tok = tok.transpose(1, 2)  # [B, L, d_model]
        return tok