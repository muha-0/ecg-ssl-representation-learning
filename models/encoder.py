import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Transformer encoder
# -------------------------

class ECGEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: nn.Module,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        proj_dim=128,
        max_len=1024,          # set based on (WINDOW_SAMPLES // patch_len)
        use_cls=False,         # False => mean pooling
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_cls = use_cls

        # +1 for CLS if used
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + (1 if use_cls else 0), d_model))

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

        # init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_h = False):  # x: [B, T]
        tok = self.tokenizer(x)  # [B, L, d_model]
        B, L, D = tok.shape

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)      # [B, 1, D]
            tok = torch.cat([cls, tok], dim=1)          # [B, 1+L, D]
            tok = tok + self.pos_embed[:, : (L + 1), :]
        else:
            tok = tok + self.pos_embed[:, :L, :]

        z = self.tr(tok)          # [B, L(+1), D]
        z = self.norm(z)

        if self.use_cls:
            h = z[:, 0]           # CLS
        else:
            h = z.mean(dim=1)     # mean pooling over tokens

        y = self.proj(h)
        y = F.normalize(y, dim=-1)

        if return_h:
            return h, y
        return y