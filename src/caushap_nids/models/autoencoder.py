from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .interface import AnomalyDetector

_DEVICE_MAP = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}
# Preload training data to device when it fits — avoids per-batch CPU→GPU transfers,
# which dominate for small models on both MPS (Apple Silicon) and CUDA.
# Bumped to 6 GB so the 12 M-row NF-CIC2018 benign split (≈ 2 GB in float32, plus
# val + permutation buffers) fits resident on a 16 GB RTX 5080 / 4090.
_PRELOAD_LIMIT_BYTES = 6 * 1024 ** 3  # 6 GB

_CUDA_TUNED = False


def _tune_cuda_once() -> None:
    """Enable TF32 + cuDNN autotune on first CUDA use.

    These knobs are global on the process — flipping them once is enough.
    They cut AE training/inference time by ~30 % on Ampere/Ada/Blackwell GPUs
    (RTX 30/40/50 series) at no measurable accuracy cost for this network.
    """
    global _CUDA_TUNED
    if _CUDA_TUNED or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _CUDA_TUNED = True


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        # Explicit override (e.g. CAUSHAP_DEVICE=cpu on Apple Silicon, where the
        # 39-dim AE is so small that MPS kernel-launch overhead makes the many
        # tiny XAI inferences 5-12x slower than CPU — see RUN_GUIDE_MAC_M4.md).
        import os
        forced = os.environ.get("CAUSHAP_DEVICE", "").strip().lower()
        if forced in {"cpu", "cuda", "mps"}:
            return torch.device(forced)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class _AENet(nn.Module):
    """Internal PyTorch module — encoder/decoder with LayerNorm + Dropout."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()

        enc, prev = [], in_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc)

        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def per_feature_error(self, x: torch.Tensor) -> torch.Tensor:
        """Squared error per feature, shape (B, D). Used by Module 5a hooks."""
        return (x - self(x)) ** 2

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        method: str = "mean",
        topk: int | None = None,
    ) -> torch.Tensor:
        err = self.per_feature_error(x)
        if method == "mean" or topk is None or topk >= err.size(1):
            return err.mean(dim=1)
        top, _ = err.topk(topk, dim=1)
        return top.mean(dim=1)


class DeepAutoEncoder(AnomalyDetector):
    """Benign-only autoencoder anomaly detector.

    Default architecture: in_dim → 64 → 32 → 16 → 32 → 64 → in_dim
    Trained with MSE loss + L1 bottleneck regularisation.

    `per_feature_error` is exposed for Module 5a Causal Shapley hooks.
    """

    def __init__(
        self,
        in_dim: int = 41,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        lambda_l1: float = 3e-4,
        epochs: int = 100,
        batch_size: int = 2048,
        patience: int = 8,
        min_delta: float = 1e-6,
        grad_clip: float = 1.0,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.in_dim       = in_dim
        self.hidden_dims  = list(hidden_dims)
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.lambda_l1    = lambda_l1
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.min_delta    = min_delta
        self.grad_clip    = grad_clip
        self.device       = _resolve_device(device)
        self.seed         = seed

        if str(self.device).startswith("cuda"):
            _tune_cuda_once()

        torch.manual_seed(seed)
        self._net = _AENet(in_dim, hidden_dims, dropout).to(self.device)
        self._best_state = {k: v.clone() for k, v in self._net.state_dict().items()}

    # ── AnomalyDetector interface ──────────────────────────────────────────

    def fit(self, X_benign: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        n = len(X_benign)
        n_val   = max(1, int(0.15 * n))
        n_train = n - n_val

        is_cuda = str(self.device).startswith("cuda")
        is_mps  = str(self.device).startswith("mps")

        # Preload path: move full training tensor to device once.
        # Eliminates per-batch CPU→GPU transfers that dominate for small models on
        # both MPS (Apple Silicon unified memory still has domain-switch overhead)
        # and CUDA. Safe when the training split fits within _PRELOAD_LIMIT_BYTES.
        preload = (is_cuda or is_mps) and (X_benign[:n_train].nbytes < _PRELOAD_LIMIT_BYTES)

        perm_gen = torch.Generator()
        perm_gen.manual_seed(self.seed)

        if preload:
            X_train_dev = torch.tensor(X_benign[:n_train], dtype=torch.float32, device=self.device)
            X_val_t     = torch.tensor(X_benign[n_train:], dtype=torch.float32, device=self.device)
        else:
            X_val_t = torch.tensor(X_benign[n_train:], dtype=torch.float32, device=self.device)
            loader  = DataLoader(
                TensorDataset(torch.tensor(X_benign[:n_train], dtype=torch.float32)),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                pin_memory=is_cuda,
                generator=torch.Generator().manual_seed(self.seed),
            )

        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        patience_cnt = 0
        self.train_losses_: list[float] = []
        self.val_losses_:   list[float] = []
        infer_batch = max(self.batch_size, 131072 if str(self.device).startswith("cuda") else 32768)

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            ep_loss, n_seen = 0.0, 0

            if preload:
                # Shuffle indices on CPU (cheap), move to device, gather rows on device.
                # Per-batch gather of 2048 rows from a resident tensor avoids any
                # cross-domain copy entirely.
                perm     = torch.randperm(n_train, generator=perm_gen).to(self.device)
                for start in range(0, n_train, self.batch_size):
                    xb = X_train_dev[perm[start : start + self.batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    z    = self._net.encode(xb)
                    loss = criterion(self._net.decoder(z), xb) + self.lambda_l1 * z.abs().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._net.parameters(), self.grad_clip)
                    optimizer.step()
                    ep_loss += loss.detach().item() * xb.size(0)
                    n_seen  += xb.size(0)
            else:
                for (xb,) in loader:
                    xb = xb.to(self.device, non_blocking=is_cuda)
                    optimizer.zero_grad(set_to_none=True)
                    z    = self._net.encode(xb)
                    loss = criterion(self._net.decoder(z), xb) + self.lambda_l1 * z.abs().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._net.parameters(), self.grad_clip)
                    optimizer.step()
                    ep_loss += loss.detach().item() * xb.size(0)
                    n_seen  += xb.size(0)

            ep_loss /= max(n_seen, 1)
            self._net.eval()
            with torch.no_grad():
                _vl, _vn = 0.0, 0
                for i in range(0, len(X_val_t), infer_batch):
                    xv = X_val_t[i : i + infer_batch]
                    _vl += criterion(self._net(xv), xv).item() * xv.size(0)
                    _vn += xv.size(0)
                val_loss = _vl / max(_vn, 1)

            scheduler.step(val_loss)
            self.train_losses_.append(ep_loss)
            self.val_losses_.append(val_loss)

            if val_loss < best_val - self.min_delta:
                best_val = val_loss
                self._best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        self._net.load_state_dict(self._best_state)
        self._net.eval()

    def score(
        self,
        X: np.ndarray,
        *,
        feature_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
        topk: int | None = None,
    ) -> np.ndarray:
        """Per-flow reconstruction error, shape (N,).

        By default this is the mean squared reconstruction error over all
        features.  `feature_indices` supports validation-selected residual
        subsets for low-FPR operating points; `topk` keeps the largest k
        residuals inside the chosen feature set.
        """
        self._net.eval()
        score_batch = max(self.batch_size, 131072 if str(self.device).startswith("cuda") else 32768)
        feat_idx = None
        if feature_indices is not None:
            feat_arr = np.asarray(feature_indices, dtype=np.int64).ravel()
            if len(feat_arr) == 0:
                raise ValueError("feature_indices must contain at least one feature")
            if feat_arr.min() < 0 or feat_arr.max() >= self.in_dim:
                raise ValueError(f"feature_indices must be in [0, {self.in_dim})")
            feat_idx = torch.as_tensor(feat_arr, dtype=torch.long, device=self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), score_batch):
                xb = torch.tensor(X[i : i + score_batch], dtype=torch.float32, device=self.device)
                err = self._net.per_feature_error(xb)
                if feat_idx is not None:
                    err = err.index_select(1, feat_idx)
                if topk is not None and 0 < topk < err.size(1):
                    err = err.topk(topk, dim=1).values
                out.append(err.mean(dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return latent representation for debugging and Module 5a hooks."""
        self._net.eval()
        infer_batch = max(self.batch_size, 131072 if str(self.device).startswith("cuda") else 32768)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), infer_batch):
                xb = torch.tensor(X[i : i + infer_batch], dtype=torch.float32, device=self.device)
                out.append(self._net.encode(xb).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def per_feature_error(self, X: np.ndarray) -> np.ndarray:
        """Squared error per feature, shape (N, D). Called by Module 5a."""
        self._net.eval()
        infer_batch = max(self.batch_size, 131072 if str(self.device).startswith("cuda") else 32768)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), infer_batch):
                xb = torch.tensor(X[i : i + infer_batch], dtype=torch.float32, device=self.device)
                out.append(self._net.per_feature_error(xb).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict":  self._net.state_dict(),
                "in_dim":      self.in_dim,
                "hidden_dims": self.hidden_dims,
                "dropout":     self.dropout,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if "state_dict" in ckpt:
            self._net.load_state_dict(ckpt["state_dict"])
        else:
            self._net.load_state_dict(ckpt)
        self._net.eval()
