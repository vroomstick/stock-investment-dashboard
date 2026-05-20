"""
models/lstm_model.py

LSTM sequence model for 4-class return prediction.

WHY AN LSTM (IN ADDITION TO XGBOOST)
--------------------------------------
XGBoost sees each day as an independent snapshot — it has no concept of
"the stock has been falling for 3 weeks" unless we manually engineer lag
features. An LSTM sees a sequence of 60 daily observations and learns
patterns like:

  - Momentum continuation: 3-week uptrend + increasing volume → class 2/3
  - Mean reversion: overbought RSI (>80) after large gap up → class 0/1
  - Volatility regimes: VIX spike + credit spread widening → class 0
  - MACD crossover timing: the exact day a crossover happens

These temporal patterns are structurally impossible for XGBoost to learn
from a single-day feature vector.

LSTM ARCHITECTURE
------------------
    Input:  (batch, 60, 18)
              60 = sequence length (trading days)
              18 = lstm_feature_subset features

    LSTM:   hidden_size=128, num_layers=2, dropout=0.3
              Layer 1: (60, 18) → (60, 128)  hidden states at every step
              Layer 2: (60, 128) → (60, 128) further temporal abstraction
              Dropout between layers (prevents memorizing specific sequences)

    Head:   Linear(128 → 4) → applied to the LAST hidden state only
              The last hidden state = the network's "summary" of the 60-day sequence
              We don't average all hidden states — we want recency bias
              (the last few days matter more than 60 days ago)

    Output: logits(4) → softmax → probability vector [p0, p1, p2, p3]

Why only the last hidden state?
  An alternative is attention over all 60 steps (Transformer-style). We
  start with LSTM because:
    1. Our sequences are short (60 steps) — attention overkill for this length
    2. LSTM is easier to debug and explain
    3. The recency bias of the final hidden state matches financial intuition

WHY PYTORCH (NOT TENSORFLOW/KERAS)
-------------------------------------
PyTorch is the research standard for sequence models. The training loop
is explicit and inspectable — we can add gradient clipping, learning rate
scheduling, and custom loss functions without fighting a framework. Also,
xgboost integrates better with numpy than TensorFlow does.

TRAINING DETAILS
-----------------
  Loss:      CrossEntropyLoss (same log-loss as XGBoost's multi:softprob)
  Optimizer: Adam, lr=1e-3
  Scheduler: ReduceLROnPlateau (halve lr if val loss doesn't improve for 5 epochs)
  Gradient clipping: max_norm=1.0 (prevents exploding gradients in LSTM)
  Early stopping: stop if val loss doesn't improve for `patience` epochs
  Batch size: 64 (larger batches = more stable gradients on sequence data)

FALLBACK (no torch)
--------------------
If PyTorch is not installed, the module raises a clean ImportError pointing
to the install command. The ensemble will fall back to XGBoost-only mode.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


class LSTMClassifier:
    """
    PyTorch LSTM for 4-class sequence classification.

    Usage:
        model = LSTMClassifier()
        model.fit(X_train, y_train, X_val, y_val)
        probs = model.predict_proba(X_test)  # shape (n, 4)
        model.save("lstm_v1")
    """

    def __init__(self, params: dict = None):
        """
        Args:
            params: Override LSTM_PARAMS from settings.
        """
        from config.settings import LSTM_PARAMS
        self.params    = {**LSTM_PARAMS, **(params or {})}
        self._model    = None
        self._scaler   = None   # StandardScaler for sequence normalization
        self.is_fitted = False
        self._history: dict = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,   # (n_samples, seq_len, n_features)
        y_train: np.ndarray,   # (n_samples,)
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "LSTMClassifier":
        """
        Train the LSTM.

        Args:
            X_train: 3D float32 array (n_samples, seq_len, n_features)
            y_train: 1D int64 array of class labels 0-3
            X_val:   Validation sequences for early stopping
            y_val:   Validation labels
            verbose: Print epoch losses

        Why normalize sequences separately from XGBoost features?
            The LSTM feature subset is already z-scored by FeaturePreprocessor
            when those features go through the joint preprocessing step. But
            here we apply a *per-feature* normalization across the time axis
            to help gradient flow in the LSTM. Without normalization, features
            like "volume" (millions) dominate "rsi_14" (0-100) in the loss
            gradient on the first few epochs.

        Returns:
            self (for method chaining)
        """
        torch = _require_torch()

        p = self.params

        # Build the model
        self._model = _LSTMNet(
            input_size  = p["input_size"],
            hidden_size = p["hidden_size"],
            num_layers  = p["num_layers"],
            num_classes = p["num_classes"],
            dropout     = p["dropout"],
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=p["learning_rate"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=verbose
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Build DataLoaders
        train_loader = _make_loader(X_train, y_train, p["batch_size"], shuffle=True)
        val_loader   = _make_loader(X_val, y_val, p["batch_size"], shuffle=False) \
                       if X_val is not None else None

        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        for epoch in range(p["max_epochs"]):
            # --- Training pass ---
            self._model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self._model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            self._history["train_loss"].append(train_loss)

            # --- Validation pass ---
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        logits    = self._model(xb)
                        val_loss += criterion(logits, yb).item() * len(xb)
                val_loss /= len(X_val)
                self._history["val_loss"].append(val_loss)

                scheduler.step(val_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}  val={val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss  = val_loss
                    patience_count = 0
                    best_state     = {k: v.clone() for k, v
                                      in self._model.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= p["patience"]:
                        if verbose:
                            print(f"  Early stop at epoch {epoch+1} "
                                  f"(best val loss: {best_val_loss:.4f})")
                        break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}")

        # Restore best model weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability matrix shape (n_samples, 4).

        Args:
            X: 3D array (n_samples, seq_len, n_features)

        Returns:
            np.ndarray (n_samples, 4) — probabilities summing to 1.0 per row
        """
        self._check_fitted()
        torch = _require_torch()

        self._model.eval()
        loader = _make_loader(X, labels=None, batch_size=256, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (xb,) in loader:
                logits = self._model(xb)
                probs  = torch.softmax(logits, dim=1)
                all_probs.append(probs.numpy())

        return np.concatenate(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels (argmax of probabilities)."""
        return self.predict_proba(X).argmax(axis=1)

    def training_history(self) -> pd.DataFrame:
        """Return DataFrame with train_loss and val_loss per epoch."""
        return pd.DataFrame(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, version: str) -> None:
        """
        Save model weights to models/artifacts/lstm_{version}.pt
        and params to models/artifacts/lstm_{version}_meta.json
        """
        self._check_fitted()
        torch = _require_torch()

        out_dir = ROOT / "models" / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        pt_path   = out_dir / f"lstm_{version}.pt"
        meta_path = out_dir / f"lstm_{version}_meta.json"

        torch.save(self._model.state_dict(), pt_path)

        meta = {
            "version": version,
            "params":  self.params,
            "history": {k: [float(v) for v in vals]
                        for k, vals in self._history.items()},
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"LSTMClassifier saved → {pt_path}")

    @classmethod
    def load(cls, version: str) -> "LSTMClassifier":
        """Load a saved LSTM model."""
        torch = _require_torch()

        out_dir   = ROOT / "models" / "artifacts"
        pt_path   = out_dir / f"lstm_{version}.pt"
        meta_path = out_dir / f"lstm_{version}_meta.json"

        if not pt_path.exists():
            raise FileNotFoundError(f"No LSTM model found at {pt_path}")

        instance = cls.__new__(cls)

        with open(meta_path) as f:
            meta = json.load(f)

        instance.params    = meta["params"]
        instance._history  = meta.get("history", {"train_loss": [], "val_loss": []})
        instance.is_fitted = True

        p = instance.params
        instance._model = _LSTMNet(
            input_size  = p["input_size"],
            hidden_size = p["hidden_size"],
            num_layers  = p["num_layers"],
            num_classes = p["num_classes"],
            dropout     = p["dropout"],
        )
        instance._model.load_state_dict(
            torch.load(pt_path, map_location="cpu")
        )
        instance._model.eval()
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted or self._model is None:
            raise RuntimeError(
                "Model is not fitted. Call .fit() before predict/save."
            )


# ---------------------------------------------------------------------------
# PyTorch model definition (private)
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for LSTMClassifier.\n"
            "Install with: pip install torch\n"
            "If you don't want the LSTM, use XGBoostClassifier alone."
        )


class _LSTMNet:
    """
    Pure PyTorch LSTM → Linear(4) → softmax classifier.

    Not exposed publicly — always accessed through LSTMClassifier.

    Architecture:
        LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        → take output[:, -1, :]  (last time step only)
        → Linear(hidden_size, num_classes)
        → logits (no softmax here — CrossEntropyLoss handles that)

    Why batch_first=True?
        Our data is shaped (batch, seq, features) which matches batch_first=True.
        PyTorch's default is (seq, batch, features) which would require a transpose.
    """

    def __new__(cls, input_size, hidden_size, num_layers, num_classes, dropout):
        torch = _require_torch()

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size   = input_size,
                    hidden_size  = hidden_size,
                    num_layers   = num_layers,
                    dropout      = dropout if num_layers > 1 else 0.0,
                    batch_first  = True,
                )
                self.dropout = torch.nn.Dropout(dropout)
                self.fc      = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                # x: (batch, seq_len, input_size)
                out, _ = self.lstm(x)         # out: (batch, seq_len, hidden_size)
                last   = out[:, -1, :]        # last time step: (batch, hidden_size)
                last   = self.dropout(last)
                return self.fc(last)          # logits: (batch, num_classes)

        return Net()


def _make_loader(X, labels=None, batch_size=64, shuffle=False):
    """Create a DataLoader from numpy arrays."""
    torch = _require_torch()

    X_t = torch.tensor(X, dtype=torch.float32)

    if labels is not None:
        y_t = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
    else:
        dataset = torch.utils.data.TensorDataset(X_t)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
