"""Multi-task MLP strategy for CSI500 stock selection.

Design choices and the *why* behind them
----------------------------------------
1. Inputs are the v2 35-D feature panel.  Already daily-winsorized + z-scored,
   so the network sees a ~standard-normal cross-section and we don't need
   batch-norm.

2. Architecture: a 3-layer shared trunk (128 -> 128 -> 64, GELU, dropout 0.2)
   and 4 single-linear heads, one per horizon target.  This is the *one*
   thing GBM ensembles cannot do: actually share representation weights
   across multiple horizons.  Round-2's `xgb_v2_multi_target` ensemble
   trained four independent boosters and averaged their post-hoc ranks --
   no representation sharing -> nothing learned that XGBoost can't.

3. Loss: per-day rank-correlation loss (Pearson on within-day ranks of
   prediction & target).  Direct surrogate for rank IC.  Pure regression
   MSE optimises magnitudes that we don't care about and lets one big
   outlier dominate gradients.

4. Top-K aggregation: predict scores from each head on the as_of day,
   z-score per-head, average, then feed into `build_portfolio` like every
   other strategy.  No leakage: each head is just a linear combo of trunk
   features, all of which are computed strictly before as_of.

5. Determinism: one fresh model per (as_of) call.  Seed pinned.  No
   warm-start between rolling windows -- that would leak future data
   indirectly via initialisation.

Compatible with the EnsembleStrategy interface: provides `fit_predict_scores`
returning (Series, dict).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

import features_v2
from strategies import (
    DEFAULT_TOP_K,
    DEFAULT_EMBARGO,
    StrategyResult,
    _train_val_split,
    build_portfolio,
    rank_ic,
)

# Pin determinism as much as possible without losing CPU-throughput.
os.environ.setdefault("PYTHONHASHSEED", "42")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _per_day_rank_corr_loss(
    pred: torch.Tensor, target: torch.Tensor, day_id: torch.Tensor
) -> torch.Tensor:
    """Average within-day Pearson correlation between predicted ranks and
    target ranks, returned as a *negative* number (since we minimise loss).

    For each unique day we:
        rp = soft_rank(pred)
        rt = ranked target  (computed once outside, here just z-score it)
    and compute Pearson on (rp, rt).

    Note: `target` is expected to already be the per-day rank of the raw
    forward return -> we just standardise it to mean 0 std 1 and do the
    same to a soft-ranked-by-magnitude `pred`.  Differentiable enough.
    """
    losses = []
    unique_days = torch.unique(day_id)
    for d in unique_days:
        mask = day_id == d
        if mask.sum() < 5:
            continue
        p = pred[mask]
        t = target[mask]
        # standardise both to (0, 1) -> Pearson reduces to mean(p*t)
        p = (p - p.mean()) / (p.std(unbiased=False) + 1e-8)
        t = (t - t.mean()) / (t.std(unbiased=False) + 1e-8)
        losses.append(-(p * t).mean())
    if not losses:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return torch.stack(losses).mean()


class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int, n_heads: int,
                 hidden: Tuple[int, ...] = (128, 128, 64),
                 dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (N, n_heads) -- one score per horizon."""
        z = self.trunk(x)
        return torch.cat([h(z) for h in self.heads], dim=1)


@dataclass
class MLPStrategy:
    """Multi-task MLP over v2 features and 4 forward-return horizons."""
    name: str = "mlp_v2"
    feature_columns: tuple = tuple(features_v2.ALL_FEATURES)
    target_columns: tuple = (
        "target_3d", "target_5d", "target_10d", "target_5d_sharpe",
    )
    hidden: tuple = (128, 128, 64)
    dropout: float = 0.2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_days: int = 8           # mini-batch = N consecutive trading days
    val_days: int = 30
    embargo: int = DEFAULT_EMBARGO
    seed: int = 42
    early_stop_patience: int = 5

    def build_panel(self, prices, index_df):
        return features_v2.build_features(prices, index_df)

    def _training_frame_fn(self):
        # We need *all* targets non-null, so dropna against every target column.
        from functools import partial
        def _fn(panel, min_date=None, max_date=None, target=None):
            df = panel.dropna(
                subset=list(self.feature_columns) + list(self.target_columns)
            ).copy()
            if min_date is not None:
                df = df[df["date"] >= pd.Timestamp(min_date)]
            if max_date is not None:
                df = df[df["date"] <= pd.Timestamp(max_date)]
            return df
        return _fn

    def _prediction_frame_fn(self):
        return features_v2.prediction_frame

    def _per_day_rank(self, df: pd.DataFrame, target: str) -> pd.Series:
        """Per-day rank of the raw forward return, scaled to [-1, 1]."""
        r = df.groupby("date")[target].rank(method="average", pct=True)
        return (r * 2 - 1).astype(np.float32)

    def fit_predict_scores(self, panel, as_of):
        _set_seed(self.seed)

        feats = list(self.feature_columns)
        targets = list(self.target_columns)
        train_df, val_df, train_end, val_start = _train_val_split(
            panel, as_of, self.val_days, self.embargo,
            training_frame_fn=self._training_frame_fn(),
        )

        # Replace each forward-return target with its per-day rank in [-1, 1]
        # (matches the loss's expectation and removes return-magnitude noise).
        for c in targets:
            train_df = train_df.copy()
            val_df = val_df.copy()
            train_df[c + "_r"] = self._per_day_rank(train_df, c)
            val_df[c + "_r"] = self._per_day_rank(val_df, c)
        rank_targets = [c + "_r" for c in targets]

        # Day-id maps each row to a contiguous integer per date (0..D-1).
        # Used by the loss for per-day correlation pooling.
        train_df = train_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        val_df = val_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        tr_dates = train_df["date"].unique()
        val_dates = val_df["date"].unique()
        tr_d2i = {d: i for i, d in enumerate(tr_dates)}
        va_d2i = {d: i for i, d in enumerate(val_dates)}

        X_tr = torch.from_numpy(train_df[feats].to_numpy(dtype=np.float32))
        Y_tr = torch.from_numpy(train_df[rank_targets].to_numpy(dtype=np.float32))
        D_tr = torch.from_numpy(train_df["date"].map(tr_d2i).to_numpy(dtype=np.int64))

        X_va = torch.from_numpy(val_df[feats].to_numpy(dtype=np.float32))
        Y_va = torch.from_numpy(val_df[rank_targets].to_numpy(dtype=np.float32))
        D_va = torch.from_numpy(val_df["date"].map(va_d2i).to_numpy(dtype=np.int64))

        model = MultiTaskMLP(
            in_dim=len(feats),
            n_heads=len(targets),
            hidden=self.hidden,
            dropout=self.dropout,
        )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Day-aware mini-batches: pick `batch_days` random distinct training
        # days per step so per-day correlation loss has well-formed groups.
        n_train_days = len(tr_dates)
        steps_per_epoch = max(1, n_train_days // self.batch_days)

        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.epochs):
            model.train()
            day_idx = np.random.permutation(n_train_days)
            for s in range(steps_per_epoch):
                pick = day_idx[s * self.batch_days:(s + 1) * self.batch_days]
                if len(pick) == 0:
                    continue
                m = torch.from_numpy(np.isin(D_tr.numpy(), pick))
                xb, yb, db = X_tr[m], Y_tr[m], D_tr[m]
                preds = model(xb)  # (B, H)
                # Sum loss across heads
                loss = sum(
                    _per_day_rank_corr_loss(preds[:, h], yb[:, h], db)
                    for h in range(len(targets))
                ) / len(targets)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # validation: per-head per-day rank corr averaged
            model.eval()
            with torch.no_grad():
                vp = model(X_va)
                vloss = sum(
                    _per_day_rank_corr_loss(vp[:, h], Y_va[:, h], D_va)
                    for h in range(len(targets))
                ).item() / len(targets)
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Validation IC for diagnostics (against the canonical 5d target).
        model.eval()
        with torch.no_grad():
            vp = model(X_va).numpy()
        # Average per-head z-scored predictions then correlate vs raw 5d return.
        zp = np.zeros_like(vp)
        for h in range(vp.shape[1]):
            col = vp[:, h]
            zp[:, h] = (col - col.mean()) / (col.std() + 1e-8)
        avg_pred = zp.mean(axis=1)
        target_5d = "target_5d"
        # use the raw 5d *return* (not its rank) so the IC is comparable to
        # XGBoost strategies which all report rank-IC vs raw forward return.
        val_ic = rank_ic(
            val_df[target_5d].to_numpy(),
            avg_pred,
            val_df["date"].to_numpy(),
        )

        # Predict at as_of.
        pred_df = self._prediction_frame_fn()(panel, as_of=as_of)
        if pred_df.empty:
            raise RuntimeError(f"No prediction rows on {as_of.date()}")
        Xp = torch.from_numpy(pred_df[feats].to_numpy(dtype=np.float32))
        with torch.no_grad():
            yp = model(Xp).numpy()
        # Per-head cross-sectional z-score then average -> single score per stock.
        zp = np.zeros_like(yp)
        for h in range(yp.shape[1]):
            col = yp[:, h]
            zp[:, h] = (col - col.mean()) / (col.std() + 1e-8)
        scores = pd.Series(zp.mean(axis=1), index=pred_df["stock_code"].values,
                           name="score")
        diag = {
            "val_ic": val_ic,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_pred": len(pred_df),
            "n_features": len(feats),
            "n_heads": len(targets),
            "best_val_loss": float(best_val),
            "epochs_run": int(epoch + 1),
            "train_end": train_end.date().isoformat(),
            "val_start": val_start.date().isoformat(),
        }
        return scores, diag

    def fit_predict(self, panel, as_of, top_k=DEFAULT_TOP_K):
        scores, diag = self.fit_predict_scores(panel, as_of)
        weights = build_portfolio(scores, top_k=top_k)
        return StrategyResult(weights=weights, diagnostics=diag)
