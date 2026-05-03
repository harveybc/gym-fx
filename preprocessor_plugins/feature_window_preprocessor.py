"""
feature_window_preprocessor.py

Observation builder that consumes a configured set of feature columns from
the dataframe and emits a `(window_size, n_features)` feature tensor for
the agent, alongside the standard agent-state signals from `default_preprocessor`.

Scaling policy (leakage-safe by construction):
  - "none":            raw column values
  - "rolling_zscore":  z-score using ONLY rows in [step - window, step)
                       (no future data; no train/test split required)
  - "expanding_zscore":z-score over rows [0, step)

Binary columns (configured via `feature_binary_columns`) are passed through
without scaling, preserving 0/1 or signed semantics.

Returns a Dict observation; SAC and DQN plugins flatten it via
FlattenObservation. No custom architecture required.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


_VALID_SCALINGS = ("none", "rolling_zscore", "expanding_zscore")


class Plugin:
    plugin_params: Dict[str, Any] = {
        "window_size": 32,
        "price_column": "CLOSE",
        "feature_columns": [],
        "feature_binary_columns": [],
        "feature_scaling": "rolling_zscore",
        "feature_scaling_window": 256,
        "include_price_window": True,
        "include_agent_state": True,
        "feature_clip": 10.0,
    }

    plugin_debug_vars: List[str] = [
        "window_size",
        "price_column",
        "feature_scaling",
        "feature_scaling_window",
        "include_price_window",
        "include_agent_state",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        self._cached_columns: List[str] | None = None
        self._cached_binary_mask: np.ndarray | None = None
        if config:
            self.set_params(**config)

    def set_params(self, **kwargs: Any) -> None:
        self.params.update(kwargs)

    def get_debug_info(self) -> Dict[str, Any]:
        info = {var: self.params.get(var) for var in self.plugin_debug_vars}
        info["n_features"] = len(self.params.get("feature_columns") or [])
        return info

    def add_debug_info(self, debug_info: Dict[str, Any]) -> None:
        debug_info.update(self.get_debug_info())

    # ------------------------------------------------------------------
    def _resolve_columns(
        self, data: pd.DataFrame, config: Dict[str, Any]
    ) -> tuple[List[str], np.ndarray]:
        cols: Sequence[str] = (
            config.get("feature_columns") or self.params["feature_columns"] or []
        )
        if not cols:
            raise ValueError(
                "feature_window_preprocessor requires non-empty 'feature_columns'."
            )
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(
                "feature_window_preprocessor: configured feature_columns "
                f"missing from dataframe: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        binary_cols = set(
            config.get("feature_binary_columns")
            or self.params["feature_binary_columns"]
            or []
        )
        binary_mask = np.array([c in binary_cols for c in cols], dtype=bool)
        return list(cols), binary_mask

    @staticmethod
    def _scale_window(
        feature_window: np.ndarray,
        history: np.ndarray,
        binary_mask: np.ndarray,
        mode: str,
        clip: float,
    ) -> np.ndarray:
        """Apply leakage-safe scaling. `history` is rows strictly before the
        current step (used to fit the scaler). Binary columns are passed
        through unchanged."""
        if mode == "none" or history.shape[0] < 2:
            scaled = feature_window.astype(np.float32)
        else:
            mean = history.mean(axis=0)
            std = history.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            scaled = (feature_window - mean) / std
            scaled = scaled.astype(np.float32)

        if binary_mask.any():
            scaled[:, binary_mask] = feature_window[:, binary_mask].astype(np.float32)

        if clip and clip > 0:
            np.clip(scaled, -clip, clip, out=scaled)
        # Last-resort guard (e.g. a constant binary column producing no NaN
        # since we bypassed scaling, but other columns might).
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=clip, neginf=-clip)
        return scaled

    def _feature_window(
        self,
        data: pd.DataFrame,
        step: int,
        cols: List[str],
        binary_mask: np.ndarray,
        config: Dict[str, Any],
    ) -> np.ndarray:
        window_size = int(config.get("window_size", self.params["window_size"]))
        scale_mode = str(
            config.get("feature_scaling", self.params["feature_scaling"])
        ).lower()
        if scale_mode not in _VALID_SCALINGS:
            raise ValueError(
                f"feature_scaling must be one of {_VALID_SCALINGS}; got {scale_mode!r}"
            )
        scale_window = int(
            config.get(
                "feature_scaling_window", self.params["feature_scaling_window"]
            )
        )
        clip = float(config.get("feature_clip", self.params["feature_clip"]))

        values = data[cols].to_numpy(dtype=np.float64, copy=False)
        n_rows, n_features = values.shape

        # Slice [step - window_size, step)
        left = max(0, step - window_size)
        win = values[left:step] if step > 0 else values[:0]
        if win.shape[0] < window_size:
            pad_row = win[0] if win.shape[0] else (
                values[0] if n_rows else np.zeros(n_features, dtype=np.float64)
            )
            pad = np.tile(pad_row, (window_size - win.shape[0], 1))
            win = np.concatenate([pad, win], axis=0)

        # History for scaler fit — strictly past rows only.
        if scale_mode == "rolling_zscore":
            hist_left = max(0, step - scale_window)
            history = values[hist_left:step]
        elif scale_mode == "expanding_zscore":
            history = values[:step]
        else:
            history = np.empty((0, n_features), dtype=np.float64)

        return self._scale_window(win, history, binary_mask, scale_mode, clip)

    # ------------------------------------------------------------------
    def make_observation(
        self,
        *,
        data: pd.DataFrame,
        step: int,
        bridge_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        cols, binary_mask = self._resolve_columns(data, config)
        window_size = int(config.get("window_size", self.params["window_size"]))
        price_col = config.get("price_column", self.params["price_column"])

        features = self._feature_window(data, step, cols, binary_mask, config)

        obs: Dict[str, np.ndarray] = {
            "features": features.astype(np.float32),
        }

        include_price = bool(
            config.get("include_price_window", self.params["include_price_window"])
        )
        if include_price:
            prices_full = data[price_col].astype(float).to_numpy()
            left = max(0, step - window_size)
            window = prices_full[left:step] if step > 0 else prices_full[:0]
            if len(window) < window_size:
                fill = float(window[0]) if len(window) else float(
                    prices_full[0] if len(prices_full) else 0.0
                )
                pad = np.full(window_size - len(window), fill, dtype=float)
                window = np.concatenate([pad, window])
            returns = np.diff(window, prepend=window[0])
            obs["prices"] = window.astype(np.float32)
            obs["returns"] = returns.astype(np.float32)

        if bool(config.get("include_agent_state", self.params["include_agent_state"])):
            initial_cash = float(bridge_state.get("initial_cash", 1.0) or 1.0)
            equity = float(bridge_state.get("equity", initial_cash))
            price = float(bridge_state.get("price", 0.0) or 0.0)
            position = int(bridge_state.get("position", 0))
            bar_index = int(bridge_state.get("bar_index", 0))
            total_bars = int(bridge_state.get("total_bars", 1) or 1)

            pos_size = float(config.get("position_size", 1.0))
            ref_price = (
                float(obs["prices"][-1])
                if include_price and obs["prices"].size
                else price
            )
            unrealized_pnl = position * (price - ref_price) * pos_size

            equity_norm = (equity - initial_cash) / initial_cash if initial_cash else 0.0
            pnl_norm = unrealized_pnl / initial_cash if initial_cash else 0.0
            remaining = max(0, total_bars - bar_index) / max(1, total_bars)

            obs["position"] = np.array([float(position)], dtype=np.float32)
            obs["equity_norm"] = np.array([float(equity_norm)], dtype=np.float32)
            obs["unrealized_pnl_norm"] = np.array([float(pnl_norm)], dtype=np.float32)
            obs["steps_remaining_norm"] = np.array([float(remaining)], dtype=np.float32)

        return obs
