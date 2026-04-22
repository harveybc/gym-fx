#!/usr/bin/env python
"""Smoke-test to catch memory blowups early.

Runs a few build+train iterations and prints RSS + GPU memory info.
Use this before launching multi-day GA runs.

Example:
  source set_env.sh
  python tools/oom_smoke_test.py --config examples/config/phase_1_daily/phase_1_cnn_25200_1d_config.json --iters 5 --epochs 2

Notes:
- Uses disable_postfit_uncertainty=True to avoid MC inference.
- Uses batched prediction for safety.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import gc

import numpy as np


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = float(parts[1])
                    return kb / 1024.0
    except Exception:
        return float("nan")
    return float("nan")


def _gpu_mem_info() -> str:
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return "GPU: none"
        try:
            info = tf.config.experimental.get_memory_info("GPU:0")
            cur = info.get("current", 0) / (1024**2)
            peak = info.get("peak", 0) / (1024**2)
            return f"GPU:0 current={cur:.1f}MB peak={peak:.1f}MB"
        except Exception:
            return f"GPU count={len(gpus)} (no mem info)"
    except Exception as e:
        return f"GPU info unavailable: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Safety knobs
    cfg["disable_postfit_uncertainty"] = True
    cfg["mc_samples"] = 1
    cfg["predict_batch_size"] = int(cfg.get("batch_size", 32))

    from app.plugin_loader import load_plugin

    predictor_class, _ = load_plugin("ioin.plugins", cfg.get("predictor_plugin", cfg.get("plugin", "cnn")))
    preproc_class, _ = load_plugin("preprocessor.plugins", cfg.get("preprocessor_plugin", "stl_preprocessor"))
    target_class, _ = load_plugin("target.plugins", cfg.get("target_plugin", "default_target"))

    target = target_class()
    target.set_params(**cfg)

    pre = preproc_class()
    pre.set_params(**cfg)

    for i in range(args.iters):
        import tensorflow as tf

        tf.keras.backend.clear_session()
        gc.collect()

        pred = predictor_class()
        pred.set_params(**cfg)

        datasets = pre.run_preprocessing(target, cfg)
        if isinstance(datasets, tuple):
            datasets = datasets[0]

        x_train, y_train = datasets["x_train"], datasets["y_train"]
        x_val, y_val = datasets["x_val"], datasets["y_val"]

        window_size = int(cfg.get("window_size", x_train.shape[1]))
        input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)

        pred.build_model(input_shape=input_shape, x_train=x_train, config=cfg)

        t0 = time.time()
        pred.train(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=int(cfg.get("batch_size", 32)),
            threshold_error=float(cfg.get("threshold_error", 0.001)),
            x_val=x_val,
            y_val=y_val,
            config=cfg,
        )
        dt = time.time() - t0

        print(f"Iter {i+1}/{args.iters}: train_time={dt:.1f}s rss={_rss_mb():.1f}MB {_gpu_mem_info()}")

        # Cleanup
        try:
            if hasattr(pred, "model"):
                del pred.model
        except Exception:
            pass
        tf.keras.backend.clear_session()
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
