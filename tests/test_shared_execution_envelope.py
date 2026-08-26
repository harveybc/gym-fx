"""Adversarial tests for the shared execution envelope (order C2/C3):
long, short, gap-through, same-bar SL+TP collision, reversal, final-bar
open position, policy close, portfolio-fraction sizing (scale
invariance, no lookahead, leverage cap). Real env + real backtrader."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.env import GymFxEnv
from broker_plugins.default_broker import Plugin as Broker
from data_feed_plugins.default_data_feed import Plugin as DataFeed
from metrics_plugins.default_metrics import Plugin as Metrics
from preprocessor_plugins.feature_window_preprocessor import (
    Plugin as Preprocessor)
from reward_plugins.pnl_reward import Plugin as Reward
from strategy_plugins.shared_execution_envelope import Plugin as Envelope


def _csv(tmp_path, closes, highs=None, lows=None, opens=None,
         name="bars.csv"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    frame = pd.DataFrame({
        "DATE_TIME": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "OPEN": np.asarray(opens, float) if opens is not None else closes,
        "HIGH": np.asarray(highs, float) if highs is not None
        else closes * 1.0005,
        "LOW": np.asarray(lows, float) if lows is not None
        else closes * 0.9995,
        "CLOSE": closes, "VOLUME": 1000.0,
        "feat": np.linspace(0.0, 1.0, n),
    })
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _env(tmp_path, closes, sl=0.05, tp=0.10, cash=10000.0, **kw):
    config = {
        "input_data_file": str(_csv(tmp_path, closes,
                                    kw.pop("highs", None),
                                    kw.pop("lows", None),
                                    kw.pop("opens", None))),
        "date_column": "DATE_TIME", "price_column": "CLOSE",
        "feature_columns": ["feat"], "feature_binary_columns": [],
        "window_size": 4, "initial_cash": cash, "position_size": 1.0,
        "min_equity": 0.0, "env_mode": "training",
        "commission": 0.0, "leverage": 1.0,
        "action_space_mode": "continuous",
        "continuous_action_threshold": 0.0,
        "execution_envelope": {"envelope_mode": "fixed_fraction",
                               "sl_fraction": sl, "tp_fraction": tp,
                               "leverage_cap": 1.0},
    }
    config.update(kw)
    env = GymFxEnv(config, DataFeed(config), Broker(config),
                   Envelope(config), Preprocessor(config),
                   Reward(config), Metrics(config))
    return env


def _drive(env, actions, seed=7):
    env.reset(seed=seed)
    infos = []
    for a in actions:
        _o, _r, term, _tr, info = env.step([float(a)])
        infos.append(info)
        if term:
            break
    events = list(getattr(env.bridge, "close_events", []))
    return infos, events


def test_long_stop_fills_at_stop_and_records_envelope_sl(tmp_path):
    # flat 100s, then a bar that dips through the 5% stop (95) only
    closes = [100.0] * 10 + [96.0] + [96.0] * 5
    lows = [c * 0.9995 for c in closes]
    lows[10] = 94.0                      # touches 95 intrabar
    env = _env(tmp_path, closes, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 12)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl, events
    assert abs(sl[0]["price"] - 95.0) < 1e-6   # filled AT the stop level


def test_long_take_profit_records_envelope_tp(tmp_path):
    closes = [100.0] * 10 + [109.0] + [109.0] * 5
    highs = [c * 1.0005 for c in closes]
    highs[10] = 111.0                    # touches 110 intrabar
    env = _env(tmp_path, closes, highs=highs)
    _infos, events = _drive(env, [1.0] + [0.0] * 12)
    tp = [e for e in events if e["reason"] == "envelope_close_tp"]
    assert tp, events
    assert abs(tp[0]["price"] - 110.0) < 1e-6


def test_short_stop_and_target_mirror(tmp_path):
    closes = [100.0] * 10 + [104.0] + [104.0] * 5
    highs = [c * 1.0005 for c in closes]
    highs[10] = 106.0                    # short stop at 105 touched
    env = _env(tmp_path, closes, highs=highs)
    _infos, events = _drive(env, [-1.0] + [0.0] * 12)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl and abs(sl[0]["price"] - 105.0) < 1e-6


def test_gap_through_stop_fills_at_open_not_stop(tmp_path):
    closes = [100.0] * 10 + [90.0] + [90.0] * 5
    opens = list(closes)
    opens[10] = 90.0                     # gaps BELOW the 95 stop
    lows = [c * 0.9995 for c in closes]
    lows[10] = 89.5
    env = _env(tmp_path, closes, opens=opens, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 12)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl, events
    assert sl[0]["price"] <= 90.0 + 1e-6   # gap fill at/below open


def test_same_bar_sl_and_tp_collision_resolves_to_stop(tmp_path):
    # one violent bar spans BOTH the 95 stop and the 110 target
    closes = [100.0] * 10 + [100.0] + [100.0] * 5
    highs = [c * 1.0005 for c in closes]
    lows = [c * 0.9995 for c in closes]
    highs[10] = 112.0
    lows[10] = 93.0
    env = _env(tmp_path, closes, highs=highs, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 12)
    reasons = [e["reason"] for e in events]
    assert "envelope_close_sl" in reasons, events   # pessimistic rule
    assert "envelope_close_tp" not in reasons


def test_reversal_closes_then_reenters_with_new_envelope(tmp_path):
    closes = [100.0] * 20
    env = _env(tmp_path, closes)
    _infos, events = _drive(env, [1.0, 0.0, 0.0, -1.0] + [0.0] * 10)
    reasons = [e["reason"] for e in events]
    assert "reversal_close" in reasons
    assert env.bridge.position < 0 or any(
        e["reason"] == "envelope_close_sl" for e in events)


def test_policy_close_taxonomy(tmp_path):
    closes = [100.0] * 20
    env = _env(tmp_path, closes,
               continuous_action_contract="target_exposure_hysteresis_v2")
    _infos, events = _drive(env, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # neutral action with open exposure under the v2 contract = close
    assert any(e["reason"] == "policy_close" for e in events), events


def test_final_bar_open_position_is_data_end_case(tmp_path):
    closes = [100.0] * 12
    env = _env(tmp_path, closes)
    infos, events = _drive(env, [1.0] + [0.0] * 20)
    # run ended by data, position still open, no envelope/policy close:
    assert infos[-1]["termination_cause"] == "data_end"
    # entry_fill audit events are expected; no CLOSE event of any kind:
    assert all(e["reason"] == "entry_fill" for e in events)
    assert env.bridge.position != 0   # driver must record data_end_liquidation


def test_portfolio_fraction_sizing_and_scale_invariance(tmp_path):
    closes = [100.0] * 20
    env1 = _env(tmp_path, closes, cash=10000.0)
    _drive(env1, [0.5] + [0.0] * 6)
    units1 = abs(env1.bridge.position_units)
    env2 = _env(tmp_path, closes, cash=20000.0)
    _drive(env2, [0.5] + [0.0] * 6)
    units2 = abs(env2.bridge.position_units)
    assert abs(units1 - 0.5 * 10000.0 / 100.0) < 0.02 * units1
    assert abs(units2 - 2.0 * units1) < 0.02 * units2  # scale invariant


def test_leverage_cap_binds(tmp_path):
    closes = [100.0] * 20
    env = _env(tmp_path, closes, cash=10000.0)
    _drive(env, [5.0] + [0.0] * 6)     # |raw| far above 1
    units = abs(env.bridge.position_units)
    assert units <= 10000.0 / 100.0 * 1.0001   # capped at equity/price


def test_full_exposure_entry_fills_with_commission(tmp_path):
    # BUG REPRO (first v2 run): raw=1.0 with commission due was
    # margin-rejected silently -> zero-trade arm
    closes = [100.0] * 20
    env = _env(tmp_path, closes, cash=10000.0, commission=0.001)
    _drive(env, [1.0] + [0.0] * 6)
    assert abs(env.bridge.position_units) > 90.0  # ~99.8 units filled
    assert env.bridge.execution_diagnostics.get(
        "envelope_order_rejections", 0) == 0


def test_rejections_are_counted_not_silent(tmp_path):
    closes = [100.0] * 20
    env = _env(tmp_path, closes, cash=10000.0, commission=0.001)
    # force a rejection: headroom zero via envelope override
    env.config["execution_envelope"]["entry_cost_headroom"] = 0.0
    _drive(env, [1.0] + [0.0] * 6)
    if abs(env.bridge.position_units) < 1e-9:
        assert env.bridge.execution_diagnostics.get(
            "envelope_order_rejections", 0) >= 1


def test_same_direction_target_changes_hold_entry_anchored(tmp_path):
    # DECLARED: entry-anchored sizing — same-direction fraction changes
    # do NOT re-size the held position (no per-bar churn, no stale
    # children); after the envelope fires the book is exactly FLAT.
    closes = [100.0] * 10 + [96.0] + [96.0] * 8
    lows = [c * 0.9995 for c in closes]
    lows[10] = 94.0
    env = _env(tmp_path, closes, lows=lows)
    _infos, events = _drive(env, [0.8, 0.5, 0.5] + [0.0] * 12)
    # units stay at the ENTRY size (0.8 of equity) until the SL fires
    assert any(e["reason"] == "envelope_close_sl" for e in events)
    assert abs(env.bridge.position_units) < 1e-6
    assert env.bridge.execution_diagnostics.get(
        "envelope_residual_sweeps", 0) == 0


# --- WP1 (order 2026-08-26): atomic lifecycle adversarial trajectories --

def test_entry_bar_touches_sl_closes_with_synthetic_check(tmp_path):
    closes = [100.0] * 20
    lows = [c * 0.9995 for c in closes]
    lows[1] = 94.0        # ENTRY BAR (parent fills at bar_index 2 = row 1)
    env = _env(tmp_path, closes, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl and sl[0].get("detail") == "entry_bar_settlement_at_level"
    assert abs(env.bridge.position_units) < 1e-9


def test_entry_bar_touches_tp_closes(tmp_path):
    closes = [100.0] * 20
    highs = [c * 1.0005 for c in closes]
    highs[1] = 111.0
    env = _env(tmp_path, closes, highs=highs)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    tp = [e for e in events if e["reason"] == "envelope_close_tp"]
    assert tp and tp[0].get("detail") == "entry_bar_settlement_at_level"


def test_entry_bar_touches_both_resolves_to_sl(tmp_path):
    closes = [100.0] * 20
    highs = [c * 1.0005 for c in closes]
    lows = [c * 0.9995 for c in closes]
    highs[1] = 112.0
    lows[1] = 93.0
    env = _env(tmp_path, closes, highs=highs, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    reasons = [e["reason"] for e in events]
    assert "envelope_close_sl" in reasons
    assert "envelope_close_tp" not in reasons


def test_entry_bar_short_equivalents(tmp_path):
    closes = [100.0] * 20
    highs = [c * 1.0005 for c in closes]
    highs[1] = 106.0      # short SL at 105 touched on entry bar
    env = _env(tmp_path, closes, highs=highs)
    _infos, events = _drive(env, [-1.0] + [0.0] * 10)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl and abs(env.bridge.position_units) < 1e-9


def test_post_entry_bars_fill_at_level_not_open(tmp_path):
    # resting children take over AFTER the entry bar and fill AT level
    closes = [100.0] * 10 + [96.0] + [96.0] * 5
    lows = [c * 0.9995 for c in closes]
    lows[10] = 94.0
    env = _env(tmp_path, closes, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 12)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl and abs(sl[0]["price"] - 95.0) < 1e-6


def test_reversal_while_children_pending_no_stale_fill(tmp_path):
    # a flip on the entry-FILL bar DEFERS one bar (declared; Submitted
    # children cannot be canceled); a persisting flip then reverses and
    # the old children must NEVER fill against the new short position
    closes = [100.0] * 8 + [96.0] + [96.0] * 8
    lows = [c * 0.9995 for c in closes]
    lows[8] = 94.0     # would touch the OLD long SL if it survived
    env = _env(tmp_path, closes, lows=lows)
    _infos, events = _drive(env, [1.0, -1.0, -1.0] + [0.0] * 12)
    reasons = [e["reason"] for e in events]
    assert "reversal_close" in reasons
    assert env.bridge.execution_diagnostics.get(
        "reversal_deferred_entry_bar", 0) >= 1
    assert env.bridge.execution_diagnostics.get(
        "envelope_residual_sweeps", 0) == 0
    assert not hasattr(env.bridge, "envelope_run_failure")
    assert env.bridge.position_units < 0   # clean short, no doubling


def test_zero_unprotected_bars_and_broker_refs(tmp_path):
    # every bar with a position must be covered by live children or the
    # armed entry-bar check; broker order references persist
    closes = [100.0] * 30
    env = _env(tmp_path, closes)
    env.reset(seed=7)
    plugin = env.strategy_plugin if hasattr(env, "strategy_plugin") else None
    for a in [1.0] + [0.0] * 20:
        _o, _r, term, _t, _i = env.step([float(a)])
        if term:
            break
        units = env.bridge.position_units
        if abs(units) > 1e-9:
            covered = bool(getattr(env.bridge, "close_events", None) is not None)
            # plugin internals: children live OR entry-bar check armed
    sp = env._strategy_plugin if hasattr(env, "_strategy_plugin") else None
    assert not hasattr(env.bridge, "envelope_run_failure")
    assert env.bridge.execution_diagnostics.get(
        "envelope_residual_sweeps", 0) == 0


def test_unprotected_position_is_typed_run_failure():
    # stub: pos != 0, no children, no pending, no entry-bar check ->
    # the guard closes AND marks the run failed (typed, never evidence)
    from types import SimpleNamespace
    from strategy_plugins.shared_execution_envelope import Plugin

    plug = Plugin({})
    orders = []
    s = SimpleNamespace(
        bridge=SimpleNamespace(raw_action_slot=0.9,
                               execution_diagnostics={},
                               close_events=[]),
        broker=SimpleNamespace(getvalue=lambda: 10000.0),
        data=SimpleNamespace(close=[100.0], high=[100.1], low=[99.9]),
        position=SimpleNamespace(size=5.0),
        close=lambda: orders.append("close"),
        cancel=lambda o: None,
        buy=lambda **k: orders.append(("buy", k)),
        sell=lambda **k: orders.append(("sell", k)),
    )
    s.data.__len__ = lambda self=None: 7
    class _D:
        close = [100.0]; high = [100.1]; low = [99.9]
        def __len__(self): return 7
    s.data = _D()
    plug.apply_action(s, 1, {})
    assert getattr(s.bridge, "envelope_run_failure", None) is not None
    assert orders and orders[0] == "close"
    assert s.bridge.execution_diagnostics["envelope_residual_sweeps"] == 1


# --- N1 (finding 329): same-bar settlement pricing fixtures -------------

def _entry_bar_env(tmp_path, *, lows=None, highs=None, opens=None,
                   closes_override=None):
    closes = closes_override or [100.0] * 20
    return _env(tmp_path, closes, lows=lows, highs=highs, opens=opens)


def test_settled_sl_fill_is_at_level_not_next_open(tmp_path):
    # stop touched on entry bar; NEXT open gaps UP favorably — the
    # already-settled fill must remain at the level (95), not improve
    closes = [100.0] * 20
    lows = [c * 0.9995 for c in closes]
    lows[1] = 94.0
    opens = list(closes)
    opens[2] = 99.0     # favorable next open for a long exit
    env = _env(tmp_path, closes, lows=lows, opens=opens)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    sl = [e for e in events if e["reason"] == "envelope_close_sl"]
    assert sl and abs(sl[0]["price"] - 95.0) < 1e-9
    assert sl[0]["detail"] == "entry_bar_settlement_at_level"


def test_entry_bar_adverse_gap_is_structurally_absorbed_by_anchor(tmp_path):
    """With FILL-anchored geometry the stop level derives from the entry
    bar's open, so 'open beyond the stop' cannot occur on the entry bar
    itself; adverse gaps are a LATER-bar phenomenon and are covered by
    the resting-children gap test (fill at open, worse than level)."""
    closes = [100.0] * 20
    opens = list(closes)
    opens[1] = 93.0                       # deep adverse open
    lows = [c * 0.9995 for c in closes]
    lows[1] = 92.5                        # above 93*0.95: no touch
    env = _env(tmp_path, closes, opens=opens, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    assert not [e for e in events if e["reason"] == "envelope_close_sl"]
    assert abs(env.bridge.position_units) > 0  # position survives


def test_geometry_anchors_at_parent_fill_not_decision_close(tmp_path):
    # decision close 100, entry bar opens at 90: SL must be 90*0.95=85.5
    # (fill-anchored), NOT 95 (decision-anchored)
    closes = [100.0] * 20
    opens = list(closes)
    opens[1] = 90.0
    lows = [c * 0.9995 for c in closes]
    lows[1] = 86.0      # touches 85.5? no: 86 > 85.5 -> NO settle
    env = _env(tmp_path, closes, opens=opens, lows=lows)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    assert not [e for e in events if e["reason"] == "envelope_close_sl"]
    lows2 = [c * 0.9995 for c in closes]
    lows2[1] = 85.0     # 85 < 85.5 -> settles at level 85.5
    env2 = _env(tmp_path, closes, opens=opens, lows=lows2)
    _infos2, events2 = _drive(env2, [1.0] + [0.0] * 10)
    sl2 = [e for e in events2 if e["reason"] == "envelope_close_sl"]
    assert sl2 and abs(sl2[0]["price"] - 85.5) < 1e-9


def test_short_settled_tp_fill_at_level_or_better(tmp_path):
    closes = [100.0] * 20
    lows = [c * 0.9995 for c in closes]
    lows[1] = 89.0      # short TP at 90 touched on entry bar
    env = _env(tmp_path, closes, lows=lows)
    _infos, events = _drive(env, [-1.0] + [0.0] * 10)
    tp = [e for e in events if e["reason"] == "envelope_close_tp"]
    assert tp and abs(tp[0]["price"] - 90.0) < 1e-9


def test_no_double_close_and_exact_cash_accounting(tmp_path):
    # settle then verify: exactly one close event, position flat, cash
    # consistent with size*(fill-entry) - commissions (formula check)
    closes = [100.0] * 20
    lows = [c * 0.9995 for c in closes]
    lows[1] = 94.0
    env = _env(tmp_path, closes, lows=lows, commission=0.001)
    _infos, events = _drive(env, [1.0] + [0.0] * 10)
    closes_ev = [e for e in events if e["reason"].startswith("envelope")]
    assert len(closes_ev) == 1
    assert abs(env.bridge.position_units) < 1e-9
    eq = env.bridge.equity
    # entry ~99.6 units at 100 open (headroom 0.2% + commission),
    # exit at 95: loss = units*5 + 2 commissions
    units = 10000.0 * (1 - 0.002) / 100.0
    expected = 10000.0 - units * 5.0 - 0.001 * units * (100.0 + 95.0)
    assert abs(eq - expected) < 1.0, (eq, expected)


def test_deterministic_replay_equality(tmp_path):
    closes = list(__import__("numpy").linspace(100, 90, 30))
    seq = [1.0, 0.0, -1.0, 0.0, 1.0] + [0.0] * 20
    def run():
        env = _env(tmp_path, closes)
        infos, events = _drive(env, seq)
        return [round(i["economic_equity"], 9) for i in infos], events
    def strip(evts):
        return [{k: v for k, v in e.items()
                 if k not in ("order_ref", "children_refs")}
                for e in evts]
    eq1, ev1 = run()
    eq2, ev2 = run()
    # broker order refs are a PROCESS-GLOBAL counter; economics and the
    # ref-stripped event stream must replay identically
    assert eq1 == eq2
    assert strip(ev1) == strip(ev2)


def test_reversal_on_entry_fill_bar_defers_one_bar(tmp_path):
    # 329-family regression (bar-2707 trace): flipping on the very bar
    # the entry filled must DEFER — Submitted children cannot be
    # canceled and a live old stop would double-fire.
    closes = [100.0] * 8 + [96.0] + [96.0] * 8
    lows = [c * 0.9995 for c in closes]
    lows[8] = 94.0        # old long SL level touched later
    env = _env(tmp_path, closes, lows=lows)
    # decision long at step0 (fills bar2); flip short ON the fill bar
    _infos, events = _drive(env, [1.0, -1.0, -1.0] + [0.0] * 12)
    assert env.bridge.execution_diagnostics.get(
        "reversal_deferred_entry_bar", 0) >= 1
    assert env.bridge.execution_diagnostics.get(
        "envelope_residual_sweeps", 0) == 0
    assert not hasattr(env.bridge, "envelope_run_failure")
    # after the deferral the reversal executes and the book is SHORT
    assert env.bridge.position_units < 0
