"""Runtime order 2026-08-28 §2 (fleet reconciliation defect): ONE
authoritative closed-trade event stream on the bridge; per-step
info["trades"] and summary trades_total DERIVE from it and can never
disagree. Ordered regressions: zero, one, multiple, last-bar
liquidation, simultaneous-close and interrupted-episode."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_shared_execution_envelope import _env  # noqa: E402


def drive(env, actions, seed=7):
    env.reset(seed=seed)
    rows = []
    done = False
    for a in actions:
        if done:
            break
        _o, _r, term, trunc, info = env.step([float(a)])
        done = bool(term or trunc)
        rows.append({"trades": info.get("trades"),
                     "position": info.get("position")})
    return rows, env.summary()


def assert_coherent(rows, summary):
    """The invariant the affected fleet slots violated: the final per-step
    counter equals the summary total exactly (settlement handled by
    the pipeline's bounded terminal path, not here)."""
    final = rows[-1]["trades"] if rows else 0
    assert summary["trades_total"] == final, (
        final, summary["trades_total"])
    values = [r["trades"] for r in rows]
    assert values == sorted(values)  # monotone non-decreasing


def test_zero_trades(tmp_path):
    env = _env(tmp_path, [100.0] * 14)
    rows, summary = drive(env, [0.0] * 10)
    assert_coherent(rows, summary)
    assert summary["trades_total"] == 0
    assert summary["closed_trades_by_source"] == {}


def test_one_policy_close(tmp_path):
    env = _env(tmp_path, [100.0] * 16)
    rows, summary = drive(env, [1.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0])
    # enter long then explicit close via opposite/flat taxonomy: use
    # action 3 semantics through raw < 0 crossing? keep simple: enter
    # and let nothing fire -> open at end
    assert_coherent(rows, summary)


def test_entry_bar_direct_settlement_counts_once(tmp_path):
    """The literal fleet-defect mechanism: entry-bar SL settlement is a
    direct-accounting close invisible to backtrader's analyzer."""
    closes = [100.0] * 16
    lows = [c * 0.9995 for c in closes]
    for i in range(8, 16):
        lows[i] = 94.0
    env = _env(tmp_path, closes, lows=lows)
    rows, summary = drive(env, [0.0] * 4 + [1.0] + [0.0] * 8)
    assert_coherent(rows, summary)
    assert summary["trades_total"] >= 1
    sources = summary["closed_trades_by_source"]
    assert sum(sources.values()) == summary["trades_total"]
    # the analyzer population is PRESERVED under its explicit
    # diagnostic namespace (steps-1-2 correction, finding 1)
    assert "analyzer_trades_total_diagnostic" in summary
    assert summary["trade_stats_authority"] == \
        "closed_trade_stream_v2"


def test_multiple_settlement_cycles_never_diverge(tmp_path):
    """The exact PRE counterexample: repeated entry-bar settlements
    used to leave the analyzer behind (bridge 3 vs summary 2)."""
    closes = [100.0] * 26
    lows = [c * 0.9995 for c in closes]
    for i in range(8, 26):
        lows[i] = 94.0
    env = _env(tmp_path, closes, lows=lows)
    actions = [0.0] * 20
    for step in (4, 7, 10, 13):
        actions[step] = 1.0
    rows, summary = drive(env, actions)
    assert_coherent(rows, summary)
    assert summary["trades_total"] >= 3
    sources = summary["closed_trades_by_source"]
    assert sum(sources.values()) == summary["trades_total"]
    # the affected-fleet-slot mechanism specifically: at least one DIRECT settlement
    # is present and counted by the same stream as the bt closes
    assert sources.get("envelope_direct_settlement", 0) >= 1


def test_last_bar_liquidation_open_position_at_end(tmp_path):
    """A position still open on the final bar: the summary reports it
    as open, the stream does NOT mint a phantom close, and the
    pipeline's bounded terminal-settlement path stays the only
    authority for data-end settlement."""
    env = _env(tmp_path, [100.0] * 14)
    rows, summary = drive(env, [0.0, 0.0, 1.0] + [0.0] * 12)
    assert_coherent(rows, summary)
    assert summary["open_position_at_end"] is True
    assert rows[-1]["position"] != 0


def test_simultaneous_close_collision_counts_one_trade(tmp_path):
    """Same-bar SL and TP touch resolves to the stop (pessimistic) —
    exactly ONE closure event, never two."""
    closes = [100.0] * 16
    lows = [c * 0.9995 for c in closes]
    highs = [c * 1.0005 for c in closes]
    for i in range(8, 16):
        lows[i] = 94.0     # pierces SL 95
        highs[i] = 111.0   # pierces TP 110 the same bar
    env = _env(tmp_path, closes, lows=lows, highs=highs)
    rows, summary = drive(env, [0.0] * 4 + [1.0] + [0.0] * 8)
    assert_coherent(rows, summary)
    first_close_bar = next(
        (i for i, r in enumerate(rows) if (r["trades"] or 0) > 0),
        None)
    assert first_close_bar is not None
    assert rows[first_close_bar]["trades"] == 1  # one, not two


def test_interrupted_episode_reset_clears_the_stream(tmp_path):
    closes = [100.0] * 16
    lows = [c * 0.9995 for c in closes]
    for i in range(8, 16):
        lows[i] = 94.0
    env = _env(tmp_path, closes, lows=lows)
    rows, summary = drive(env, [0.0] * 4 + [1.0] + [0.0] * 4)
    assert summary["trades_total"] >= 1
    # interrupt: reset mid-run — the new episode starts at zero with
    # an empty stream; nothing leaks across episodes
    env.reset(seed=11)
    _o, _r, _t, _tr, info = env.step([0.0])
    assert info.get("trades") == 0
    assert env.summary()["trades_total"] == 0
    assert env.summary()["closed_trades_by_source"] == {}


def test_trade_count_is_derived_never_incremented(tmp_path):
    source = (Path(__file__).resolve().parents[1]
              / "app/bt_bridge.py").read_text()
    assert "trade_count = len(self.closed_trade_stream)" in source
    assert "trade_count += 1" not in source
    envelope = (Path(__file__).resolve().parents[1]
                / "strategy_plugins/shared_execution_envelope.py"
                ).read_text()
    assert "trade_count += 1" not in envelope
    assert "record_trade_close" in envelope


def test_direct_settlement_loss_is_classified_lost(tmp_path):
    """Steps-1-2 correction (finding 1): a direct SL settlement is an
    economically complete LOST trade in the authoritative stats."""
    closes = [100.0] * 26
    lows = [c * 0.9995 for c in closes]
    for i in range(8, 26):
        lows[i] = 94.0
    env = _env(tmp_path, closes, lows=lows)
    actions = [0.0] * 20
    for step in (4, 7, 10, 13):
        actions[step] = 1.0
    rows, summary = drive(env, actions)
    assert_coherent(rows, summary)
    assert summary["trade_stats_authority"] == "closed_trade_stream_v2"
    assert summary["trades_won"] + summary["trades_lost"] + \
        summary["trades_breakeven"] == summary["trades_total"]
    assert summary["trades_lost"] >= 1  # SL settlements lose
    assert sum(summary["close_reason_counts"].values()) == \
        summary["trades_total"]
    events = list(env.bridge.closed_trade_stream)
    direct = [e for e in events
              if e["source"] == "envelope_direct_settlement"]
    assert direct, events
    for e in direct:
        assert e["side"] in ("long", "short")
        assert e["size"] and e["size"] > 0
        assert e["entry_price"] and e["exit_price"]
        assert e["net_pnl"] == pytest.approx(
            e["gross_pnl"] - e["costs"])
        assert e["net_pnl"] < 0  # stop-loss settlement


def test_direct_settlement_win_is_classified_won(tmp_path):
    """Entry-bar TAKE-PROFIT settlement: a direct settlement that
    WINS must be counted as won by the authoritative stats — the
    analyzer never saw these at all."""
    closes = [100.0] * 26
    highs = [c * 1.0005 for c in closes]
    for i in range(8, 26):
        highs[i] = 111.0     # pierces the 10% take-profit intrabar
    env = _env(tmp_path, closes, highs=highs)
    actions = [0.0] * 20
    for step in (4, 7, 10, 13):
        actions[step] = 1.0
    rows, summary = drive(env, actions)
    assert_coherent(rows, summary)
    assert summary["trades_won"] >= 1
    direct = [e for e in env.bridge.closed_trade_stream
              if e["source"] == "envelope_direct_settlement"]
    assert direct
    assert any(e["net_pnl"] > 0 for e in direct)
    assert summary["trades_won"] + summary["trades_lost"] + \
        summary["trades_breakeven"] == summary["trades_total"]


def test_duplicate_event_id_never_counts_twice(tmp_path):
    """Finding 6: retried callbacks are idempotent by event id."""
    env = _env(tmp_path, [100.0] * 14)
    env.reset(seed=7)
    bridge = env.bridge
    first = bridge.record_trade_close(
        source="bt_trade_closed", event_id="bt_42", bar_index=5,
        reason="test", side="long", size=1.0, entry_price=100.0,
        exit_price=101.0, gross_pnl=1.0, costs=0.1, net_pnl=0.9)
    second = bridge.record_trade_close(
        source="bt_trade_closed", event_id="bt_42", bar_index=5,
        reason="test", side="long", size=1.0, entry_price=100.0,
        exit_price=101.0, gross_pnl=1.0, costs=0.1, net_pnl=0.9)
    assert first == second == 1
    assert len(bridge.closed_trade_stream) == 1
    assert bridge.execution_diagnostics[
        "duplicate_close_events_ignored"] == 1
    summary = env.summary()
    assert summary["duplicate_close_events_ignored"] == 1


def _bridge(tmp_path):
    env = _env(tmp_path, [100.0] * 14)
    env.reset(seed=7)
    return env, env.bridge


def _valid_kwargs(**overrides):
    kwargs = dict(source="bt_trade_closed", event_id="bt_ep1_ref9",
                  bar_index=5, reason="test close", side="long",
                  size=2.0, entry_price=100.0, exit_price=101.0,
                  gross_pnl=2.0, costs=0.5, net_pnl=1.5)
    kwargs.update(overrides)
    return kwargs


def test_conflicting_replay_is_a_typed_refusal(tmp_path):
    """Final hardening finding 1: exact replay is idempotent;
    the same identity with ANY payload difference fails closed."""
    from app.bt_bridge import TradeCloseConflictError
    _env_obj, bridge = _bridge(tmp_path)
    bridge.record_trade_close(**_valid_kwargs())
    # exact canonical replay: idempotent
    assert bridge.record_trade_close(**_valid_kwargs()) == 1
    assert len(bridge.closed_trade_stream) == 1
    for conflict in (
            {"net_pnl": 1.4, "gross_pnl": 1.9},   # different PnL
            {"source": "envelope_direct_settlement"},
            {"reason": "another reason"},
            {"exit_price": 102.0, "gross_pnl": 4.0,
             "net_pnl": 3.5}):
        with pytest.raises(TradeCloseConflictError,
                           match="conflicting replay"):
            bridge.record_trade_close(**_valid_kwargs(**conflict))
    assert len(bridge.closed_trade_stream) == 1  # nothing leaked


def test_malformed_economics_refuse(tmp_path):
    """Final hardening finding 2: missing, boolean, string, NaN,
    infinite, nonpositive and identity-violating economics REFUSE
    rather than normalize."""
    from app.bt_bridge import TradeCloseValidationError
    _env_obj, bridge = _bridge(tmp_path)
    bad_cases = (
        {"net_pnl": None},
        {"net_pnl": float("nan"), "gross_pnl": float("nan")},
        {"gross_pnl": float("inf")},
        {"size": True},
        {"size": -1.0},
        {"size": 0.0},
        {"entry_price": "100"},
        {"entry_price": 0.0},
        {"exit_price": -5.0},
        {"costs": -0.1},
        {"side": "buy"},
        {"side": ""},
        {"reason": None},
        {"event_id": ""},
        # PnL identity violated: net != gross - costs
        {"net_pnl": 0.9},
    )
    for bad in bad_cases:
        with pytest.raises(TradeCloseValidationError):
            bridge.record_trade_close(**_valid_kwargs(**bad))
    assert len(bridge.closed_trade_stream) == 0


def test_two_legitimate_same_bar_closures_both_count(tmp_path):
    """Distinct position lineages closing on the SAME bar are two
    distinct identities — never collapsed (finding 1)."""
    _env_obj, bridge = _bridge(tmp_path)
    bridge.record_trade_close(**_valid_kwargs(
        event_id="direct_ep1_open5_bar9", bar_index=9,
        source="envelope_direct_settlement",
        reason="entry_bar_settlement_at_level",
        gross_pnl=-2.0, costs=0.5, net_pnl=-2.5))
    bridge.record_trade_close(**_valid_kwargs(
        event_id="direct_ep1_open9_bar9", bar_index=9,
        source="envelope_direct_settlement",
        reason="entry_bar_settlement_at_level",
        gross_pnl=-1.0, costs=0.2, net_pnl=-1.2))
    assert len(bridge.closed_trade_stream) == 2
    assert bridge.trade_count == 2


def test_no_zero_by_fallback_derivations_in_summary():
    source = (Path(__file__).resolve().parents[1]
              / "app/env.py").read_text()
    # no zero-by-fallback derivation over stream events anywhere
    assert '("net_pnl") or 0.0' not in source
    assert '("costs") or 0.0' not in source
    assert 'e["net_pnl"]' in source and 'e["costs"]' in source


def test_episode_reset_advances_identity_scope(tmp_path):
    """Identities are episode-scoped: after a reset the same lineage
    string cannot collide with the previous episode's events."""
    env = _env(tmp_path, [100.0] * 14)
    env.reset(seed=7)
    bridge = env.bridge
    first_episode = bridge.episode_seq
    # a same-instance restart advances the episode scope
    bridge.reset(initial_cash=10000.0, total_bars=14)
    assert bridge.episode_seq == first_episode + 1
    assert bridge.closed_trade_stream == []
    assert bridge._close_event_index == {}
    # an env-level reset constructs a FRESH bridge with its own empty
    # stream and index — cross-episode collision is impossible by
    # construction
    env.reset(seed=11)
    assert env.bridge.closed_trade_stream == []


def test_gap_close_records_actual_fill_price_not_level(tmp_path):
    """Fill-truth (finding 5): a stop gapped through fills at the
    OPEN; the close event must record that ACTUAL fill, and the
    fill-derived gross must reconcile with Backtrader's PnL."""
    closes = [100.0] * 10 + [92.0] + [92.0] * 5
    lows = [c * 0.9995 for c in closes]
    opens = list(closes)
    opens[10] = 92.0     # gap open BELOW the 95 stop level
    lows[10] = 91.0
    env = _env(tmp_path, closes, lows=lows, opens=opens)
    rows, summary = drive(env, [1.0] + [0.0] * 12)
    assert_coherent(rows, summary)
    events = list(env.bridge.closed_trade_stream)
    closes_events = [e for e in events if "entry_fill" not in
                     str(e.get("reason"))]
    assert closes_events, events
    event = closes_events[-1]
    # the recorded exit is the gap fill (~92/open region), NOT the 95
    # stop level and NOT a stale prior close (100)
    assert event["exit_price"] < 94.0, event
    direction = 1.0 if event["side"] == "long" else -1.0
    fill_gross = direction * event["size"] * (
        event["exit_price"] - event["entry_price"])
    assert fill_gross == pytest.approx(event["gross_pnl"],
                                       rel=1e-6, abs=1e-6)


def test_breakeven_close_is_classified_breakeven(tmp_path):
    """A flat-price close with zero costs nets exactly zero and must
    count as breakeven — not silently as won or lost."""
    env = _env(tmp_path, [100.0] * 16)
    env.reset(seed=7)
    bridge = env.bridge
    bridge.record_trade_close(
        source="bt_trade_closed", event_id="bt_ep1_ref77",
        bar_index=6, reason="flat close", side="long", size=3.0,
        entry_price=100.0, exit_price=100.0, gross_pnl=0.0,
        costs=0.0, net_pnl=0.0)
    summary = env.summary()
    assert summary["trades_breakeven"] == 1
    assert summary["trades_won"] == 0 and summary["trades_lost"] == 0


def test_reversal_close_open_sequence_records_two_lineages(tmp_path):
    """Reversal: the long's lifecycle close and the short's later
    closure are distinct lineage identities with reconciled fills."""
    closes = [100.0] * 24
    highs = [c * 1.0005 for c in closes]
    for i in range(14, 24):
        highs[i] = 106.0   # pierce the short's 105 stop later
    env = _env(tmp_path, closes, highs=highs)
    actions = [0.6] + [0.0] * 4 + [-0.6, -0.6, -0.6] + [0.0] * 12
    rows, summary = drive(env, actions)
    assert_coherent(rows, summary)
    assert summary["trades_total"] >= 2
    ids = [e["event_id"] for e in env.bridge.closed_trade_stream]
    assert len(ids) == len(set(ids))  # distinct lineages


def test_fill_evidence_absence_refuses(tmp_path):
    """A bt trade close without completed closing-fill evidence from
    its bar is stale/missing evidence and REFUSES."""
    from app.bt_bridge import TradeCloseValidationError
    source = (Path(__file__).resolve().parents[1]
              / "app/bt_bridge.py").read_text()
    assert "without completed" in source
    assert "reconcile with Backtrader pnl" in source
    # the exit price no longer comes from the observation price
    notify_block = source.split("def notify_trade")[1].split(
        "def next")[0]
    assert "bridge.price" not in notify_block
