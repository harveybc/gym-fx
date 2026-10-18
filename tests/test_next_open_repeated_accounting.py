"""Round-trip equity must reconcile after synthetic and ordinary closures."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shared_execution_envelope import _env


@pytest.mark.parametrize('direction', [1., -1.])
@pytest.mark.parametrize('leverage', [1., 5.])
def test_entry_bar_then_ordinary_stop_reconciles_both_trades(tmp_path, direction, leverage):
    highs, lows = [100.05] * 40, [99.95] * 40
    for index in (1, 5):
        (lows if direction > 0 else highs)[index] = 94. if direction > 0 else 106.
    env = _env(tmp_path, [100.] * 40, highs=highs, lows=lows,
               execution_transition='next_open_v2', commission=.001, leverage=leverage)
    try:
        env.reset(seed=7)
        for action in (direction, 0., direction, 0., 0.):
            env.step([action])
        assert env.bridge.bar_index == 6
        assert env.bridge.position_units == 0
        events = env.bridge.closed_trade_stream
        assert len(events) == 2
        assert sum(e['net_pnl'] for e in events) == pytest.approx(env.bridge.equity - 10000)
        assert sum(e['costs'] for e in events) == pytest.approx(env.bridge.commission_paid)
        assert env.summary()['trades_total'] == 2
    finally:
        env.close()


@pytest.mark.parametrize('direction', [1., -1.])
@pytest.mark.parametrize('leverage', [1., 5.])
def test_ordinary_close_reconciles_slippage_and_financing(tmp_path, direction, leverage):
    env = _env(tmp_path, [100.] * 40, execution_transition='next_open_v2',
               commission=.001, leverage=leverage, slippage_perc=.0001,
               max_entries_per_day=1, financing_enabled=True,
               financing_model='backtrader_annual_notional_daily_v1',
               financing_annual_rate=.365)
    try:
        env.reset(seed=7)
        env.step([direction * .5])
        size = abs(env.bridge.position_units)
        entry = env.bridge.entry_price
        assert direction * (entry - 100.) > 0
        for _ in range(12):
            env.step([0.])
        env.step([-direction * .5])
        assert env.bridge.position_units == 0
        assert len(env.bridge.closed_trade_stream) == 1
        event = env.bridge.closed_trade_stream[0]
        exit_price = event['exit_price']
        assert direction * (100. - exit_price) > 0
        fees = size * (entry + exit_price) * .001
        financing = size * entry * .365 / 365 * 2
        gross = direction * size * (exit_price - entry)
        assert event['net_pnl'] == pytest.approx(gross - fees - financing)
        assert env.bridge.equity == pytest.approx(10000 + gross - fees - financing)
    finally:
        env.close()


def test_exact_zero_net_pnl_does_not_become_gross_profit():
    from app.bt_bridge import BTBridge, BTBridgeStrategy
    bridge = BTBridge(10000)
    strategy = SimpleNamespace(bridge=bridge, _transition='next_open_v2', data=[0, 0],
        _completed_fills=[{'size': -1., 'price': 101., 'order_ref': 7, 'bar_index': 2}])
    trade = SimpleNamespace(isclosed=True, pnl=1., pnlcomm=0., price=100.,
        long=True, ref=1, history=[SimpleNamespace(event=SimpleNamespace(order=SimpleNamespace(ref=7)))])
    BTBridgeStrategy.notify_trade(strategy, trade)
    event = bridge.closed_trade_stream[0]
    assert event['net_pnl'] == 0.
    assert event['costs'] == 1.
