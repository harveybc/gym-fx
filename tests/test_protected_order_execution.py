from __future__ import annotations

from types import SimpleNamespace

import backtrader as bt
import pytest

from app.bt_bridge import BTBridgeStrategy
from strategy_plugins.direct_atr_sltp import Plugin


class _Line:
    def __init__(self, value: float):
        self.value = value

    def __getitem__(self, index: int) -> float:
        assert index == 0
        return self.value


class _Order:
    def __init__(self) -> None:
        self.live = True

    def alive(self) -> bool:
        return self.live


class _Strategy:
    def __init__(self, raw_action: float = 0.5):
        self.data = SimpleNamespace(
            high=_Line(101.0),
            low=_Line(99.0),
            close=_Line(100.0),
        )
        self.position = SimpleNamespace(size=0)
        self.broker = SimpleNamespace(getcash=lambda: 10_000.0)
        self.bridge = SimpleNamespace(
            raw_action_slot=raw_action,
            execution_diagnostics={},
        )
        self.submitted: list[tuple[str, dict]] = []
        self.cancelled: list[_Order] = []
        self.closed = 0

    def buy_bracket(self, **kwargs):
        self.submitted.append(("buy", kwargs))
        return [_Order(), _Order(), _Order()]

    def sell_bracket(self, **kwargs):
        self.submitted.append(("sell", kwargs))
        return [_Order(), _Order(), _Order()]

    def cancel(self, order):
        order.live = False
        self.cancelled.append(order)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize(
    ("mode", "expected_exectype", "has_price"),
    [
        ("market", bt.Order.Market, False),
        ("limit", bt.Order.Limit, True),
        ("stop", bt.Order.Stop, True),
    ],
)
def test_every_entry_type_is_submitted_as_protected_bracket(
    mode: str,
    expected_exectype: int,
    has_price: bool,
) -> None:
    config = {
        "atr_period": 2,
        "position_size": 1.0,
        "entry_order_mode": mode,
        "k_sl": 2.0,
        "k_tp": 3.0,
    }
    plugin = Plugin(config)
    strategy = _Strategy()
    plugin.on_reset(strategy, config)

    plugin.apply_action(strategy, 0, config)
    plugin.apply_action(strategy, 0, config)
    plugin.apply_action(strategy, 1, config)

    assert len(strategy.submitted) == 1
    side, order = strategy.submitted[0]
    assert side == "buy"
    assert order["exectype"] == expected_exectype
    assert ("price" in order) is has_price
    assert order["stopprice"] < (order.get("price") or 100.0)
    assert order["limitprice"] > (order.get("price") or 100.0)
    assert order["stopexec"] == bt.Order.Stop
    assert order["limitexec"] == bt.Order.Limit


def test_opposite_signal_cancels_protection_and_closes_before_reversal() -> None:
    config = {
        "atr_period": 2,
        "position_size": 1.0,
        "entry_order_mode": "market",
    }
    plugin = Plugin(config)
    strategy = _Strategy(raw_action=1.0)
    plugin.on_reset(strategy, config)
    plugin.apply_action(strategy, 0, config)
    plugin.apply_action(strategy, 0, config)
    plugin.apply_action(strategy, 1, config)
    strategy.position.size = 1

    plugin.apply_action(strategy, 2, config)

    assert len(strategy.cancelled) == 3
    assert strategy.closed == 1
    assert len(strategy.submitted) == 1


def test_plugin_failure_cannot_fall_back_to_naked_entry() -> None:
    submitted: list[str] = []

    def broken_plugin(*args, **kwargs):
        raise RuntimeError("broken protected execution")

    strategy = SimpleNamespace(
        _order_cost_accum=0.0,
        _plugin_apply=broken_plugin,
        _plugin_config={},
        _require_protected_entries=True,
        bridge=SimpleNamespace(
            execution_diagnostics={
                "plugin_apply_errors": 0,
                "protected_entry_rejections": 0,
            }
        ),
        position=SimpleNamespace(size=0),
        p=SimpleNamespace(position_size=1.0),
        buy=lambda **kwargs: submitted.append("buy"),
        sell=lambda **kwargs: submitted.append("sell"),
        close=lambda: submitted.append("close"),
    )

    BTBridgeStrategy._apply_action(strategy, 1)

    assert submitted == []
    assert strategy.bridge.execution_diagnostics["plugin_apply_errors"] == 1
    assert strategy.bridge.execution_diagnostics["protected_entry_rejections"] == 1
