from __future__ import annotations

from types import SimpleNamespace

from strategy_plugins.direct_fixed_sltp import Plugin


class _Strategy:
    def __init__(self) -> None:
        self.data = SimpleNamespace(close=[100.0])
        self.position = SimpleNamespace(size=1.0)
        self._owned_orders = [object(), object()]
        self.cancelled = []
        self.closed = 0
        self.buys = 0
        self.sells = 0

    def cancel(self, order) -> None:
        self.cancelled.append(order)

    def close(self) -> None:
        self.closed += 1

    def buy_bracket(self, **_kwargs) -> None:
        self.buys += 1

    def sell_bracket(self, **_kwargs) -> None:
        self.sells += 1


def test_explicit_close_cancels_protection_then_closes_position() -> None:
    strategy = _Strategy()

    Plugin().apply_action(strategy, 3, {})

    assert strategy.cancelled == strategy._owned_orders
    assert strategy.closed == 1


def test_opposite_signal_closes_without_same_bar_reversal() -> None:
    long_strategy = _Strategy()
    Plugin().apply_action(long_strategy, 2, {})
    assert long_strategy.closed == 1
    assert long_strategy.sells == 0

    short_strategy = _Strategy()
    short_strategy.position.size = -1.0
    Plugin().apply_action(short_strategy, 1, {})
    assert short_strategy.closed == 1
    assert short_strategy.buys == 0
