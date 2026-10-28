"""Screen B / doc 40 B3: fractional sizing in the default action path.

Default OFF preserves the historical fixed-size behavior; ON scales the
order size by min(1, |raw_action|) and rebalances same-direction
fraction changes. Stub-based: _apply_action only touches p/position/
bridge/buy/sell/close/broker."""
from types import SimpleNamespace

from app.bt_bridge import BTBridgeStrategy


class _Recorder:
    def __init__(self, position_size=1.0, fractional=False,
                 current=0.0, raw=0.0):
        self.p = SimpleNamespace(position_size=position_size,
                                 fractional_position_sizing=fractional,
                                 min_equity=0.0)
        self.position = SimpleNamespace(size=current)
        self.bridge = SimpleNamespace(raw_action_slot=raw,
                                      execution_diagnostics={},
                                      force_flat_request=False)
        self.broker = SimpleNamespace(get_orders_open=lambda: [])
        self._plugin_apply = None
        self._plugin_config = {}
        self._require_protected_entries = False
        self._order_cost_accum = 0.0
        self.orders = []

    def buy(self, size):
        self.orders.append(("buy", round(float(size), 9)))

    def sell(self, size):
        self.orders.append(("sell", round(float(size), 9)))

    def close(self):
        self.orders.append(("close", abs(self.position.size)))

    def cancel(self, order):
        pass

    def apply(self, action):
        BTBridgeStrategy._apply_action(self, action)
        return self.orders


def test_flag_off_entry_uses_full_size_regardless_of_raw():
    r = _Recorder(fractional=False, raw=0.37)
    assert r.apply(1) == [("buy", 1.0)]


def test_flag_off_same_direction_no_rebalance():
    r = _Recorder(fractional=False, current=1.0, raw=0.2)
    assert r.apply(1) == []


def test_fractional_entry_scales_by_raw_magnitude():
    r = _Recorder(fractional=True, raw=0.5)
    assert r.apply(1) == [("buy", 0.5)]
    r = _Recorder(fractional=True, raw=-0.25)
    assert r.apply(2) == [("sell", 0.25)]


def test_fractional_cap_at_one():
    r = _Recorder(fractional=True, raw=3.7)
    assert r.apply(1) == [("buy", 1.0)]


def test_fractional_rebalance_up_and_down_same_direction():
    r = _Recorder(fractional=True, current=0.5, raw=0.8)
    assert r.apply(1) == [("buy", 0.3)]
    r = _Recorder(fractional=True, current=0.8, raw=0.5)
    assert r.apply(1) == [("sell", 0.3)]
    assert r.bridge.execution_diagnostics["fractional_rebalance_orders"] == 1


def test_fractional_no_order_when_fraction_unchanged():
    r = _Recorder(fractional=True, current=0.5, raw=0.5)
    assert r.apply(1) == []


def test_fractional_zero_fraction_closes_to_flat():
    r = _Recorder(fractional=True, current=0.7, raw=0.0)
    assert r.apply(1) == [("close", 0.7)]
    assert r.bridge.execution_diagnostics["fractional_flat_orders"] == 1


def test_fractional_direction_flip_closes_then_opens_scaled():
    r = _Recorder(fractional=True, current=0.6, raw=-0.4)
    assert r.apply(2) == [("close", 0.6), ("sell", 0.4)]


def test_hold_never_orders():
    for fractional in (False, True):
        r = _Recorder(fractional=fractional, current=0.5, raw=0.9)
        assert r.apply(0) == []


def test_fractional_short_rebalance_increase_uses_sell():
    # BUG REPRODUCTION (first Screen B run): short rebalances were
    # inverted, accumulating runaway positions in downtrends.
    r = _Recorder(fractional=True, current=-0.5, raw=-0.8)
    assert r.apply(2) == [("sell", 0.3)]


def test_fractional_short_rebalance_decrease_uses_buy():
    r = _Recorder(fractional=True, current=-0.8, raw=-0.5)
    assert r.apply(2) == [("buy", 0.3)]


def test_fractional_short_unchanged_no_order():
    r = _Recorder(fractional=True, current=-0.5, raw=-0.5)
    assert r.apply(2) == []
