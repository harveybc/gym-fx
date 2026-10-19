"""Mandatory protection and balance-budget sizing: no learning or real broker."""
from types import SimpleNamespace

import pytest

from strategy_plugins.shared_execution_envelope import Plugin


def config(**overrides):
    return dict(sizing_mode="balance_budget", rel_volume=0.1,
                leverage=10, leverage_cap=10, entry_cost_headroom=0,
                min_order_volume=1, max_order_volume=1e12, volume_step=1,
                **overrides)


def strategy(balance=10000, equity=99999):
    return SimpleNamespace(broker=SimpleNamespace(
        getcash=lambda: balance, getvalue=lambda: equity))


@pytest.mark.parametrize("fraction,expected", [(1, 8000), (.2, 1600), (0, 0), (5, 8000)])
def test_cash_budget_times_leverage_converts_to_base_units(fraction, expected):
    units = Plugin()._entry_units(strategy(), config(), fraction, 1.25)
    assert units == expected
    assert units * 1.25 / 10 <= min(fraction, 1) * .1 * 10000


@pytest.mark.parametrize("leverage", [1, 10, 50, 100])
def test_configurable_leverage_does_not_change_margin_budget(leverage):
    p = config()
    p.update(leverage=leverage, leverage_cap=leverage)
    units = Plugin()._entry_units(strategy(), p, .2, 1.25)
    assert units * 1.25 / leverage == 200


def test_below_minimum_abstains_instead_of_upscaling():
    p = config()
    p.update(min_order_volume=10000)
    assert Plugin()._entry_units(strategy(), p, 1, 1.25) == 0


def test_step_rounds_down_and_maximum_caps_without_exceeding_budget():
    p = config()
    p.update(volume_step=3, max_order_volume=100)
    assert Plugin()._entry_units(strategy(), p, 1, 1.25) == 99
    p.update(volume_step=.1, max_order_volume=1e12)
    assert Plugin()._entry_units(strategy(), p, .2, 1.21) <= 2000 / 1.21


@pytest.mark.parametrize("key,value", [
    ("rel_volume", 0), ("rel_volume", 1.1), ("rel_volume", float("nan")),
    ("volume_step", 0), ("min_order_volume", -1),
    ("max_order_volume", 0), ("leverage", 0),
    ("leverage_cap", 0), ("entry_cost_headroom", 1),
    ("entry_cost_headroom", -1),
])
def test_invalid_budget_configuration_fails_closed(key, value):
    p = config()
    p[key] = value
    with pytest.raises(ValueError, match="ENVELOPE_"):
        Plugin()._entry_units(strategy(), p, 1, 1.25)


@pytest.mark.parametrize("cash,price,fraction", [
    (float("nan"), 1.25, 1), (10000, float("nan"), 1),
    (10000, 0, 1), (10000, 1.25, float("nan")),
])
def test_nonfinite_decision_inputs_rejected(cash, price, fraction):
    with pytest.raises(ValueError, match="ENVELOPE_"):
        Plugin()._entry_units(strategy(cash), config(), fraction, price)


def test_insolvent_balance_cannot_create_order():
    assert Plugin()._entry_units(strategy(-10), config(), 1, 1.25) == 0


@pytest.mark.parametrize("sl,tp", [(0, .1), (.1, 0), (-.1, .1),
                                  (float("nan"), .1), (.1, float("inf"))])
def test_invalid_protection_rejected_before_any_parent_order(sl, tp):
    plugin = Plugin()
    plugin._distances = lambda *_: (sl, tp)
    # No buy/sell/data methods: a call before the guard would fail differently.
    with pytest.raises(ValueError, match="ENVELOPE_INVALID_PROTECTION"):
        plugin._submit_bracket(SimpleNamespace(), {}, 1, 1, 1.25)


def test_unknown_sizing_mode_is_not_silently_legacy():
    p = config()
    p["sizing_mode"] = "typo"
    with pytest.raises(ValueError, match="ENVELOPE_UNKNOWN_SIZING_MODE"):
        Plugin()._entry_units(strategy(), p, 1, 1.25)
