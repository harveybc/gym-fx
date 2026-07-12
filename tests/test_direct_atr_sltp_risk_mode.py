from __future__ import annotations

import pytest

from strategy_plugins.direct_atr_sltp import Plugin


def test_rel_volume_aware_sltp_preserves_historical_baseline():
    plugin = Plugin()
    params = plugin.plugin_params.copy()
    params.update(
        {
            "sltp_risk_mode": "rel_volume_aware_atr",
            "rel_volume": 0.05,
            "baseline_rel_volume": 0.05,
            "max_risk_rel_volume": 0.50,
            "k_sl": 2.0,
            "k_tp": 3.0,
        }
    )

    k_sl_eff, k_tp_eff = plugin._effective_sltp_multiples(params)

    assert k_sl_eff == pytest.approx(2.0)
    assert k_tp_eff == pytest.approx(3.0)


def test_rel_volume_aware_sltp_shrinks_high_exposure_and_keeps_tp_at_least_sl():
    plugin = Plugin()
    params = plugin.plugin_params.copy()
    params.update(
        {
            "sltp_risk_mode": "rel_volume_aware_atr",
            "rel_volume": 0.50,
            "baseline_rel_volume": 0.05,
            "max_risk_rel_volume": 0.50,
            "rel_volume_sl_shrink_alpha": 0.35,
            "rel_volume_tp_shrink_alpha": 0.20,
            "min_reward_risk_ratio": 1.0,
            "k_sl": 2.0,
            "k_tp": 3.0,
        }
    )

    k_sl_eff, k_tp_eff = plugin._effective_sltp_multiples(params)

    assert k_sl_eff == pytest.approx(1.30)
    assert k_tp_eff == pytest.approx(2.40)
    assert k_tp_eff >= k_sl_eff
