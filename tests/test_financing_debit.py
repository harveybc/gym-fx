"""A declared analytical financing scenario must actually debit equity."""
from pathlib import Path
import sys

import pytest

from broker_plugins.default_broker import Plugin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_shared_execution_envelope import _env


@pytest.mark.parametrize('direction', [1., -1.])
def test_daily_financing_debits_both_sides_from_real_broker(tmp_path, direction):
    env = _env(tmp_path, [100.] * 40, execution_transition='next_open_v2',
               financing_enabled=True, financing_model='backtrader_annual_notional_daily_v1',
               financing_annual_rate=.365)
    try:
        env.reset(seed=7)
        env.step([direction * .5])
        units = abs(env.bridge.position_units)
        assert env.bridge.equity == pytest.approx(10000)
        for _ in range(12):
            env.step([0.])
        # Four-hour bars: Jan 1 04:00 entry -> Jan 3 04:00, two UTC date changes.
        assert env.bridge.equity == pytest.approx(10000 - units * 100 * .365 / 365 * 2)
    finally:
        env.close()


@pytest.mark.parametrize('patch', [{}, {'financing_model': 'unknown'},
    {'financing_model': 'backtrader_annual_notional_daily_v1'},
    {'financing_model': 'backtrader_annual_notional_daily_v1', 'financing_annual_rate': float('nan')},
    {'financing_model': 'backtrader_annual_notional_daily_v1', 'financing_annual_rate': -.1}])
def test_enabled_financing_requires_explicit_supported_contract(patch):
    with pytest.raises(ValueError, match='FINANCING'):
        Plugin().build_bt_broker({'financing_enabled': True, **patch})
