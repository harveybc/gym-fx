"""Cancellation must include orders submitted since the last broker pass."""
import backtrader as bt
import pandas as pd
import pytest
from test_shared_execution_envelope import _env

from broker_plugins.default_broker import Plugin


@pytest.mark.parametrize('bracket', [False, True])
def test_cancel_before_acceptance_cannot_leave_executable_siblings(bracket):
    class CancelImmediately(bt.Strategy):
        def __init__(self):
            self.orders = []
            self.fills = []

        def next(self):
            if len(self) != 1:
                return
            if bracket:
                parent = self.buy(size=1, transmit=False)
                stop = self.sell(size=1, exectype=bt.Order.Stop,
                                 price=99, parent=parent, transmit=False)
                limit = self.sell(size=1, exectype=bt.Order.Limit,
                                  price=101, parent=parent, transmit=True)
                self.orders = [parent, stop, limit]
                self.cancel(parent)
            else:
                stop = self.sell(size=1, exectype=bt.Order.Stop, price=99)
                limit = self.sell(size=1, exectype=bt.Order.Limit,
                                  price=101, oco=stop)
                self.orders = [stop, limit]
                self.cancel(stop)

        def notify_order(self, order):
            if order.status == order.Completed:
                self.fills.append(order.ref)

    frame = pd.DataFrame(dict(open=[100]*4, high=[102]*4,
                              low=[98]*4, close=[100]*4),
                         index=pd.date_range('2024-01-01', periods=4, freq='h'))
    engine = bt.Cerebro()
    broker = Plugin().build_bt_broker({})
    engine.setbroker(broker)
    engine.adddata(bt.feeds.PandasData(dataname=frame))
    engine.addstrategy(CancelImmediately)
    strategy = engine.run()[0]
    assert strategy.fills == []
    assert strategy.position.size == 0
    assert all(order.status == order.Canceled for order in strategy.orders)
    assert not broker.submitted
    assert not broker.get_orders_open()


@pytest.mark.parametrize('direction', [1., -1.])
def test_policy_close_on_entry_bar_leaves_no_orphan_protection(tmp_path, direction):
    env = _env(tmp_path, [100.] * 30,
               execution_transition='next_open_v2',
               continuous_action_contract='target_exposure_hysteresis_v2',
               max_entries_per_day=1)
    try:
        env.reset(seed=7)
        env.step([direction * .5])
        assert env.bridge.position_units * direction > 0
        env.step([0.])
        assert env.bridge.position_units == 0
        assert env.bridge.open_order_count == 0
        for _ in range(10):
            env.step([0.])
            assert env.bridge.position_units == 0
            assert env.bridge.open_order_count == 0
        assert env.bridge.trade_count == 1
        assert not getattr(env.bridge, 'envelope_run_failure', None)
    finally:
        env.close()
