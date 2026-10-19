"""A submitted bracket canceled through OCO must not retain its children."""

import backtrader as bt
import pandas as pd

from broker_plugins.default_broker import Plugin
import pytest


@pytest.mark.parametrize('cancel_parent', [False, True])
def test_cancel_submitted_oco_leader_retires_sibling_bracket_children(cancel_parent):
    class CancelOcoLeader(bt.Strategy):
        def __init__(self):
            self.created = []

        def next(self):
            if len(self) != 1:
                return
            leader = self.buy(size=1, exectype=bt.Order.Limit, price=90)
            parent = self.buy(size=1, oco=leader, transmit=False)
            stop = self.sell(size=1, exectype=bt.Order.Stop, price=99,
                             parent=parent, transmit=False)
            limit = self.sell(size=1, exectype=bt.Order.Limit, price=101,
                              parent=parent, oco=stop, transmit=True)
            self.created = [leader, parent, stop, limit]
            assert self.broker.cancel(parent if cancel_parent else leader) is True

    frame = pd.DataFrame(
        dict(open=[100] * 5, high=[102] * 5,
             low=[98] * 5, close=[100] * 5),
        index=pd.date_range("2024-01-01", periods=5, freq="h"),
    )
    engine = bt.Cerebro()
    broker = Plugin().build_bt_broker({})
    engine.setbroker(broker)
    engine.adddata(bt.feeds.PandasData(dataname=frame))
    engine.addstrategy(CancelOcoLeader)
    strategy = engine.run()[0]

    assert strategy.position.size == 0
    assert not broker.submitted
    assert all(order.status == order.Canceled for order in strategy.created), [
        order.getstatusname() for order in strategy.created
    ]
    assert not broker.get_orders_open()
