# Mandatory order budget

The selected-input RL comparison uses `balance_budget`, not the historical
`portfolio_fraction` control. Each entry must have a finite positive SL and TP
distance before any parent order is submitted. Existing bracket, stop-first
collision and session-close mechanics remain unchanged.

The allocation ceiling in account currency is
`min(abs(action), 1) * rel_volume * balance_at_decision`.
Here balance is broker cash while flat, not floating account equity. The
allocation is a margin budget, not a promised maximum loss at SL. Notional is
budget times `min(leverage, leverage_cap)`; base units are notional divided by
the decision price. The declared entry-cost headroom reduces the allocation;
units round down to `volume_step`, then obey `max_order_volume`. Below
`min_order_volume` the entry abstains: the minimum never increases allocation.
Entry sizing is fixed at submission, not silently rebalanced every bar.

The initial comparator keeps leverage 1:1, rel_volume 0.1 and an illustrative
one-base-unit minimum/step. These are simulation settings, not verified Alpaca,
Oanda or MT5 contract sizes. A broker adapter must provide account-currency
conversion, contract multiplier, minimum and step for other instruments.
The scalar action chooses allocation and direction; it is not a calibrated
probability of market risk. ATR distances supply mandatory SL/TP independently.

The observations remain 13 samples, stride 2 observed H1 bars, including the
current completed bar. Closed-market hours are not fabricated as bars.

## Traceability and checks

| Requirement | Evidence |
| --- | --- |
| Cash-based allocation, not equity fraction | cash/equity-different unit test |
| Relative fraction and leverage conversion | 100%, 20%, leverage 1/10/50/100 |
| Minimum cannot override budget | under-minimum abstention |
| Finite geometry before submission | malformed SL/TP guard tests |
| Preserve prior controls | existing real Backtrader envelope tests |

Design and tests precede implementation. No claims of profitability, guaranteed
stop fill or broker certification follow from these engineering checks.
