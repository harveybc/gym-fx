# gym-forex

The [Forex environment](https://github.com/harveybc/gym-forex) is a forex
trading simulator featuring: configurable initial capital, multiple currency pair
trading, dynamic or dataset-based spread, CSV history timeseries for trading
currencies and observations for the agent, fixed or agent-controlled take-profit, stop-loss and order volume.

The environment features discrete action spaces and optionally continuous
action spaces if the orders dont have fixed take-profit/stop-loss and order
volume.

## Observation Space

A concatenation of num_ticks vectors for the lastest: 
vector of values from timeseries, equity and its variation, 
order_status( -1=closed,1=opened),time_opened (normalized with
max_order_time), order_profit and its variation, order_drawdown
/order_volume_pips,  consecutive_drawdown/max_consecutive_dd

## Action Space

discrete action 0: 0=nop,1=close,2=buy,3=sell
discrete action 0 parameter: symbol
(optional) continuous action 0 parameter: percent_tp, percent_sl,percent_max

## Reward Function

TODO: Describe the Reward Function

# Installation
Install openAI gym, tensorflow and keras.

Copy as a new directory inside the gym/env and update the __init.py__ on gym/env to include the new wnvironment.

Work In Progress. Proper documentation commming soon.