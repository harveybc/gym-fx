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

The reward function is the average of the area under the curve of equity and the 
balance variation.

# Installation
### Step 1 - Setup Dependencies

Install Python, pip and other dependencies and OpenAI Gym:  

sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools libboost-all-dev libsdl2-dev python-pip 

pip install graphviz matplotlib neat-python python-tk git gym neat-python matplotlib

### Step 2 - Setup gym-forex from GitHub

cd 
git clone https://github.com/harveybc/gym-forex  
cd gym-forex  

### Step 3 - Configure the NEAT parameters

set the PYTHONPATH venvironment variable, you may add the following line to the .profile file. Replace <username> with your username.

export PYTHONPATH=/home/<username>/gym-forex/:${PYTHONPATH}
  
### Step 4 - Configure the NEAT parameters

nano agents/config  

Configure the population size and other parameters according to your computing 
capacity or requirements, start with the defaults.

### Step 5 - Configure a startup/restart script

nano res  

For pulling the latest changes and executing the optimizer with a connection to 
your singularity node Address and Port. For example:  

#!/bin/bash
git stash
git pull
python3 agents/agent_NEAT.py ./datasets/ts_5min_1w.CSV ./datasets/vs_5min_1w.CSV config_20

After editing, change the permission of the file to be executable:  

chmod 777 res  

### Step 6 - Start your optimizer that uses the gym-forex environment and an agent.

./res  


