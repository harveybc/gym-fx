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

Install openAI Gym from Github according to [these instructions](https://github.com/harveybc/gym-forex)these instructions.  
Install Python3, pip3 and other dependencies:  
sudo apt-get install -y python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools libboost-all-dev libsdl2-dev  
pip3 install graphviz matplotlib neat-python python-tk

### Step 2 - Setup gym-forex from GitHub

git clone https://github.com/harveybc/gym-forex  
cd gym-forex  
  
### Step 3 - Configure the NEAT parameters

nano agents/config  

Configure the population size and other parameters according to your computing 
capacity or requirements, start with the defaults.

### Step 4 - Configure a startup/restart script

nano res  

For pulling the latest changes and executing the optimizer with a connection to 
your singularity node Address and Port. For example:  

\#!/bin/bash  
git pull  
python agents/agent_NEAT_reps_v3.py ../datasets/ts_1y.csv http://192.168.0.241:3338 config  

After editing, change the permission of the file to be executable:  

chmod 777 res  

### Step 5 - Start your optimizer that uses the gym-forex environment and an agent.

./res  

### Step 6 - Verify its Working

Access the web interface from a browser in the address and port you configured.  
Access the Processes menu to monitor your optimization process progress.  
Also access the Optimization/Parameters menu to see details of individual optima found.
