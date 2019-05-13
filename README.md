# gym-forex

The [Forex environment](https://github.com/harveybc/gym-forex) is a forex
trading simulator featuring: configurable initial capital, dynamic or dataset-based spread, CSV history timeseries for trading
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

## MQL4 Dataset Generator

The datasets used for the tests were generated with a MQL4 program located in the
agents folder (Pending documentation).

# Installation
### Step 1 - Setup Dependencies

Install Python, pip,  OpenAI Gym and other dependencies:  

sudo apt-get install -y python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg libboost-all-dev libsdl2-dev python3-pip git gcc make perl  

pip3 install graphviz neat-python gitpython gym neat-python matplotlib

### Step 2 - Setup gym-forex from GitHub

git clone https://github.com/harveybc/gym-forex  

### Step 3 - Configure the NEAT parameters

Set the PYTHONPATH venvironment variable, you may add the following line to the .profile file in your home directory to export on start of sessions. Replace <username> with your username.

export PYTHONPATH=/home/username/gym-forex/:${PYTHONPATH}  
  
### Step 4 - Configure the NEAT parameters

cd gym-forex  
nano agents/config   

Configure the population size and other parameters according to your computing 
capacity or requirements, start with the defaults.  

### Step 5 - Configure a startup/restart script

nano res  

For example:  

#!/bin/bash  
git pull  
python3 agents/agent_NEAT.py ./datasets/ts_5min_1w.CSV ./datasets/vs_5min_1w.CSV config_20  

After editing, change the permission of the file to be executable:  

chmod 777 res  

### Step 6 - Start your optimizer that uses the gym-forex environment and an agent.

./res   



