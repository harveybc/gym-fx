import gym
import numpy as np
import pandas as pd
from collections import deque
import zlib, pickle
import math

class Plugin:
    """
    An environment plugin for forex trading automation tasks, compatible with both NEAT and OpenRL.
    """
    plugin_params = {
        'initial_balance': 10000,
        'fitness_function': 'brute_profit',  # 'sharpe_ratio' can be another option
        'min_orders': 4,
        'tp': 10000,  # Default TP (in pips) – used if not overridden at order open
        'sl': 10000,  # Default SL (in pips)
        'rel_volume': 0.05,  # Size of the new orders relative to the current balance
        'max_order_volume': 1000000,  # Maximum order volume
        'min_order_volume': 10000,    # Minimum order volume
        'leverage': 1000,
        'pip_cost': 0.00001,  # 1 pip cost in EURUSD/pip
        'min_order_time': 6,   # Minimum Order Time to allow manual closing
        'max_order_time': 96,  # Maximum Order Time to allow manual closing
        'spread': 0.0003       # Default spread value
    }

    plugin_debug_vars = ['initial_balance', 'max_steps', 'fitness_function', 'final_balance', 'final_fitness']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.env = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def build_environment(self, x_train, y_train, config):
        self.initial_balance = config.get('initial_balance', self.params['initial_balance'])
        self.max_steps = config.get('max_steps', None)
        self.fitness_function = config.get('fitness_function', self.params['fitness_function'])
        self.min_orders = config.get('min_orders', self.params['min_orders'])
        self.sl = config.get('sl', self.params['sl'])
        self.tp = config.get('tp', self.params['tp'])
        # Note: The optimizer will set the ideal SL/TP (in pips) for each order.
        sl_buy = self.sl
        sl_sell = self.sl
        self.rel_volume = config.get('rel_volume', self.params['rel_volume'])
        self.leverage = config.get('leverage', self.params['leverage'])
        self.pip_cost = config.get('pip_cost', self.params['pip_cost'])
        self.min_order_time = config.get('min_order_time', self.params['min_order_time'])
        self.max_order_time = config.get('max_order_time', self.params['max_order_time'])
        self.spread = config.get('spread', self.params['spread'])
        self.max_order_volume = config.get('max_order_volume', self.params['max_order_volume'])
        self.min_order_volume = config.get('min_order_volume', self.params['min_order_volume'])
        self.genome = config.get('genome', None)
        self.env = AutomationEnv(x_train, y_train, self.initial_balance, self.max_steps, self.fitness_function,
                                 self.min_orders, self.sl, self.tp, self.rel_volume, self.leverage,
                                 self.pip_cost, self.min_order_time, self.max_order_time, self.spread,
                                 self.max_order_volume, self.min_order_volume, self.genome)
        return self.env

    def reset(self, genome=None):
        self.returns = []  # Initialize returns to track rewards
        observation, info, max_steps = self.env.reset(genome)
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def calculate_fitness(self, rewards, equity_curve=None):
        if self.fitness_function == 'sharpe_ratio':
            return self._calculate_sharpe_ratio(equity_curve)
        else:  # Default to brute_profit
            return rewards.sum() / len(rewards)

    def _calculate_sharpe_ratio(self, equity_curve):
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        return return_ratio * np.sqrt(252)

class AutomationEnv(gym.Env):
    def __init__(self, x_train, y_train, initial_balance, max_steps, fitness_function,
                 min_orders, sl, tp, rel_volume, leverage, pip_cost,
                 min_order_time, max_order_time, spread, max_order_volume, min_order_volume, genome):
        super(AutomationEnv, self).__init__()
        self.max_steps = max_steps
        self.x_train = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else x_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.balance_ant = self.balance
        self.equity_ant = self.balance
        self.current_step = 0
        self.order_status = 0  # 0 = no order, 1 = buy open, 2 = sell open
        self.order_price = 0.0
        self.order_close = 0.0
        self.ticks_per_hour = 1
        self.order_date = 0
        self.profit_pips = 0.0
        self.real_profit = 0.0
        self.orders_list = []
        self.order_volume = 0.0
        self.fitness = 0.0
        self.done = False
        self.reward = 0.0
        self.equity_curve = [initial_balance]
        self.min_orders = min_orders
        self.sl = sl
        self.tp = tp
        self.rel_volume = rel_volume
        self.leverage = leverage
        self.pip_cost = pip_cost
        self.min_order_time = min_order_time
        self.max_order_time = max_order_time
        self.spread = spread
        self.margin = 0.0
        self.order_time = 0
        self.num_ticks = self.x_train.shape[0]
        self.num_closes = 0  # Track number of closes
        self.c_c = 0       # Closing cause
        self.ant_c_c = 0   # Previous closing cause
        self.max_order_volume = max_order_volume
        self.min_order_volume = min_order_volume
        if self.y_train is None:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.x_train.shape[1],), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.y_train.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(3),  # Buy, sell, hold
            gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # Continuous volume action
        ))
        self.genome = genome
        self.reset(genome)
    
    def reset(self, genome=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.order_status = 0
        self.order_price = 0.0
        self.order_volume = 0.0
        self.reward = 0.0
        self.done = False
        self.num_closes = 0
        self.c_c = 0
        self.ant_c_c = 0
        self.margin = 0.0
        self.order_time = 0
        self.max_dd_pips = 0
        self.genome = genome
        self.kolmogorov_c = 0
        if self.genome is not None:
            try:
                self.kolmogorov_c = self.kolmogorov_complexity(self.genome)
            except:
                self.kolmogorov_c = 0
        self.returns = []
        self.orders_list = []
        self.equity_curve = [self.initial_balance]
        observation = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]
        self.fitness = 0.0
        info = {
            "date": self.x_train[self.current_step, 0],
            "close": self.x_train[self.current_step, 4],
            "high": self.x_train[self.current_step, 3],
            "low": self.x_train[self.current_step, 2],
            "open": self.x_train[self.current_step, 1],
            "action": 0,
            "observation": observation,
            "episode_over": self.done,
            "tick_count": 0,
            "num_closes": 0,
            "balance": self.balance,
            "equity": self.balance,
            "reward": 0.0,
            "order_status": 0,
            "order_volume": 0,
            "spread": self.spread,
            "initial_balance": self.initial_balance
        }
        max_steps = self.max_steps
        if max_steps > self.x_train.shape[0]:
            max_steps = self.x_train.shape[0]
            self.max_steps = max_steps - 1
        return observation, info, max_steps
    
    def step(self, action, verbose=True, step_fitness=0.0, genome_id=0, num_closes=0, reward_auc_prev=0.0, act_values=[0.0, 0.0, 0.0]):
        if self.done:
            return np.zeros(self.x_train.shape[1]), self.reward, self.done, {}
        if self.current_step >= self.max_steps:
            self.done = True
        else:
            self.equity_ant = self.equity
            current_date = self.x_train[self.current_step, 0]
            High = self.x_train[self.current_step, 3]
            Low = self.x_train[self.current_step, 2]
            Close = self.x_train[self.current_step, 4]
            discrete_action = action[0]
            volume_action = action[1][0]
            volume_action = (volume_action + 1) / 2
            self.profit_pips = 0
            self.real_profit = 0
            if self.order_status == 1:
                self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                if self.profit_pips < 0:
                    if self.max_dd_pips < -self.profit_pips:
                        self.max_dd_pips = -self.profit_pips
            if self.order_status == 2:
                self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                if self.profit_pips < 0:
                    if self.max_dd_pips < -self.profit_pips:
                        self.max_dd_pips = -self.profit_pips
            self.equity = self.balance + self.real_profit
            self.c_c = 0
        if not self.done:
            # --- Order Entry Logic (unchanged) ---
            if (self.order_status == 0) and discrete_action == 1:
                self.order_status = 1
                self.order_price = High + self.spread
                self.margin += (self.order_volume / self.leverage)
                self.order_time = self.current_step
                self.order_date = current_date
                self.max_dd_pips = 0
                max_volume = self.equity * self.rel_volume * self.leverage
                if max_volume < self.min_order_volume:
                    max_volume = self.min_order_volume
                volume_range = self.max_order_volume - self.min_order_volume
                self.order_volume = self.min_order_volume + volume_action * volume_range
                if self.order_volume > max_volume:
                    self.order_volume = max_volume
                if self.order_volume < self.min_order_volume:
                    self.order_volume = self.min_order_volume
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Opening order - Action: Buy, Price: {self.order_price}, volume_action: {volume_action}, Volume: {self.order_volume}")
                    print(f"Current balance (after BUY action): {self.balance}, Number of closes: {self.num_closes}")
                    print(f"Order Status after buy action: {self.order_status}")
            if (self.order_status == 0) and discrete_action == 2:
                self.order_status = 2
                self.order_price = Low
                self.margin += (self.order_volume / self.leverage)
                self.order_time = self.current_step
                self.order_date = current_date
                self.max_dd_pips = 0
                max_volume = self.equity * self.rel_volume * self.leverage
                if max_volume < self.min_order_volume:
                    max_volume = self.min_order_volume
                volume_range = self.max_order_volume - self.min_order_volume
                self.order_volume = self.min_order_volume + volume_action * volume_range
                if self.order_volume > max_volume:
                    self.order_volume = max_volume
                if self.order_volume < self.min_order_volume:
                    self.order_volume = self.min_order_volume
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Opening order - Action: Sell, Price: {self.order_price}, volume_action: {volume_action}, Volume: {self.order_volume}")
                    print(f"Current balance (after SELL action): {self.balance}, Number of closes: {self.num_closes}")
                    print(f"Order Status after sell action: {self.order_status}")
            # --- Manual close logic (unchanged) ---
            if ((self.order_status == 1 and discrete_action == 2) or 
                (self.order_status == 2 and discrete_action == 1)) or \
               (self.order_status == 1 and discrete_action == 0) or \
               (self.order_status == 2 and discrete_action == 0):
                if (self.current_step - self.order_time) > self.min_order_time:
                    if self.order_status == 1:
                        self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                        self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                        self.order_close = Low
                    if self.order_status == 2:
                        self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                        self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                        self.order_close = High + self.spread
                    self.order_status = 0
                    self.equity = self.balance + self.real_profit
                    self.balance = self.equity
                    self.margin = 0.0
                    self.c_c = 4  # Normal close
                    self.num_closes += 1
                    order = {
                        'volume':  self.order_volume,
                        'equity':  self.equity,
                        'close_date': current_date,
                        'open_date': self.order_date,
                        'ticks': self.current_step - self.order_time,
                        'order_type': self.order_status,
                        'order_price': self.order_price,
                        'order_close': self.order_close,
                        'profit_pips': self.profit_pips,
                        'real_profit': self.real_profit,
                        'max_dd_pips': self.max_dd_pips,
                        'closing_cause': self.c_c
                    }
                    self.order_volume = 0.0
                    self.orders_list.append(order)
                    if verbose:
                        print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Normal Close")
                        print(f"Current balance: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                        print(f"Order Status after normal close: {self.order_status}")
            # --- Fixed SL and TP Checks (in pips) ---
            if self.order_status == 1:
                current_tp = self.tp_buy  # Fixed TP for BUY (in pips, as set when order opened)
                current_sl = self.sl_buy  # Fixed SL for BUY (in pips)
            elif self.order_status == 2:
                current_tp = self.tp_sell  # Fixed TP for SELL (in pips)
                current_sl = self.sl_sell  # Fixed SL for SELL (in pips)
            else:
                current_tp, current_sl = self.tp, self.sl
            if self.order_status == 1 and self.profit_pips <= (-1 * current_sl):
                self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                self.order_close = Low
                self.order_status = 0
                self.equity = self.balance + self.real_profit
                self.balance = self.equity
                self.margin = 0.0
                self.c_c = 2  # Stop Loss triggered.
                self.num_closes += 1
                order = {
                    'volume':  self.order_volume,
                    'equity':  self.equity,
                    'close_date': current_date,
                    'open_date': self.order_date,
                    'ticks': self.current_step - self.order_time,
                    'order_type': self.order_status,
                    'order_price': self.order_price,
                    'order_close': self.order_close,
                    'profit_pips': self.profit_pips,
                    'real_profit': self.real_profit,
                    'max_dd_pips': self.max_dd_pips,
                    'closing_cause': self.c_c
                }
                self.order_volume = 0.0
                self.orders_list.append(order)
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Stop Loss")
                    print(f"Current balance: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                    print(f"Order Status after stop loss: {self.order_status}")
            if self.order_status == 2 and self.profit_pips >= current_tp:
                self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                self.order_close = High + self.spread
                self.order_status = 0
                self.equity = self.balance + self.real_profit
                self.balance = self.equity
                self.margin = 0.0
                self.c_c = 3  # Take Profit triggered.
                self.num_closes += 1
                order = {
                    'volume':  self.order_volume,
                    'equity':  self.equity,
                    'close_date': current_date,
                    'open_date': self.order_date,
                    'ticks': self.current_step - self.order_time,
                    'order_type': self.order_status,
                    'order_price': self.order_price,
                    'order_close': self.order_close,
                    'profit_pips': self.profit_pips,
                    'real_profit': self.real_profit,
                    'max_dd_pips': self.max_dd_pips,
                    'closing_cause': self.c_c
                }
                self.order_volume = 0.0
                self.orders_list.append(order)
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Take Profit")
                    print(f"Current balance: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                    print(f"Order Status after take profit: {self.order_status}")
            if (self.order_status == 1 or self.order_status == 2):
                if (self.current_step - self.order_time) > self.max_order_time:
                    if self.order_status == 1:
                        self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                        self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                        self.order_close = Low
                    if self.order_status == 2:
                        self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                        self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                        self.order_close = High + self.spread
                    self.order_status = 0
                    self.equity = self.balance + self.real_profit
                    self.balance = self.equity
                    self.margin = 0.0
                    self.c_c = 5  # Order timeout.
                    self.num_closes += 1
                    order = {
                        'volume':  self.order_volume,
                        'equity':  self.equity,
                        'close_date': current_date,
                        'open_date': self.order_date,
                        'ticks': self.current_step - self.order_time,
                        'order_type': self.order_status,
                        'order_price': self.order_price,
                        'order_close': self.order_close,
                        'profit_pips': self.profit_pips,
                        'real_profit': self.real_profit,
                        'max_dd_pips': self.max_dd_pips,
                        'closing_cause': self.c_c
                    }
                    self.order_volume = 0.0
                    self.orders_list.append(order)
                    if verbose:
                        print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Order Timeout")
                        print(f"Current balance: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                        print(f"Order Status after timeout: {self.order_status}")
        margin_call_lambda = 100
        reward_margin_call = 0.0
        reward = 0.0
        if self.current_step > 0:
            penalty_cost = -1 / self.max_steps
            if self.done and self.c_c == 1:
                reward_margin_call = (self.max_steps - self.current_step) * penalty_cost
        if not self.done:
            ob = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]
        reward += reward_margin_call * margin_call_lambda
        self.balance_ant = self.balance
        self.fitness = self.fitness + reward
        self.current_step += 1                
        if self.current_step >= self.max_steps:
            self.done = True
        if self.done:
            returns = [order['profit_pips'] for order in self.orders_list]
            durations_hours = [order['ticks'] for order in self.orders_list]
            sharpe_ratio = self.calculate_sharpe_ratio(returns, durations_hours)
        info = {
            "date": self.x_train[self.current_step - 1, 0],
            "close": self.x_train[self.current_step - 1, 4],
            "balance": self.balance,
            "equity": self.equity,
            "reward": reward,
            "c_c": self.c_c,
            "sharpe_ratio": sharpe_ratio if self.done else 0,
            "orders": self.orders_list,
            "initial_balance": self.initial_balance,
            "order_status": self.order_status,
            "profit_pips": self.profit_pips
        }
        return ob, reward, self.done, info

    def render(self, mode='human'):
        pass
    
    def calculate_sharpe_ratio(self, returns, durations_hours, annual_risk_free_rate=0.1):
        if len(returns) <= 1:
            return 0
        total_duration_hours = sum(durations_hours)
        avg_duration_hours = total_duration_hours / len(durations_hours)
        adjusted_risk_free_rate = 0
        mean_return = np.mean(returns)
        return_std = np.std(returns)
        if return_std == 0:
            return -100
        sharpe_ratio = (mean_return - adjusted_risk_free_rate) / (return_std)
        if sharpe_ratio > 1 and len(returns) < 30:
            sharpe_ratio = sharpe_ratio/10
        if sharpe_ratio > 1 and len(returns) < 20:
            sharpe_ratio = sharpe_ratio/10
        if sharpe_ratio > 1 and len(returns) < 10:
            sharpe_ratio = sharpe_ratio/10
        if sharpe_ratio > 1 and len(returns) < 5:
            sharpe_ratio = sharpe_ratio/10
        return sharpe_ratio

    def kolmogorov_complexity(self, genome):
        genome_bytes = pickle.dumps(genome)
        compressed_data = zlib.compress(genome_bytes)
        return len(compressed_data)

    def calculate_final_debug_vars(self):
        return {
            'final_balance': self.balance,
            'final_fitness': self.reward
        }
