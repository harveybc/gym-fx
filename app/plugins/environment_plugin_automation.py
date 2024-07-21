import gym
import numpy as np
import pandas as pd
from collections import deque

class Plugin:
    """
    An environment plugin for forex trading automation tasks, compatible with both NEAT and OpenRL.
    """

    plugin_params = {
        'initial_balance': 10000,
        'fitness_function': 'brute_profit',  # 'sharpe_ratio' can be another option
        'min_orders': 4,
        'sl': 100,  # Adjusted Stop Loss
        'tp': 100,  # Adjusted Take Profit
        'rel_volume': 0.1, # size of the new orders relative to the current balance
        'max_order_volume': 1000000, # Maximum order volume = 10 lots (1 lot = 100,000 units)
        'min_order_volume': 10000, # Minimum order volume = 0.1 lots (1 lot = 100,000 units)
        'leverage': 100,
        'pip_cost': 0.0001,
        'min_order_time': 5,  #  Minimum Order Time to allow manual closing by an action inverse to the current order.
        'spread': 0.002  # Default spread value
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
        self.max_steps = config.get('max_steps', self.params['max_steps'])
        self.fitness_function = config.get('fitness_function', self.params['fitness_function'])
        self.min_orders = config.get('min_orders', self.params['min_orders'])
        self.sl = config.get('sl', self.params['sl'])
        self.tp = config.get('tp', self.params['tp'])
        self.rel_volume = config.get('rel_volume', self.params['rel_volume'])
        self.leverage = config.get('leverage', self.params['leverage'])
        self.pip_cost = config.get('pip_cost', self.params['pip_cost'])
        self.min_order_time = config.get('min_order_time', self.params['min_order_time'])
        self.spread = config.get('spread', self.params['spread'])
        self.max_order_volume = config.get('max_order_volume', self.params['max_order_volume'])
        self.min_order_volume = config.get('min_order_volume', self.params['min_order_volume'])
        self.env = AutomationEnv(x_train, y_train, self.initial_balance, self.max_steps, self.fitness_function,
                                 self.min_orders, self.sl, self.tp, self.rel_volume, self.leverage, self.pip_cost, self.min_order_time, self.spread, self.max_order_volume, self.min_order_volume)

    def reset(self):
        observation, info = self.env.reset()
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
                 min_orders, sl, tp, rel_volume, leverage, pip_cost, min_order_time, spread, max_order_volume, min_order_volume):
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
        self.order_status = 0  # 0 = no order, 1 = buy, 2 = sell
        self.order_price = 0.0
        self.order_close = 0.0
        

        self.profit_pips = 0.0
        self.real_profit = 0.0
        
        self.order_volume = 0.0
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
        self.spread = spread
        self.margin = 0.0
        self.order_time = 0
        self.num_ticks = self.x_train.shape[0]
        self.num_closes = 0  # Track number of closes
        self.c_c = 0  # Track closing cause
        self.ant_c_c = 0  # Track previous closing cause
        self.max_order_volume = max_order_volume
        self.min_order_volume = min_order_volume
        if y_train is None:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.x_train.shape[1],), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.y_train.shape[1],), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(3)  # Buy, sell, hold
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.order_status = 0
        self.order_price = 0.0
        self.order_volume = 0.0
        self.reward = 0.0
        self.done = False
        self.num_closes = 0  # Track number of closes
        self.c_c = 0  # Track closing cause
        self.ant_c_c = 0  # Track previous closing cause
        self.margin = 0.0
        self.order_time = 0

        self.equity_curve = [self.initial_balance]
        observation = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]
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
        return observation, info

    def step(self, action, verbose=True):
        if self.done:
            return np.zeros(self.x_train.shape[1]), self.reward, self.done, {}

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Read time variables from CSV (Format: 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v)
        High = self.x_train[self.current_step, 3]
        Low = self.x_train[self.current_step, 2]
        Close = self.x_train[self.current_step, 4]

        # Calculate profit
        self.profit_pips = 0
        self.real_profit = 0
        # Calculate for existing BUY order (status=1)
        if self.order_status == 1:
            self.profit_pips = ((Low - self.order_price) / self.pip_cost)
            self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
        # Calculate for existing SELL order (status=2)
        if self.order_status == 2:
            self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
            self.real_profit = self.profit_pips * self.pip_cost * self.order_volume

        # Calculate equity
        self.equity = self.balance + self.real_profit

        

        # Verify if Margin Call
        if self.equity < self.margin:
            self.order_status = 0
            self.profit_pips = (self.equity - self.balance)/self.pip_cost
            self.real_profit = self.order_profit_pips * self.pip_cost * self.order_volume
            self.balance = 0.0
            self.equity = 0.0
            self.margin = 0.0
            self.order_close = Close
            self.c_c = 1  # Set closing cause to margin call
            self.done = True
            if verbose:
                print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Margin Call")
                print(f"Current balance 7: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                print(f"Order Status after margin call check: {self.order_status}")

        if not self.done:
            # Verify if close by SL
            if self.profit_pips <= (-1 * self.sl):
                # Calculate for existing BUY order (status=1)
                if self.order_status == 1:
                    self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                    self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                    self.order_close = Low
                # Calculate for existing SELL order (status=2)
                if self.order_status == 2:
                    self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                    self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                    self.order_close = High + self.spread
                self.order_status = 0
                # Calculate equity
                self.equity = self.balance + self.real_profit
                self.balance = self.equity
                self.margin = 0.0
                self.c_c = 2  # Set closing cause to stop loss
                self.order_volume = 0.0
                self.num_closes += 1
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Stop Loss")
                    print(f"Current balance 6: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                    print(f"Order Status after stop loss check: {self.order_status}")

            # Verify if close by TP
            if self.profit_pips >= self.tp:
                # Calculate for existing BUY order (status=1)
                if self.order_status == 1:
                    self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                    self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                    self.order_close = Low
                # Calculate for existing SELL order (status=2)
                if self.order_status == 2:
                    self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                    self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                    self.order_close = High + self.spread
                self.order_status = 0
                self.equity = self.balance + self.real_profit
                self.balance = self.equity
                self.margin = 0.0
                self.c_c = 3  # Set closing cause to take profit
                self.order_volume = 0.0
                self.num_closes += 1
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Take Profit")
                    print(f"Current balance 5: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                    print(f"Order Status after take profit check: {self.order_status}")

            # Executes BUY action, order status = 1
            if (self.order_status == 0) and action == 1:
                self.order_status = 1
                self.order_price = High + self.spread
                self.order_volume = self.equity * self.rel_volume * self.leverage
                if self.order_volume > self.max_order_volume:
                    self.order_volume = self.max_order_volume
                if self.order_volume < self.min_order_volume:
                    self.order_volume = self.min_order_volume    
                self.margin += (self.order_volume / self.leverage)
                self.order_time = self.current_step
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Opening order - Action: Buy, Price: {self.order_price}, Volume: {self.order_volume}")
                    print(f"Current balance (after BUY action): {self.balance}, Number of closes: {self.num_closes}")
                    print(f"Order Status after buy action: {self.order_status}")

            # Executes SELL action, order status = 2
            if (self.order_status == 0) and action == 2:
                self.order_status = 2
                self.order_price = Low
                self.order_volume = self.equity * self.rel_volume * self.leverage
                if self.order_volume > self.max_order_volume:
                    self.order_volume = self.max_order_volume
                if self.order_volume < self.min_order_volume:
                    self.order_volume = self.min_order_volume    
                self.margin += (self.order_volume / self.leverage)
                self.order_time = self.current_step
                if verbose:
                    print(f"{self.x_train[self.current_step, 0]} - Opening order - Action: Sell, Price: {self.order_price}, Volume: {self.order_volume}")
                    print(f"Current balance (after SELL action): {self.balance}, Number of closes: {self.num_closes}")
                    print(f"Order Status after sell action: {self.order_status}")

            # Manual close by action (Buy -> Sell or Sell -> Buy) if min_order_time has passed
            if (self.order_status == 1 and action == 2) or (self.order_status == 2 and action == 1):
                if (self.current_step - self.order_time) > self.min_order_time:
                    # Calculate for existing BUY order (status=1)
                    if self.order_status == 1:
                        self.profit_pips = ((Low - self.order_price) / self.pip_cost)
                        self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                        self.order_close = Low
                    # Calculate for existing SELL order (status=2)
                    if self.order_status == 2:
                        self.profit_pips = ((self.order_price - (High + self.spread)) / self.pip_cost)
                        self.real_profit = self.profit_pips * self.pip_cost * self.order_volume
                        self.order_close = High + self.spread
                    self.order_status = 0
                    self.equity = self.balance + self.real_profit
                    self.balance = self.equity
                    self.margin = 0.0
                    self.c_c = 0  # Set closing cause to normal close
                    self.order_volume = 0.0
                    self.num_closes += 1
                    if verbose:
                        print(f"{self.x_train[self.current_step, 0]} - Closed order at {self.order_close} - Cause: Normal Close")
                        print(f"Current balance 4: {self.balance}, Profit PIPS: {self.profit_pips}, Real Profit: {self.real_profit}, Number of closes: {self.num_closes}")
                        print(f"Order Status after normal close: {self.order_status}")

        # Simplified reward calculation
        if self.current_step > 1:
            equity_increment = self.equity - self.equity_ant
            balance_increment = self.balance - self.balance_ant 
            reward = (balance_increment + equity_increment) / 2
            reward = (reward / self.initial_balance) / self.max_steps # Normalize the reward
        else:
            reward = 0

        # set the observation as y_train if not None, else x_train
        ob = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]
        self.equity_ant = self.equity
        
        self.balance_ant = self.balance
        self.reward = reward

        if self.current_step >= (self.num_ticks - 1):
            self.done = True

        info = {
            "date": self.x_train[self.current_step-1, 0],
            "close": self.x_train[self.current_step-1, 4],
            "high": self.x_train[self.current_step-1, 3],
            "low": self.x_train[self.current_step-1, 2],
            "open": self.x_train[self.current_step-1, 1],
            "action": action,
            "observation": ob,
            "episode_over": self.done,
            "tick_count": self.current_step,
            "num_closes": self.num_closes,
            "balance": self.balance,
            "equity": self.equity,
            "reward": self.reward,
            "order_status": self.order_status,
            "order_volume": self.order_volume,
            "spread": self.spread,
            "margin": self.margin,
            "initial_balance": self.initial_balance
        }

        if self.order_status == 0:
            self.profit_pips = 0
            self.real_profit = 0
        
        return ob, reward, self.done, info


    def render(self, mode='human'):
        pass

    def calculate_final_debug_vars(self):
        return {
            'final_balance': self.balance,
            'final_fitness': self.reward
        }
