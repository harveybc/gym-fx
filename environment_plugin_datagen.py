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
        'sl': 3000,  # Adjusted Stop Loss
        'tp': 3000,  # Adjusted Take Profit
        'rel_volume': 0.05, # size of the new orders relative to the current balance
        'max_order_volume': 1000000, # Maximum order volume = 10 lots (1 lot = 100,000 units)
        'min_order_volume': 10000, # Minimum order volume = 0.1 lots (1 lot = 100,000 units)
        'leverage': 1000,
        'pip_cost': 0.00001,
        'min_order_time': 3,  #  Minimum Order Time to allow manual closing by an action inverse to the current order.
        'spread': 0.0002  # Default spread value
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
        self.genome = config.get('genome', None)
        self.env = AutomationEnv(x_train, y_train, self.initial_balance, self.max_steps, self.fitness_function,
                                 self.min_orders, self.sl, self.tp, self.rel_volume, self.leverage, self.pip_cost, self.min_order_time, self.spread, self.max_order_volume, self.min_order_volume, self.genome)

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
             min_orders, sl, tp, rel_volume, leverage, pip_cost, min_order_time, spread, max_order_volume, min_order_volume, genome):
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
        self.spread = spread
        self.margin = 0.0
        self.num_ticks = self.x_train.shape[0]
        self.num_closes = 0  # Track number of closes
        self.c_c = 0  # Track closing cause
        self.ant_c_c = 0  # Track previous closing cause
        self.max_order_volume = max_order_volume
        self.min_order_volume = min_order_volume
        self.orders_list = []  # List of all closed orders
        self.open_orders = []  # List of currently open orders
        
        # Set observation space based on input data
        if y_train is None:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.x_train.shape[1],), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.y_train.shape[1],), dtype=np.float32)

        # Action space is continuous from 0 to 1, controlling volume
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Set genome and reset the environment
        self.genome = genome
        self.reset(genome)


    def reset(self, genome=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.reward = 0.0
        self.done = False
        self.num_closes = 0  # Track number of closes
        self.c_c = 0  # Track closing cause
        self.ant_c_c = 0  # Track previous closing cause
        self.margin = 0.0
        self.genome = genome 
        self.kolmogorov_c = self.kolmogorov_complexity(self.genome) if self.genome is not None else 0
        self.returns = []  # Initialize returns to track rewards
        self.orders_list = []  # Initialize list to track closed orders
        self.open_orders = []  # Initialize list to track currently open orders
        self.equity_curve = [self.initial_balance]
        self.fitness = 0.0
        
        # Set initial observation based on input data
        observation = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]
        
        # Initialize info dictionary with relevant fields
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
            "spread": self.spread,
            "initial_balance": self.initial_balance
        }
        
        # Return observation, info, and max steps
        max_steps = self.max_steps
        return observation, info, max_steps


    def step(self, action_volume, verbose=True, step_fitness=0.0, genome_id=0, num_closes=0, reward_auc_prev=0.0):
        if self.done:
            return np.zeros(self.x_train.shape[1]), self.reward, self.done, {}

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Get the relevant values for the current step
        current_date = self.x_train[self.current_step, 0]
        High = self.x_train[self.current_step, 3]
        Low = self.x_train[self.current_step, 2]
        Close = self.x_train[self.current_step, 4]

        # Set closing cause to none in this tick
        self.c_c = 0

        # Process existing open orders
        total_real_profit = 0.0
        closed_orders_indices = []
        for idx, order in enumerate(self.open_orders):
            # Update profit_pips and real_profit
            if order['type'] == 'buy':
                profit_pips = ((Low - order['price']) / self.pip_cost)
            elif order['type'] == 'sell':
                profit_pips = ((order['price'] - (High + self.spread)) / self.pip_cost)
            else:
                continue  # Should not happen

            real_profit = profit_pips * self.pip_cost * order['volume']

            # Update order's profit
            order['profit_pips'] = profit_pips
            order['real_profit'] = real_profit

            # Check for SL
            if profit_pips <= -order['sl']:
                # Close order due to SL
                self.c_c = 2  # Closing cause: Stop Loss
                order['close_date'] = current_date
                order['closing_cause'] = self.c_c
                self.balance += real_profit
                self.num_closes += 1
                self.orders_list.append(order)
                closed_orders_indices.append(idx)
                if verbose:
                    print(f"{current_date} - Closed order at {Close} - Cause: Stop Loss")
            # Check for TP
            elif profit_pips >= order['tp']:
                # Close order due to TP
                self.c_c = 3  # Closing cause: Take Profit
                order['close_date'] = current_date
                order['closing_cause'] = self.c_c
                self.balance += real_profit
                self.num_closes += 1
                self.orders_list.append(order)
                closed_orders_indices.append(idx)
                if verbose:
                    print(f"{current_date} - Closed order at {Close} - Cause: Take Profit")
            else:
                # Order remains open
                total_real_profit += real_profit

        # Remove closed orders from open_orders
        for idx in sorted(closed_orders_indices, reverse=True):
            del self.open_orders[idx]

        # Update equity
        self.equity = self.balance + total_real_profit

        # Remove margin call condition (system must work even if balance is negative)

        # Open a new order at each step with the current action_volume
        if action_volume > 0:
            # Calculate order volume
            order_volume = action_volume * self.rel_volume * self.equity * self.leverage

            # Ensure order volume is within min and max limits
            if order_volume > self.max_order_volume:
                order_volume = self.max_order_volume
            if order_volume < self.min_order_volume:
                order_volume = self.min_order_volume

            # Calculate SL and TP based on action_volume
            order_tp = self.tp + action_volume * self.tp  # Varies from tp to 2*tp
            order_sl = self.sl + (1 - action_volume) * self.sl  # Varies from 2*sl to sl

            # Open a new buy order
            order = {
                'type': 'buy',
                'price': High + self.spread,
                'volume': order_volume,
                'open_date': current_date,
                'sl': order_sl,
                'tp': order_tp,
                'order_time': self.current_step,
                'profit_pips': 0.0,
                'real_profit': 0.0
            }
            self.open_orders.append(order)
            if verbose:
                print(f"{current_date} - Opened new order - Type: Buy, Price: {order['price']}, Volume: {order['volume']}, SL: {order_sl}, TP: {order_tp}")

        # Modify the reward to return only the balance change per step
        reward = self.balance - self.balance_ant
        self.balance_ant = self.balance

        # Update fitness
        self.fitness += reward

        # Set the observation as y_train if not None, else x_train
        ob = self.y_train[self.current_step] if self.y_train is not None else self.x_train[self.current_step]

        # If the episode is done, calculate the Sharpe ratio using the orders
        if self.done:
            returns = [order['real_profit'] for order in self.orders_list]
            durations_hours = [
                (order['close_date'] - order['open_date']).total_seconds() / 3600 for order in self.orders_list
            ]

            # Calculate the Sharpe ratio using the orders' profits and durations
            sharpe_ratio = self.calculate_sharpe_ratio(returns, durations_hours)

            # Calculate final fitness using the same approach as in the optimizer
            num_orders = len(self.orders_list)
            final_reward = self.fitness

            # Calculate fitness
            profit_factor = self.balance / self.initial_balance
            sqrt_orders = math.sqrt(num_orders)
            if num_orders < 1:
                self.fitness = -200
            else:
                # Loss, good behavior
                if sharpe_ratio >= 0 and sharpe_ratio <= 1:
                    self.fitness = final_reward + (profit_factor * num_orders) * (sqrt_orders + sharpe_ratio)
                # Loss, bad behavior
                elif sharpe_ratio < 0:
                    self.fitness = final_reward + (profit_factor * num_orders) * sqrt_orders
                # Profit, good behavior
                else:
                    self.fitness = final_reward + (profit_factor * num_orders) * (sqrt_orders + (sharpe_ratio ** 2))

            print(f"[ENV] genome_id: {genome_id}, balance: {self.balance}, n_ord: {len(self.orders_list)}, final_reward ({final_reward}) + sharpe_ratio ({sharpe_ratio}) = Fitness: {self.fitness}")
        else:
            sharpe_ratio = 0  # Sharpe ratio is zero if episode is not done

        # Information dictionary that includes the final balance and other metrics
        info = {
            "date": self.x_train[self.current_step - 1, 0],
            "close": self.x_train[self.current_step - 1, 4],
            "balance": self.balance,
            "equity": self.equity,
            "reward": reward,
            "c_c": self.c_c,
            "sharpe_ratio": sharpe_ratio,
            "orders": self.orders_list,
            "initial_balance": self.initial_balance
        }

        return ob, reward, self.done, info




    def calculate_sharpe_ratio(self, returns, durations_hours, annual_risk_free_rate=0.1):
        """
        Calcula el Sharpe Ratio ajustando la tasa libre de riesgo anual a la duración de la operación.
        :param returns: Lista de retornos para cada orden cerrada.
        :param durations_hours: Lista de duraciones en horas para cada orden cerrada.
        :param annual_risk_free_rate: Tasa libre de riesgo anual.
        :return: El Sharpe Ratio calculado.
        """
        if len(returns) <= 1:
            return 0

        # Convertir la tasa libre de riesgo anual a la base de la duración promedio de las órdenes
        total_duration_hours = sum(durations_hours)
        avg_duration_hours = total_duration_hours / len(durations_hours)

        # Calcular la tasa libre de riesgo ajustada para la duración promedio
        hourly_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 8760) - 1
        adjusted_risk_free_rate = (1 + hourly_risk_free_rate) ** avg_duration_hours - 1
        adjusted_risk_free_rate = 0

        # Calcular el Sharpe Ratio
        mean_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = (mean_return - adjusted_risk_free_rate) / (1+return_std) 

        #correct for low count of orders

        if sharpe_ratio > 1 and len(returns) < 7:
            sharpe_ratio = sharpe_ratio/1.5

        if sharpe_ratio > 1 and len(returns) < 5:
            sharpe_ratio = sharpe_ratio/3
        
        if sharpe_ratio > 1 and len(returns) < 3:
            sharpe_ratio = sharpe_ratio/5
        
        if sharpe_ratio > 1  and len(returns) < 7:
            sharpe_ratio = sharpe_ratio/1.5

        if sharpe_ratio > 1 and len(returns) < 5:
            sharpe_ratio = sharpe_ratio/3
        
        if sharpe_ratio > 1 and len(returns) < 3:
            sharpe_ratio = sharpe_ratio/6
        

        return sharpe_ratio



    def kolmogorov_complexity(self, genome):
        # Convert the genome to a string representation
        genome_bytes = pickle.dumps(genome)
        # Compress the genome
        compressed_data = zlib.compress(genome_bytes)
        # Return the length of the compressed data as an estimate of Kolmogorov complexity
        return len(compressed_data)

    def render(self, mode='human'):
        pass

    def calculate_final_debug_vars(self):
        return {
            'final_balance': self.balance,
            'final_fitness': self.reward
        }
