from numpy import genfromtxt
from gym import utils
from gym.envs.toy_text import discrete


class ForexEnv(discrete.DiscreteEnv):
    """
    This environment simulates a Forex trading account with only one open order 
    at any time.
    
    __init__ parameters:
    
    capital: An initial_capital is loaded in the simulated account as equity.
    sl,tp:   The values for stop-loss and take-profit.
    max_volume: maximum volume of orders as percentage of equity. (def:0.1)
    max_order_time: maximum order time.
    num_ticks: number of lastest ticks to be used as obs. (def:2)
    csv_f:   A path to a CSV file containing the timeseries.
    symbol_num: The number of symbos in the timeseries.
    """

    def __init__(self, capital, min_sl, min_tp, max_sl, max_tp, leverage, num_symbols, csv_f):
        metadata = {'render.modes': ['human', 'ansi']}
        # initialize initial capital
        self.equity = capital
        self.balance = capital
        # initialize order status 0=no, 1=buy, 2=sell
        self.order_status = 0
        self.order_profit = 0.0
        # symbor of active order (from symbol list)
        self.order_symbol = 0
        # initialize reward value
        self.reward = 0
        # Min / Max SL / TP, Min / Max (Default 1000?) in pips
        self.min_sl = min_sl
        self.min_tp = mi_tp
        self.max_sl = max_sl
        self.max_tp = max_tp
        self.pip_cost = 0.00001
        # OrderVolume relative to Equity
        self.order_volume = 0.2
        # Min / Max Spread
        self.min_spread = 10
        self.max_spread = 100
        # Leverage (Default 100)
        self.leverage=leverage
        # num_symbols
        self.num_symbols = 1
        # load csv file
        self.my_data = genfromtxt(csv_f, delimiter=',')
        # initialize number of ticks from from CSV 
        # TODO: UnitTest: verificar si está volviendo número de registros(vectores) conocido, probar con mis CSV de SVM.
        # http://docs.python-guide.org/en/latest/writing/tests/
        num_ticks = len(self.my_data)

    """
    _step parameters:
    
    action from action set:
        discrete action 0: 0=nop,1=close,2=buy,3=sell
        discrete action 0 parameter: symbol
        (optional) continuous action 0 parameter: percent_tp, percent_sl, percent_of_max_volume
    
    _step return values: 
    
    observation: A concatenation of num_ticks vectors for the lastest: 
                 vector of values from timeseries, equity and its variation, 
                 order_status( -1=closed,1=opened),time_opened (normalized with
                 max_order_time), order_profit and its variation, order_drawdown
                 /order_volume_pips,  Performance?=ver archivo Reward2.xlsx tab Long-Term

    reward: Ver archivo Reward2.xlsx tab Short-Term
            TODO: ¿Perf_total=Perf*reward_acum?

    episode_over: TODO

    """

    def _step(self, action):
        # Calculates profit
        # TODO: REDONDEAR a menor fracción de lote disponible en alpari.
        if self.order_status == 1:
            self.profit_pips = ((self.current_price - self.open_price)/ self.pip_cost) * self.order_volume * self.leverage
        if self.order_status == 2:
            self.profit_pips = ((self.open_price - self.current_price)/ self.pip_cost) * self.order_volume * self.leverage
        # Calculates equity
        self.equity = self.o_equity + (self.profit_pips*self.pip_cost)
        # Verify if Margin Call
        if self.equity <= 0:
            # Close order
            self.order_status==0
            # Calculate new balance
            self.balance=self.equity
            # Set closing cause 1 = Margin call
            self.c_c = 1
        # Verify if close by SL
        if self.profit_pips <= (-1*self.max_sl):
            # Close order
            self.order_status == 0
            # Calculate new balance
            self.balance = self.equity
            # Set closing cause 2 = sl
            self.c_c = 2
        # Verify if close by TP
        if self.profit_pips >= self.max_tp:
            # Close order
            self.order_status == 0
            # Calculate new balance
            self.balance = self.equity
            # Set closing cause 3 = tp
            self.c_c = 3
        # TODO: Verify if close by Time-Limit
        # Executes action, NewState = Previous * TableOfActionsPerState :)
        if self.order_status == 0 and action == 1:
            self.order_status = 1
        if self.order_status == 0 and action == 2:
            self.order_status = 2
        if self.order_status == 1 and action == 2:
            self.order_status = 0
            self.balance = self.equity
        if self.order_status == 2 and action == 1:
            self.order_status = 0
            self.balance = self.equity
        # Calculates reward from RewardFunctionTable
        reward=0
        # Open with opened order, -128*R
        if action.
        # TODO: Sat hopefully :) grats boy, great job!
        # Returns observation and reward
        return ob, reward, episode_over, {}

    """
    _reset: coloca todas las variables en valores iniciales
    """

    def _reset(self):

        ...

    """
    _render: muestra performance de última orden, performance general y OPCIONALMENTE actualiza un gráfico del equity
     con tabla de orders y el balance por tick cuando se termine la simulación (episode_over?) similar a
    https://www.metatrader4.com/en/trading-platform/help/autotrading/tester/tester_results

    """

    def _render(self, mode='human', close=False):

        ...
