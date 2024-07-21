import collections
from collections import deque
import gym
from gym import spaces
from gym import utils
import math
import numpy as np
import numpy
from numpy import genfromtxt
import copy
# TODO: Swap rates
class ForexEnvMulti(gym.Env):
    """
    This environment simulates a Forex trading account with only one open order 
    at any time.
    
    Version multi performs trading in multiple simultaneous symbols and uses
    separate action and observation timeseries input. 
    
    Action timeseries input are the raw prices of the symbol (H,L,C,spread?). It
    is not normalized and it is used for the trading simulation system.
    
    Observation timeseries input is the preprocessing output corresponding to
    each tick of the action timeseries input.  It is used as input to the 
    action controller that decides the action to apply in the action timeseries.
    It may contain also pre-processed techincal indicators.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        metadata = {'render.modes': ['human', 'ansi']}
        # initialize environment variables
        self.num_symbols = kwargs['num_symbols']
        self.num_features = kwargs['num_features']
        self.num_components = kwargs['num_components']
        # initial capital un USD
        self.capital = kwargs['capital']
        # maximum and minimum take_profit and stop_loss in pips
        self.min_sl = kwargs['min_sl']
        self.min_tp = kwargs['min_tp']
        self.max_sl = kwargs['max_sl']
        self.max_tp = kwargs['max_tp']
        # maximum number of simultaneous orders, max 1 order per symbol
        self.max_orders = kwargs['max_orders']
        # Order Volume relative to Equity 
        self.max_volume = kwargs['max_volume']
        self.leverage = kwargs['leverage']
        # Number of past ticks per feature to be used as observations 
        # self.obs_ticks = kwargs['obsticks'] # best 48@ 700k
        # Window size is the number of past ticks per features to be used as observations 
        self.window_size = kwargs['window_size'] # best 48@ 700k
        # file path for the action dataset (non pre-processed prices)
        # csv_f = kwargs['dataset'] 
        csv_action = kwargs['csv_action']
        # file path for the observation dataset (pre-processed prices and technical indicators)
        #self.dataset = kwargs['dataset']
        self.csv_observation = kwargs['csv_observation']
        # number of timesignals in the action dataset
        self.num_timesignals = kwargs['num_timesignals']
        # minimum number of orders to depenalize reward
        self.min_orders = 4
        # counter of closed orders per symbol 
        self.num_closes = [0] * self.num_symbols
        self.num_closes_t = 0
        # Closing cause for each symbol's last order
        self.c_c = [0]*self.num_symbols
        # Closing cause general
        self.c_c_g = 0
        # anterior closing cause para verificar consecutice drawdown
        self.ant_c_c = [0]*self.num_symbols
        # variable to indicate episode over
        self.episode_over = bool(0)
        # Show debug msgs
        self.debug = 1  
        # initialize account-wide values
        self.equity = self.capital
        self.balance = self.capital
        self.balance_ant = self.capital
        # for equity variation calculus
        self.equity_ant = self.capital
        # order status for each symbol: -1=sell, 1=buy, 00=nop
        self.order_status = [0] * self.max_orders
        self.order_profit = [0.0] * self.max_orders
        # initialize reward value
        self.reward = 0.0
        # Min / Max SL / TP, Min / Max (Default 1000?) in pips
        self.pip_cost = [0.00001] * self.num_symbols
        # margin  = open_price*volume*100000/leverage , uno por orden, el acumulativo se compara para verificar si hay margin call
        self.margin = 0.0 * self.num_symbols
        # margin acumulativo, se compara con equity para verificar si hay margin call.
        self.margin_a = 0.0 
        # Minimum order time in ticks, its zero for the hourly timeframe
        self.min_order_time = 0
        # spread calculus: 0=from last csv column in pips, 1=lineal from volatility, 2=quadratic, 3=exponential, 4=constant
        self.spread_funct = 4
        # using spread=25 as average since its above the average plus the stddev in alpari but on
        self.spread = [25]*self.num_symbols

        # load action dataset, it contains high, low, close, and spread for each symbol
        self.a_data = genfromtxt(csv_action, delimiter=',', skip_header=0)
        # load csv file, The file must contain 16 cols: the 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<6 indicators>
        self.o_data = genfromtxt(self.csv_observation, delimiter=',', skip_header=0)
        # initialize number of ticks from from CSV
        self.num_ticks = len(self.a_data)
        #verify if observation and action have the same number of ticks
        if (self.num_ticks != len(self.o_data)):
            print("Error: len(a_data) != len(o_data)")
        # initialize number of columns from the observation CSV
        self.num_columns_a = len(self.a_data[0])         
        # initialize number of columns from the observation CSV
        self.num_columns_o = len(self.o_data[0])
        # the observation matrix is an array containing a 2d matrix deque with length self.window_size
        historic = deque([[0.0]*self.num_components ]*self.window_size , self.window_size)
        # Observation matrix 
        self.obs_matrix = [[[None] * self.num_components] * self.window_size ]* (self.num_features*self.num_symbols)
        for i in range(0, (self.num_features*self.num_symbols)):
            self.obs_matrix[i] = copy.deepcopy(historic)
        for i in range(0, self.window_size):
            for j in range(0, self.num_columns):
                self.obs_matrix[j].appendleft(self.o_data[i, j])
                #self.obs_matrix = self.num_columns * [deque(self.obs_tick4
        self.tick_count = self.window_size
        # set action space to 3 actions, 0=nop, 1=buy, 2=sell
        # ACTION SPACE  = (TP/TPMAX, SL/SLMAX,  VOLUME/VOLUMEMAX, DIRECTION), symbol
        self.action_space = spaces.Box(low=float(-1.0), high=float(1.0), shape=(4,self.num_symbols), dtype=np.float32)
        # observation_space=(16 columns + 3 state variables)* obs_ticks, shape=(width,height, channels?)
        self.observation_space = spaces.Box(low=float(-1.0), high=float(1.0), shape=(self.obs_ticks,  self.num_components,  self.num_features), dtype=np.float32)
        self.order_time = [0] * self.max_orders
        # TODO; Quitar cuando se controle SL Y TP
        self.sl = [self.max_sl] * self.max_orders
        self.tp = [self.max_tp] * self.max_orders
        print ("Finished INIT function")

    """
    _step parameters:
    
    action from action set:
        (TP/TPMAX, SL/SLMAX, VOLUME/VOLUMEMAX, DIRECTION), symbol. 
        
    _step return values: 
        observation: A concatenation of (num_features*num_symbols) 2D matrixes 
        with (num_components, window_size)

    reward: Area under profit curve

    self.episode_over: Imprime statistics

    """

    def step(self, action):
        # number of columns per symbol in action dataset
        cps = (self.num_columns_a - self.num_timesignals)//self.num_symbols
        # initialize hlc price arrays for each symbol
        High  = [0.0] * self.num_symbols
        Low   = [0.0] * self.num_symbols
        Close = [0.0] * self.num_symbols
        
        # read hlc from action dataset 
        for i in range(0,self.num_symbols):
            High  = self.a_data[self.tick_count, 0 + (i*cps)]
            Low   = self.a_data[self.tick_count, 1 + (i*cps)]
            Close = self.a_data[self.tick_count, 2 + (i*cps)]
            
        # read time variables from last columns of action dataset
        DoW = self.a_data[self.tick_count, 146]
        HoD = self.a_data[self.tick_count, 147]
        # Elevate spread  at 0 hours and if its weekend (DoW<=2 and Hour < 2)or(DoW>=5 and Hour > 23)
        if (DoW < 1 or DoW > 5) or (HoD < 2 and HoD > 23):
            spread = self.pip_cost * 60
        else:
            spread = self.pip_cost * 25

        # Calculates profit
        self.profit_pips = [0]*self.num_symbols
        self.real_profit = [0.0]*self.num_symbols
        # For each order (max one per symbol), calculate the new order status and new balance/equity/margin 
        for i in range (0, self.num_symbols):
            # calculate order starus for existing BUY order (status=1)
            if self.order_status[i] == 1:
                # Low[i]_Bid - order_open min and real profit pips (1 lot = 100000 units of currency)
                self.profit_pips[i] = ((Low[i] - self.open_price[i]) / self.pip_cost[i])
                self.real_profit[i] = self.profit_pips[i] * self.pip_cost[i] * self.order_volume[i] * 100000

            # calculate order status for existing SELL order (status=-1)
            elif self.order_status[i] == -1:
                # Order_open - High[i]_Ask (High[i]+spread[i])
                self.profit_pips[i] = ((self.open_price[i] - (High[i] + spread[i])) / self.pip_cost[i])
                self.real_profit[i] = self.profit_pips[i] * self.pip_cost[i] * self.order_volume[i] * 100000
            else:
                self.profit_pips[i] = 0
                self.real_profit[i] = 0
			
            # Calculates equity with the profit of all orders
            self.equity = self.balance
            for j in range(0, self.sum_symbols):
	            self.equity = self.equity + self.real_profit[j]
            
            # Verify if Margin Call, CLOSE ALL ORDERS
            if self.equity < self.margin_a:
                for j in range(0,self.num_symbols):
                    # Close order
                    self.order_status[j] = 0
                    # Calculate new balance
                    self.balance = 0.0
                    # Calculate new balance
                    self.equity = 0.0
                    # reset margin
                    self.margin[i] = 0.0
                    # reset margin
                    self.margin_a = 0.0
                    # reset profit in pips
                    self.profit_pips[j] = 0
                    self.real_profit[j] = 0
                    # Set closing cause 1 = Margin call
                    self.ant_c_c[j] = self.c_c[j]
                    self.c_c[j] = 1
                    self.c_c_g = 1
                    # increments number of orders counter
                    self.num_closes[i] += 1
                    # End episode
                    self.episode_over = bool(1)
                    # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                    if self.debug == 1:
                        print('MARGIN CALL - Balance =', self.equity, ',  Reward =', self.reward, 'Time=', self.tick_count)
           
            if (self.episode_over == False):
                # Verify if close by SL
                if self.profit_pips[i] <= (-1 * self.sl[i]):
                    # Close order
                    self.order_status[i] = 0
                    # Calculate new balance as prev balance  plus profit
                    self.balance = self.balance + self.real_profit[i]
                    # resets margin
                    # para cada orden
                    self.margin_a = self.margin_a - self.margin[i]
                    self.margin[i] = 0.0
                    # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                    if self.debug == 1:
                        print(self.tick_count, ',stop_loss, pips:', self.profit_pips[i],' profit:', self.real_profit[i], ',b:', self.balance)
                    # Set closing cause 2 = sl
                    self.ant_c_c[i] = self.c_c[i]
                    self.c_c[i] = 2
                    self.c_c_g = 2
                    # reset profit in pips
                    self.profit_pips[i] = 0
                    self.real_profit[i] = 0
                    # increments number of orders counter
                    self.num_closes[i] += 1
                    
                # Verify if close by TP
                if self.profit_pips[i] >= self.tp:
                    # Close order
                    self.order_status[i] = 0
                    # Calculate new balance as prev balance plus profit
                    self.balance = self.balance + self.real_profit[i]
                    # reset margin 
                    self.margin_a = self.margin_a - self.margin[i]
                    self.margin[i] = 0.0
                    # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                    if self.debug == 1:
                        print(self.tick_count, ',take_profit, pips:', self.profit_pips[i],' profit:', self.real_profit[i], ',b:', self.balance)
                    # Set closing cause 3 = tp
                    self.ant_c_c[i] = self.c_c[i]
                    self.c_c[i] = 3
                    self.c_c_g = 3
                    # reset profit in pips
                    self.profit_pips[i] = 0
                    self.real_profit[i] = 0
                    # increment the counter for the number of orders closed
                    self.num_closes[i] += 1

                # Executes BUY action, order status  = 1
                if (self.order_status[i] == 0) and (action[3][i]> 0):
                    self.order_status[i] = 1
                    # open price = Ask (Close_bid+Spread)
                    self.open_price[i] = Close[i] + spread[i]
                    # order_volume = lo que alcanza con rel_volume de equity
                    # Calcula sl y tp desde action space
                    #print("\naction=",action[0]);
                    self.tp[i]= (self.max_tp) * (action[0][i])
                    self.sl[i] = (self.max_sl) * (action[1][i])
                    #self.tp = self.min_tp + ((self.max_tp-self.min_tp) * ((action[0] + 1) / 2))
                    #self.sl[i] = self.min_sl + ((self.max_sl-self.min_sl) * ((action[1] + 1) / 2))
                    #self.sl[i] = self.max_sl
                    #self.tp = self.max_tp
                    # a=Tuple((Discrete(3),  Box(low=-1.0, high=1.0, shape=3, dtype=np.float32)) # nop, buy, sell vol,tp,sl
                    #self.order_volume[i] = self.equity * self.max_volume * self.leverage * ((action[2] + 1) / 2) / 100000
                    self.order_volume[i] = self.equity * self.max_volume * self.leverage * action[2][i]/ 100000
                    #self.order_volume[i] = self.equity * self.max_volume * self.leverage / 100000
                    # redondear a volumenes minimos de 0.01
                    self.order_volume[i] = math.trunc(self.order_volume[i] * 100) / 100.0
                    # si volume menos del mínimo, hace volumen= mínimo TODO: QUITAR? CUANDO SE CALCULE VOLUME
                    if self.order_volume[i] <= 0.01:
                        # close existing order
                        self.order_volume[i] = 0.01
                        self.margin[i] = 0.0
                    # set the new margin
                    self.margin[i] = (self.order_volume[i] * 100000 / self.leverage)
                    self.margin_a = self.margin_a + self.margin[i]
             
                    # TODO: Colocar accion para tamano de lote con rel_volume como maximo al abrir una orden
                    self.order_time = self.tick_count
                    # print transaction: Num,DateTime,Type,Size,Price,SL,TP,margin,equity
                    if self.debug == 1:
                        print(self.tick_count, ',buy, o', self.open_price[i], ',v', self.order_volume[i], ' tp:', self.tp[i], ' sl:', self.sl[i], ' b:', self.balance)

                # Executes SELL action, order status  = 1
                if (self.order_status[i] == 0) and (action[3][i]< 0):
                    self.order_status[i] = -1
                    # open_price = Bid
                    self.open_price[i] = Close[i]
                    # Calcula sl y tp desde action space
                    # print("\naction=", action[0]);
                    # self.sl[i] = self.max_sl 
                    # self.tp = self.max_tp
                    #self.tp = self.min_tp + ((self.max_tp-self.min_tp) * ((action[0] + 1) / 2))
                    #self.sl[i] = self.min_sl + ((self.max_sl-self.min_sl) * ((action[1] + 1) / 2))
                    self.tp[i] = (self.max_tp) * (action[0][i])
                    self.sl[i] = (self.max_sl) * (action[1][i])
                    # TODO: ADICIONAR VOLUME DESDE ACTION SPACE 
                    # a=Tuple((Discrete(3),  Box(low=-1.0, high=1.0, shape=3, dtype=np.float32)) # nop, buy, sell vol,tp,sl
                    #self.order_volume[i] = self.equity * self.max_volume * self.leverage/ 100000
                    self.order_volume[i] = self.equity * self.max_volume * self.leverage * (action[2][i]) / 100000
                    # redondear a volumenes minimos de 0.01
                    self.order_volume[i] = math.trunc(self.order_volume[i] * 100) / 100.0
                    # set the new margin
                    self.margin[i] = (self.order_volume[i] * 100000 / self.leverage)
                    self.margin_a = self.margin_a + self.margin[i]
                    self.order_time = self.tick_count
                    # TODO: Hacer version con controles para abrir y cerrar para buy y sell independientes,comparar
                    # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                    if self.debug == 1:
                        print(self.tick_count, ',sell, o', self.open_price[i], ',v', self.order_volume[i], ' tp:', self.tp[i], ' sl:', self.sl[i], ' b:', self.balance)

                # Verify si ha pasado el min_order_time desde que se abrieron antes de cerrar
                if ((self.tick_count - self.order_time) > self.min_order_time):
                    # Closes EXISTING SELL (-1) order with action=BUY (1)
                    if (self.order_status[i] == -1) and (action[3][i] > 0):
                        self.order_status[i] = 0
                        # Calculate new balance as prev balance plus profit
                        self.balance = self.balance + self.real_profit[i]
                        # reset margin 
                        self.margin_a = self.margin_a - self.margin[i]
                        self.margin[i] = 0.0
                        # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                        if self.debug == 1:
                            print(self.tick_count, ',close_sell, pips:', self.profit_pips[i],' profit:', self.real_profit[i], ',b:', self.balance)
                        # Set closing cause 0 = normal close
                        self.ant_c_c[i] = self.c_c[i]
                        self.c_c[i] = 0
                        self.c_c_g = 0
                        # reset profit in pips
                        self.profit_pips[i] = 0
                        self.real_profit[i] = 0
                        # increment counter for number of orders closed
                        self.num_closes[i] += 1
                    #if action == 0 (nop), print status
                    if (self.order_status[i] == -1) and (action[3][i] == 0):
                        print(self.tick_count, ',o_sell, pips:', self.profit_pips[i],' profit:', self.real_profit[i], ',b:', self.balance)
                    # print("action=", action)
                    # Closes EXISTING BUY (1) order with action=SELL (2)
                    if (self.order_status[i] == 1) and (action[3][i] < 0):
                        self.order_status[i] = 0
                        # Calculate new balance as prev balance plus profit
                        self.balance = self.balance + self.real_profit[i]
                        # reset margin 
                        self.margin_a = self.margin_a - self.margin[i]
                        self.margin[i] = 0.0
                        # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                        if self.debug == 1:
                            print(self.tick_count, ',close_buy, pips:', self.profit_pips[i],' profit:', self.real_profit[i], ',b:', self.balance)
                        # Set closing cause 0 = normal close
                        self.ant_c_c[i] = self.c_c[i]
                        self.c_c[i] = 0
                        self.c_c_g = 0
                        # reset profit in pips
                        self.profit_pips[i] = 0
                        self.real_profit[i] = 0
                        # incrments counter of closed orders
                        self.num_closes[i] += 1
                    if (self.order_status[i] == 1) and (action[3][i] == 0):
                        print(self.tick_count, ',o_buy, pips:', self.profit_pips[i],' profit:', self.real_profit[i], ',b:', self.balance)
                    
        # end for each symbol
        
        
        # Calculates reward from RewardFunctionTable
        equity_increment = self.equity - self.equity_ant
        balance_increment = self.balance - self.balance_ant
        if self.reward_function == 0:
            # TODO: REWARD FUNCTION:  1=Tabla
            bonus = ((self.equity - self.initial_capital) / self.num_ticks)
            # reward = reward + bonus
            reward = (balance_increment + bonus) / 2
            # penaliza inactividad hasta alcanzar total de ticks con 5 para que tenga menos que los de balance positivo con mal comportamiento
            #if equity_increment == 0.0:
            #    reward = reward - (2*self.initial_capital / self.num_ticks)
            # premia incrementos
            #if equity_increment > 0.0:
            #    reward = reward + (self.initial_capital / self.num_ticks)
            # calculate total number of orders
            self.num_closes_t = 0
            for j in range(0,self.num_symbols):
                self.num_closes_t = self.num_closes_t + self.num_closes[j]
            
            # penaliza hardly if less than min_orders/2 
            if (self.num_closes_t < self.min_orders/2) and reward > 0:
                reward = reward * (self.num_closes/self.min_orders)
            if (self.num_closes_t < self.min_orders/2) and reward <= 0:
                reward = reward - (self.initial_capital / self.num_ticks) * (1-(self.num_closes/self.min_orders))
    
            # penaliza lightly if less than min_orders
            if (self.num_closes_t < self.min_orders) and reward <= 0:
                reward = reward - ((self.initial_capital / (10*self.num_ticks))* (1-(self.num_closes/self.min_orders)))
            # penaliza margin call
            if self.c_c_g == 1:
                reward = -(5.0 * self.initial_capital)
            # penaliza red que no hace nada
            if self.tick_count >= (self.num_ticks - 2):
                if self.num_closes_t < self.min_orders:
                    reward = -(10*self.initial_capital * (1 - (self.num_closes_t / self.min_orders)))
                    self.balance = 0
                    self.equity = 0
                if self.equity == self.initial_capital:
                    reward = -(10.0 * self.initial_capital)
                    self.balance = 0
                    self.equity = 0
                    
            reward = reward / self.initial_capital
            # if self.order_status==0:
            # TODO: penalizar reward con el cuadrado del tiempo que lleva sin orden * -0.01
                # para evitar que sin acciones se obtenga ganancia 0 al final (deseado: -2, entonces variación=-2/num_ticks)
                # TODO: Auto-calcular reward descontado por inectividad como función del total de ticks?
                # reward=reward-0.00001 #Best result con 0.0001 (148k)

        # Push values from timeseries into state
        # 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<num_columns>
        for i in range(0, self.num_columns - 1):
            self.obs_matrix[i].appendleft(self.o_data[self.tick_count, i])
        # matrix for the state(order status, equity variation, reward and statistics (from reward table))
        ob = self.obs_matrix
        # increment tick counter
        self.tick_count = self.tick_count + 1
        # update equity_Ant
        self.equity_ant = self.equity
        self.balance_ant = self.balance
        self.reward = self.reward + reward
        # Episode over es TRUE cuando se termina el juego, es decir cuando tick_count=self.num_ticks
        if self.tick_count >= (self.num_ticks - 1):
            self.episode_over = bool(1)
            
            # print('Done - Balance =', self.equity, ',  Reward =', self.reward, 'Time=', self.tick_count)
            # self._reset()
            # self.__init__()
            # TODO: IMPRIMIR ESTADiSTICAS DE METATRADER
        # end of step function.
        info = {"balance":self.balance, "tick_count":self.tick_count, "order_status":self.order_status, "num_closes":self.num_closes, "equity": self.equity}
        return ob, reward, self.episode_over, info

    """
    _reset: coloca todas las variables en valores iniciales
    """

    def reset(self):
        self.equity = self.initial_capital
        self.balance = self.equity
        self.balance_ant = self.balance
        self.equity_ant = self.equity
        #print ("First my_data row = ", self.my_data[0,:])
        #print ("obs_ticks = ", self.obs_ticks)
        for i in range(0, self.num_columns):
            for j in range(0, self.obs_ticks):
                self.obs_matrix[i].appendleft(self.o_data[j, i])
        #    print ("obs_matrix_pre[",i,"] = ", self.obs_matrix[i])
        self.tick_count = self.obs_ticks
        self.order_status = [0] * self.num_symbols
        self.reward = 0.0
        self.order_profit = [0.0] * self.num_symbols
        self.margin = 0.0
        self.c_c = [0] * self.num_symbols
        self.ant_c_c = [0] * self.num_symbols
        self.num_closes = [0] * self.num_symbols
        #self.__init__(self.dataset)
        self.episode_over = bool(0)
        return self.obs_matrix

    """
    _render: muestra performance de ultima orden, performance general y OPCIONALMENTE actualiza un grafico del equity
     con tabla de orders y el balance por tick cuando se termine la simulacion (self.episode_over?) similar a
    https://www.metatrader4.com/en/trading-platform/help/autotrading/tester/tester_results
    def _render(self, mode='human', close=False):
        print 'Eq=', self.equity
    """

    def render(self, mode='human', close=False):
        if mode == 'human':
            return self.equity
        else:
            super(ForexEnvMulti, self).render(mode=mode)  # just raise an exception
