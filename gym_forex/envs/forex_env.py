import collections
import numpy
import gym
from collections import deque
from numpy import genfromtxt
from gym import utils
from gym import spaces


class ForexEnv(gym.Env):
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


    def __init__(self):
        metadata = {'render.modes': ['human', 'ansi']}
        # initialize initial capital
        capital=10000
        min_sl=100
        min_tp=100
        max_sl=200
        max_tp=200
        leverage=100
        num_symbols=1
        csv_f='D:\\Dropbox\\gym-forex\\ts_1.CSV'
        self.initial_capital=capital
        self.equity = capital
        self.balance = capital
        # for equity variation calculus
        self.equity_ant = capital
        self.order_status = 0
        self.order_profit = 0.0
        # symbor of active order (from symbol list)
        self.order_symbol = 0
        # initialize reward value
        self.reward = 0
        # Min / Max SL / TP, Min / Max (Default 1000?) in pips
        self.min_sl = min_sl
        self.min_tp = min_tp
        self.max_sl = max_sl
        self.max_tp = max_tp
        self.pip_cost = 0.00001
        # Minimum order time in ticks
        self.min_order_time=15
        # Order Volume relative to Equity
        self.rel_volume = 0.2
        # spread calculus: 0=from last csv column in pips, 1=lineal from volatility, 2=quadratic, 3=exponential
        self.spread_funct = 0
        # using spread=20 sinse its above the average plus the stddev in alpari but on
        self.spread = 20
        # Leverage (Default 100)
        self.leverage=leverage
        # num_symbols
        self.num_symbols = 1
        # initialize tick counter
        self.tick_count=0
        # Number of past ticks per feature to be used as observations (1440min=1day, 10080=1Week, 43200=1month, )
        self.obs_ticks=1440
        # flag para representación de observaciones 0=valores raw, 1=return
        self.use_return=0
        # load csv file, The file must contain 16 cols: the 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<6 indicators>
        self.my_data = genfromtxt(csv_f, delimiter=',')
        # initialize number of ticks from from CSV
        self.num_ticks = len(self.my_data)
        # initialize number of columns from the CSV
        self.num_columns = len(self.my_data[0])
        # Generate pre-processing inputs- TODO(0=no,1=FFT_maxamp,2=Poincare for 1/f(FFT_max_amp),3=FFT_2ndamp,4=Poincare for 3),etc...
        self.preprocessing=0
        # Select the column from which pre-processing observations will be generated
        self.preprocessing_column=0
        # Normalization method=0 deja los datos iguales, 1=normaliza, 2= estandariza, 3= estandariza y trunca a rango -1,1
        self.norm_method=1
        # Initialize arrays for normalization and standarization (min,max, average, stddev)
        self.max=self.num_columns*[-9999999.0]
        self.min=self.num_columns*[9999999.0]
        self.promedio=self.num_columns*[0.0]
        self.stddev=self.num_columns*[0.0]
        if self.norm_method>0:
            for i in range(0, self.num_ticks - 1):
                # para cada columna
                for j in range(0, self.num_columns - 1):
                    # actualiza max y min
                    if self.my_data[i,j]>self.max[j]:
                        self.max[j]=self.my_data[i, j]
                    if self.my_data[i,j]<self.min[j]:
                        self.min[j]=self.my_data[i,j]
                    # incrementa acumulador
                        self.promedio[j]=self.promedio[j]+self.my_data[i,j]
            self.promedio = [x/self.num_ticks for x in self.promedio]
        if self.norm_method>1:
            for i in range(0, self.num_ticks - 1):
                # para cada columna
                for j in range(0, self.num_columns - 1):
                        # calcula cuadrados de distancia a promedio
                        self.stddev[j]=self.stddev[j]+(self.my_data[i,j]-self.promedio)**2
        #calcula promedio y stddev
        self.stddev=[(x/self.num_ticks)**0.5 for x in self.stddev]
        # TODO: IF REWARD TABLE IS USED, SET THE NUMBER OR STATE COLS TO 18?
        # matrix for the state(order status, equity variation, reward and statistics (from reward table))
        self.state_columns=3
        # Serial data - to - parallel observation matrix and state matrix
        self.obs_matrix = self.num_columns*[deque( self.obs_ticks*[0.0], self.obs_ticks)]
        self.state=self.state_columns*[deque( self.obs_ticks*[0.0], self.obs_ticks)]
        # reward function 0=equity variation, 1=Table
        self.reward_function=0
        # action space = nop,buy,sell
        self.action_space=spaces.Discrete(3)
        # observation_space=(16 columns + 3 state variables)* obs_ticks, shape=(width,height, channels?)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_ticks, 1, 19))
        self.order_time=0
        #TODO; Quitar cuando se controle SL Y TP
        self.sl=self.max_sl
        self.tp=self.max_tp
    """
    _step parameters:
    
    action from action set:
        discrete action 0: 0=nop,1=buy,2=sell. 
#TODO: PROBAR CON 4 ACCIONES: 0=NOP,1=BUY, -1=SELL, CLOSE
        discrete action 0 parameter: symbol
        (optional) continuous action 0 parameter: percent_tp, percent_sl, percent_of_max_volume
    
    _step return values: 
    
    observation: A concatenation of num_ticks vectors for the lastest: 
                 vector of values from timeseries, equity and its variation, 
                 order_status( 0 nop, -1=closed,1=opened),time_opened (normalized with
                 max_order_time), order_profit and its variation, order_drawdown
                 /order_volume_pips,  Performance?=ver archivo Reward2.xlsx tab Long-Term

    reward: Ver archivo Reward2.xlsx tab Short-Term
            TODO: Perf_total=Perf*reward_acum?
    episode_over: Imprime statistics

    """

    def _step(self, action):
        # read time_variables from CSV. Format: 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<num_columns>
        High = self.my_data[self.tick_count, 0]
        Low = self.my_data[self.tick_count, 1]
        Close = self.my_data[self.tick_count, 2]
        DoW=self.my_data[self.tick_count, 7]
        HoD=self.my_data[self.tick_count, 8]
        MoY=self.my_data[self.tick_count, 5]
        DoM = self.my_data[self.tick_count, 6]
        MoH=self.my_data[self.tick_count, 9]
        # Elevate spread  at 0 hours and if its weekend (DoW<=2 and Hour < 2)or(DoW>=5 and Hour > 23)
        if (DoW<1 or DoW>5)or(HoD < 2 and HoD > 23):
            spread=60
        else:
            spread=20

        # Calculates profit
        self.profit_pips=0
        self.real_profit_pips=0
        if self.order_status == 1:
            # Low_Bid - order_open min and real profit pips
            self.profit_pips = ((Low-self.open_price)/ self.pip_cost) * self.order_volume * self.leverage
            self.real_profit_pips = ((Close - self.open_price) / self.pip_cost) * self.order_volume * self.leverage
        if self.order_status == -1:
            # Order_open - High_Ask (High+spread)
            self.profit_pips = ((self.open_price - (High+spread))/ self.pip_cost) * self.order_volume * self.leverage
            self.real_profit_pips = ((self.open_price - (Close + spread)) / self.pip_cost) * self.order_volume * self.leverage
        # Calculates min equity
        self.min_equity = self.balance + (self.profit_pips*self.pip_cost)
        self.equity = self.balance + (self.real_profit_pips*self.pip_cost)
        # Verify if Margin Call
        episode_over = bool(0)
        if self.min_equity < 0:
            # Close order
            self.order_status==0
            # Calculate new balance
            self.balance=self.min_equity
            # Set closing cause 1 = Margin call
            self.c_c = 1
            # End episode
            episode_over = bool(1)
        #TODO: ADICIONAR CONTROLES PARA SL Y TP ENTRE MAX_SL Y TP
            # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
            print self.tick_count,',',MoY,'-',DoM,' ',HoD,':',MoH,',margin_call,', 0,',',\
                0, ',', 0, ',', 0,',',0,',',self.balance
        # Verify if close by SL
        if self.profit_pips <= (-1*self.max_sl):
            # Close order
            self.order_status == 0
            # Calculate new balance
            # TODO: NO USAR MIN_EQUITY SINO OPEN_ORDER_PRICE +/- SL
            self.balance = self.min_equity
            # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
            print self.tick_count, ',', MoY, '-', DoM, ' ', HoD, ':', MoH, ',stop_loss,', self.order_volume, ',', \
                self.open_price, ',', self.sl, ',', self.tp, ',', self.real_profit_pips , ',', self.balance
            # Set closing cause 2 = sl
            self.c_c = 2
        # Verify if close by TP
        if self.profit_pips >= self.max_tp:
            # Close order
            self.order_status == 0
            # Calculate new balance
            # TODO: NO USAR MIN_EQUITY SINO OPEN_PRICE +/- TP
            self.balance = self.min_equity
            # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
            print self.tick_count, ',', MoY, '-', DoM, ' ', HoD, ':', MoH, ',take_profit,', self.order_volume, ',', \
                self.open_price, ',', self.sl, ',', self.tp, ',', self.real_profit_pips, ',', self.balance
            # Set closing cause 3 = tp
            self.c_c = 3
        # TODO: Hacer opcion realista de ordenes que se ABREN Y CIERRAN solo si durante el siguiente minuto
        #       el precio de la orden(close) no es high o low del siguiente candle.
        # Executes action, NewState = Previous * TableOfActionsPerState :)
        if self.order_status == 0 and action == 1:
            self.order_status = 1
            # open price = Ask (Close_bid+Spread)
            self.open_price = Close + spread
            # TODO: Colocar accion para tamano de lote con rel_volume como maximo al abrir una orden
            # TODO: Colocar acciones para SL y TP con los actuales como maximos
            # order_volume=equity*rel_volume*leverage/100000
            self.order_volume = self.equity * self.rel_volume / 100000
            self.order_time=self.tick_count
            # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
            print self.tick_count, ',', MoY, '-', DoM, ' ', HoD, ':', MoH, ',buy,', self.order_volume, ',', \
                self.open_price, ',', self.sl, ',', self.tp, ',', self.real_profit_pips, ',', self.balance
        if self.order_status == 0 and action == 2:
            self.order_status = -1
            # open_price = Bid
            self.open_price = Close
            # TODO: Colocar accion para tamano de lote con rel_volume como maximo al abrir una orden
            # order_volume=equity*rel_volume*leverage/100000
            self.order_volume = self.equity * self.rel_volume / 100000
            self.order_time = self.tick_count
            # TODO: Hacer version con controles para abrir y cerrar para buy y sell independientes,comparar
            # TODO: Verificar si ha pasado el min_order_time desde que se abrieron antes de cerrar
            # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
            print self.tick_count, ',', MoY, '-', DoM, ' ', HoD, ':', MoH, ',sell,', self.order_volume, ',', \
                self.open_price, ',', self.sl, ',', self.tp, ',', self.real_profit_pips, ',', self.balance
        if ((self.tick_count-self.order_time)>self.min_order_time):
            if self.order_status == 1 and action == 2 :
                self.order_status = 0
                self.balance = self.equity
                # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                print self.tick_count, ',', MoY, '-', DoM, ' ', HoD, ':', MoH, ',close_buy,', self.order_volume, ',', \
                    self.open_price, ',', self.sl, ',', self.tp, ',', self.real_profit_pips, ',', self.balance
            if self.order_status == 2 and action == 1:
                self.order_status = 0
                self.balance = self.equity
                # print transaction: Num,DateTime,Type,Size,Price,SL,TP,Profit,Balance
                print self.tick_count, ',', MoY, '-', DoM, ' ', HoD, ':', MoH, ',close_sell,', self.order_volume, ',', \
                    self.open_price, ',', self.sl, ',', self.tp, ',', self.real_profit_pips, ',', self.balance
        # Calculates reward from RewardFunctionTable
        reward=0
        equity_increment=self.equity_ant-self.equity
        if self.reward_function==0:
            reward=equity_increment
        # TODO: REWARD FUNCTION:  1=Tabla
        # Push values from timeseries into state
        # 0 = HighBid, 1 = Low, 2 = Close, 3 = NextOpen, 4 = v, 5 = MoY, 6 = DoM, 7 = DoW, 8 = HoD, 9 = MoH, ..<num_columns>
        for i in range(0, self.num_columns-1):
            #TODO: Normalizar el valor de my_data a introducir  en obs_matrix usando norm_array
            self.obs_matrix[i].append(self.my_data[self.tick_count, i])
        # matrix for the state(order status, equity variation, reward and statistics (from reward table))
        # TODO: Normalizar cada componente del estado.
        self.state[0].append(self.order_status)
        self.state[1].append(self.equity - self.equity_ant)
        self.state[2].append((self.equity-self.initial_capital)/self.initial_capital)
        # merge obs_matrix and state in ob
        ob=numpy.concatenate([self.obs_matrix,self.state])
        # increment tick counter
        self.tick_count=self.tick_count+1
        # Episode over es TRUE cuando se termina el juego, es decir cuando tick_count=self.num_ticks
        if self.tick_count>=self.num_ticks:
            episode_over=bool(1)
            print 'Eq=', self.equity
            #TODO: IMPRIMIR ESTADiSTICAS DE METATRADER
        # end of step function.
        return ob, reward, episode_over, {}

    """
    _reset: coloca todas las variables en valores iniciales
    """

    def _reset(self):
        self.tick_count=0
        self.equity=0
        self.balance=0
        self.equity_ant=0
        self.obs_matrix = self.num_columns * [deque(self.obs_ticks * [0.0], self.obs_ticks)]
        self.state = self.state_columns * [deque(self.obs_ticks * [0.0], self.obs_ticks)]
        self.order_status=0
        self.reward=0
        self.order_profit=0.0
        # Serial data - to - parallel observation matrix and state matrix
        self.obs_matrix = self.num_columns*[deque( self.obs_ticks*[0.0], self.obs_ticks)]
        self.state=self.state_columns*[deque( self.obs_ticks*[0.0], self.obs_ticks)]
        ob = numpy.concatenate([self.obs_matrix, self.state])
        return ob


    """
    _render: muestra performance de ultima orden, performance general y OPCIONALMENTE actualiza un grafico del equity
     con tabla de orders y el balance por tick cuando se termine la simulacion (episode_over?) similar a
    https://www.metatrader4.com/en/trading-platform/help/autotrading/tester/tester_results
    def _render(self, mode='human', close=False):
        print 'Eq=', self.equity
    """



