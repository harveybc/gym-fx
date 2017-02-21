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
    def __init__(self, capital, sl, tp, symbol_num, csv_f):
        metadata = {'render.modes': ['human', 'ansi']}
        # initialize initial capital
        # initialize order status    
        # load csv file
        # initialize reward values
        # initialize configuration
        
    """
    _step parameters:
    
    action set:
    discrete action 0: 0=nop,1=close,2=buy,3=sell
    discrete action 0 parameter: symbol
    (optional) continuous action 0 parameter: percent_tp, percent_sl,percent_max_volume
    
    _step return values: 
    
    observation: A concatenation of num_ticks vectors for the lastest: 
                 vector of values from timeseries, equity and its variation, 
                 order_status( -1=closed,1=opened),time_opened (normalized with
                 max_order_time), order_profit and its variation, order_drawdown
                 /order_volume_pips,  consecutive_drawdown/max_consecutive_dd
                 VER OTROS EN ESTADISTICAS DE SIMULADOR DE MQL, GENERAR GRÁFICO?   
    reward:
    """    
    def _step(self, action):
        return ob, reward, episode_over, {}  
  
  
    def _reset(self):
    ...
    def _render(self, mode='human', close=False):
    ...