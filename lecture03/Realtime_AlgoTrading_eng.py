import os
os.chdir('/Users/songyouk/PairsTradingAutomation/lecture03')
from PairTrading_eng import *
import pytz
from ib_insync import *
import datetime
import pandas_market_calendars as mcal

import logging

logging.basicConfig(filename='/Users/songyouk/PairsTradingAutomation/lecture03/app.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Script started')

try:
    if ib.isConnected():
        print('Connection is already established')
        logging.info('Connection is already established')
except NameError:
    util.startLoop()  # needed for Jupyter
    ib = IB()
    ib.connect()
    logging.info('Connection established')


if os.path.exists('/Users/songyouk/PairsTradingAutomation/data/last_pairs_update.txt'):
    with open('/Users/songyouk/PairsTradingAutomation/data/last_pairs_update.txt', 'r') as f:
        last_pairs_update = datetime.datetime.strptime(f.read(), '%Y-%m-%d %H:%M:%S.%f')  
        logging.info('old last_pairs_update')      
else:
    last_pairs_update = datetime.datetime.now() - datetime.timedelta(days=7)
    logging.info('new last_pairs_update') 
    
time_diff = datetime.datetime.now() - last_pairs_update

if time_diff >= datetime.timedelta(days=1.5): 
    os.system('kaggle kernels output dtmanager1979/stock-trading-eda-scheduled -p /Users/songyouk/PairsTradingAutomation/')
    last_pairs_update = datetime.datetime.now()
    with open('/Users/songyouk/PairsTradingAutomation/data/last_pairs_update.txt', 'w') as f:
        f.write(str(last_pairs_update))
        logging.info('create last_pairs_update.txt') 


df_sel= pd.read_pickle("/Users/songyouk/PairsTradingAutomation/data/df_sel.pkl")[["median_window", "median_zscore_threshold"]]
stocks = [stock for pair in df_sel.index for stock in pair]


if len(df_sel) > 7:
    df_sel = df_sel.iloc[0:7,:]


positions = ib.positions()
df_current_positions = pd.DataFrame(positions)
stocks_open = []
if len(df_current_positions) == 0:
    print("No open positions")
else:
    df_current_positions['symbol'] = df_current_positions['contract'].apply(lambda x: x.symbol)
    stocks_open = df_current_positions['symbol'].to_list()
    print(stocks_open)


stocks_to_trade = [stock for pair in df_sel.index.to_list() for stock in pair]
stocks_to_close = [stock for stock in stocks_open if stock not in stocks_to_trade]

if len(df_current_positions) > 0:
    for index, row in df_current_positions.iterrows():
        stock = row['contract'].symbol
        position = row['position']
        if stock in stocks_to_close:
            print(f"Closing position for {stock}")
            contract = Stock(stock, 'SMART', 'USD')
            ib.qualifyContracts(contract)
            action = 'SELL' if position > 0 else 'BUY'
            order = MarketOrder(action, abs(position))
            trade = ib.placeOrder(contract, order)



def get_positions_summary(stocks_to_trade, df_pairs_wt_paras, df_whole_intraday, df_whole, ib, margin_ratio):


    accountSummary = pd.DataFrame(ib.accountSummary())
    NetLiquidation = float(accountSummary[accountSummary["tag"] == "NetLiquidation"]["value"].iloc[0])
    BuyingPower = float(accountSummary[accountSummary["tag"] == "BuyingPower"]["value"].iloc[0])
    
    
    margin_init = NetLiquidation/len(df_pairs_wt_paras) 
    ls_current_future_positions= [PairTradingUpdatePosition(df_whole_intraday = df_whole_intraday, 
                                                    df_whole = df_whole, 
                                                    margin_init = margin_init, 
                                                    margin_ratio = margin_ratio, 
                                                    df_pairs_wt_paras = df_pairs_wt_paras, 
                                                    ib = ib,
                                                    pair = pair).update_position_summary() for pair in df_sel.index]

    df_current_future_positions = pd.DataFrame(ls_current_future_positions)

    df_current_future_positions["stock1_order_numbers"] = df_current_future_positions["stock1_future_numbers"] - df_current_future_positions["stock1_current_numbers"]
    df_current_future_positions["stock2_order_numbers"] = df_current_future_positions["stock2_future_numbers"] - df_current_future_positions["stock2_current_numbers"]

    return df_current_future_positions



while True:
    if (len(stocks)>=1):
        break


new_york_tz = pytz.timezone('America/New_York')
dummy_trading_time = datetime.datetime.now(new_york_tz).replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=7)

df_trading_time = df_sel.copy()
df_trading_time['last_trading_time'] = dummy_trading_time

def ib_order_execute(df_current_future_positions, df_trading_time):
    for index, row in df_current_future_positions.iterrows():
        stock1 = row['pair'][0]
        stock2 = row['pair'][1]
        stock1_order_numbers = row['stock1_order_numbers']
        stock2_order_numbers = row['stock2_order_numbers']
        

        now = datetime.datetime.now(pytz.timezone('America/New_York'))
        start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        last_trading_time = df_trading_time.loc[df_trading_time.index == (stock1, stock2), 'last_trading_time'].iloc[-1]
        if start_time <= now <= end_time:
            # Check if the last trading time is on the same day as the current day and time difference is at least 10 minutes
            if ((last_trading_time.date() == now.date()) and (now - last_trading_time) >= datetime.timedelta(minutes=60)) or (last_trading_time.date() != now.date()):
                
                
                if (stock1_order_numbers > 5) and (stock2_order_numbers < -5):
                    print(f"Buying {stock1_order_numbers} of {stock1}")
                    print(f"Selling {stock2_order_numbers} of {stock2}")  
                    contract_stock1 = Stock(stock1, 'SMART', 'USD')
                    ib.qualifyContracts(contract_stock1)
                    order_stock1 = MarketOrder('BUY', abs(stock1_order_numbers))
                    trade_stock1 = ib.placeOrder(contract_stock1, order_stock1)                
  
                    contract_stock2 = Stock(stock2, 'SMART', 'USD')
                    ib.qualifyContracts(contract_stock2)
                    order_stock2 = MarketOrder('SELL', abs(stock2_order_numbers))
                    trade_stock2 = ib.placeOrder(contract_stock2, order_stock2)
                    df_trading_time.loc[df_trading_time.index == (stock1, stock2), 'last_trading_time'] = datetime.datetime.now(pytz.timezone('America/New_York'))
                    
                elif (stock1_order_numbers < -5) and (stock2_order_numbers > 5):
                    print(f"Selling {stock1_order_numbers} of {stock1}")
                    print(f"Buying {stock2_order_numbers} of {stock2}")  
                    contract_stock1 = Stock(stock1, 'SMART', 'USD')
                    ib.qualifyContracts(contract_stock1)
                    order_stock1 = MarketOrder('SELL', abs(stock1_order_numbers))
                    trade_stock1 = ib.placeOrder(contract_stock1, order_stock1)                
  
                    contract_stock2 = Stock(stock2, 'SMART', 'USD')
                    ib.qualifyContracts(contract_stock2)
                    order_stock2 = MarketOrder('BUY', abs(stock2_order_numbers))
                    trade_stock2 = ib.placeOrder(contract_stock2, order_stock2)
                    df_trading_time.loc[df_trading_time.index == (stock1, stock2), 'last_trading_time'] = datetime.datetime.now(pytz.timezone('America/New_York'))


import pandas_market_calendars as mcal

while True:
    now = datetime.datetime.now(pytz.timezone('America/New_York'))
    schedule = mcal.get_calendar('NYSE').schedule(start_date=now, end_date=now)
    if len(schedule) == 0 or now.hour > 16:
        break
    stocks = [stock for pair in df_sel.index for stock in pair]
    data_1d_1y = yf.download(tickers = stocks, period="1y",interval="1d", progress = False)['Adj Close']
    data_5m_60d = yf.download(tickers = stocks, period="60d",interval="5m", progress = False)['Adj Close']
    df_current_future_positions = get_positions_summary(stocks_to_trade = stocks_to_trade, df_pairs_wt_paras = df_sel, df_whole_intraday = data_5m_60d, df_whole = data_1d_1y, ib = ib, 
                                                        margin_ratio = 0.25)
    ib_order_execute(df_current_future_positions = df_current_future_positions, df_trading_time = df_trading_time)
    ib.sleep(60)
    if now.hour > 16:
        break
