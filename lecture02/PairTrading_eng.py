#!/usr/bin/env python
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import pickle
import builtins
import os
import numpy as np
from joblib import Parallel, delayed
import math


class PairTrading:

    # attributes (속성)
    def __init__(self, pair, df_whole, window, zscore_threshold, margin_init, margin_ratio):
        """
        Initialize the PairTrading object.

        Parameters:
        - pair (tuple): A tuple containing the names of the two stocks in the pair.
        - df_whole (DataFrame): The whole DataFrame containing the historical stock data.
        - window (int): The size of the rolling window used for calculating z-scores.
        - zscore_threshold (float): The threshold value for determining trading signals based on z-scores.
        - margin_init (float): The initial margin value for trading.
        - margin_ratio (float): The ratio used for adjusting the margin value.

        Attributes:
        - stock1 (str): The name of the first stock in the pair.
        - stock2 (str): The name of the second stock in the pair.
        - window (int): The size of the rolling window used for calculating z-scores.
        - zscore_threshold (float): The threshold value for determining trading signals based on z-scores.
        - margin_init (float): The initial margin value for trading.
        - margin_ratio (float): The ratio used for adjusting the margin value.
        - margin (float): The current margin value.
        - df_pair (DataFrame): A DataFrame containing the historical data of the pair of stocks.
        - df_signal_summary (DataFrame): A DataFrame for storing the trading signals.
        - df_margin (DataFrame): A DataFrame for storing the margin values.
        
        methods:
        - zscore_calculation: Calculate the z-scores for the pair of stocks.
        - signal_calculation: Calculate the trading signals based on the z-scores.
        - signal_summary: Summarize the trading signals.
        - margin_calculation: Calculate the margin values for trading.
        - trading_summary: Provide a summary of the pair trading strategy.

        """
        self.stock1, self.stock2 = pair[0], pair[1]
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.margin_init = margin_init
        self.margin_ratio = margin_ratio
        self.margin = margin_init
        self.df_pair = df_whole.loc[:, pair].copy()
        self.df_signal_summary = pd.DataFrame()
        self.df_margin = pd.DataFrame()
        
    def __repr__(self): 
        return f"""PairTradingFinancialAnalysis(pair = {self.stock1} and {self.stock2}, window = {self.window}, zscore_threshold = {self.zscore_threshold}, 
                    margin_init = {self.margin_init}, margin_ratio = {self.margin_ratio})"""
    
    # methods (메소드)
    def zscore_calculation(self):
        """
        Calculates the z-score for a given stock pair based on the moving average and moving standard deviation of their price ratio.

        Attributes:
            - self.pair (tuple): Contains the symbols of the two stocks in the pair to be analyzed.
            - self.df_pair (DataFrame): Contains the price data for the two stocks.
            - self.window (int): The size of the rolling window for which the moving average and moving standard deviation are calculated.

        Updates:
            - The method updates self.df_pair by adding new columns:
                - "ratio": The ratio of the prices of the two stocks.
                - "ma": The moving average of the price ratio.
                - "msd": The moving standard deviation of the price ratio.
                - "zscore": The z-score calculated from the price ratio, moving average, and moving standard deviation.
        """
        self.df_pair["ratio"] = self.df_pair[self.stock1] / self.df_pair[self.stock2]
        self.df_pair["ma"] = self.df_pair["ratio"].rolling(window=self.window).mean().shift(1)
        self.df_pair["msd"] = self.df_pair["ratio"].rolling(window=self.window).std().shift(1)
        self.df_pair["zscore"] = (self.df_pair["ratio"] - self.df_pair["ma"]) / self.df_pair["msd"]

    
    def signal_calculation(self):
        """
        Calculates trading signals based on the comparison between z-score and z-score threshold.

        Attributes:
            - self.df_pair (DataFrame): DataFrame obtained from zscore_calculation.
            - self.zscore_threshold (float): Threshold value used to determine trading signals.

        Updates:
            - Updates self.df_pair to include z-score (proportional to stock1/stock2), z-score threshold, and signals.
            - If z-score > z-score_threshold and z-score < 5, it indicates that stock1 is statistically overvalued compared to stock2. This suggests shorting stock1 and going long on stock2, represented by signal = -1.
            - If z-score < -z-score_threshold and z-score > -5, it indicates that stock1 is statistically undervalued compared to stock2. This suggests going long on stock1 and shorting stock2, represented by signal = 1.
            - If z-score is between -1 and 1, it indicates that neither stock1 nor stock2 is statistically significantly valued. This suggests not to buy or sell either stock, represented by signal = 0.
            - If z-score is greater than 5 or less than -5, no trading is performed. This is because such cases are too far from the statistical norm (for example, a stock price crash), and changing any decision could be risky.
            - If none of the above cases apply, it means to maintain the signal. The existing signal is filled forward using ffill(), and the remaining NaN values are filled with 0.
        """
        import numpy as np
        self.df_pair['signal'] = np.nan
        self.df_pair['signal'] = np.where((self.df_pair['zscore'] > self.zscore_threshold) & (self.df_pair['zscore'] < 5), -1, self.df_pair['signal'])
        self.df_pair['signal'] = np.where((self.df_pair['zscore'] < -self.zscore_threshold) & (self.df_pair['zscore'] > -5), 1, self.df_pair['signal'])
        self.df_pair['signal'] = np.where((self.df_pair['zscore'] > -1) & (self.df_pair['zscore'] < 1), 0, self.df_pair['signal'])
        self.df_pair['signal'] = self.df_pair['signal'].ffill()
        self.df_pair['signal'] = self.df_pair['signal'].fillna(0)                                

    def signal_summary(self):
        """
        Groups self.df_pair based on signal, calculates start and end dates, start and end prices, and creates self.df_signal_summary.

        Attributes:
            - self.df_pair (DataFrame): DataFrame obtained from signal_calculation.

        Returns:
            DataFrame: Creates self.df_signal_summary which includes start and end dates, start and end prices, and signals.
        """ 
        self.df_pair["signal_group"] = self.df_pair["signal"].diff().ne(0).cumsum() 
        self.df_pair["time"] = self.df_pair.index
        self.df_signal_summary = (self.df_pair
                           .groupby("signal_group")
                           .agg({"signal": "first", 
                                "time": "first", 
                                self.stock1: ["first"], 
                                self.stock2: ["first"]})
                            .reset_index(drop=True)
                            )
        self.df_signal_summary.columns = ["signal", "time_start","stock1_start_price", "stock2_start_price"]
        
        self.df_signal_summary["time_end"] = self.df_signal_summary["time_start"].shift(-1)
        self.df_signal_summary["stock1_final_price"] = self.df_signal_summary["stock1_start_price"].shift(-1)
        self.df_signal_summary["stock2_final_price"] = self.df_signal_summary["stock2_start_price"].shift(-1)
        
        self.df_signal_summary.loc[self.df_signal_summary.index[-1], "time_end"] = self.df_pair.index[-1]
        self.df_signal_summary.loc[self.df_signal_summary.index[-1], "stock1_final_price"] = self.df_pair[self.stock1].iloc[-1]
        self.df_signal_summary.loc[self.df_signal_summary.index[-1], "stock2_final_price"] = self.df_pair[self.stock2].iloc[-1]

        # reorder columns
        self.df_signal_summary = self.df_signal_summary[["signal", "time_start", "time_end", "stock1_start_price", "stock1_final_price", "stock2_start_price", "stock2_final_price"]]
   
    def margin_calculation(self):
        '''
        This function calculates the margin (collateral for assets and leverage) considering the "commission" and "price adjustment" when buying and selling a pair of stocks.
        https://www.interactivebrokers.com/en/pricing/commissions-stocks.php 

        Commission details:  
            - Buy commission: $0.005 per share (minimum $1, maximum 1% of transaction value) 
            - Sell commission: $0.005 per share (minimum $1, maximum 1% of transaction value) + 0.000008 of the sale value (SEC Transaction Fee) + $0.000166 per share (FINRA Trading Activity Fee)  

        Price adjustment:
            - Buy & Sell price: Conservatively, we will adjust the price by 3 pips. Typically, 1.5 pips (1 pip = 0.0001) is used.
            - Buy & Sell price: 1.0003 (buy) & 0.9997 (sell) --> Ultimately, price adjustment is for the broker's profit.

        Attributes:
            - self.margin_init (float): The initial collateral amount.
            - self.margin_rate (float): The margin ratio of the leverage account. For example, if the margin is 3000 and the margin ratio is 0.25, the total investment amount of the leverage account is 12,000.
            - self.df_signal_summary (DataFrame): DataFrame created from signal_summary. 
        
        Returns:
            - DataFrame: Creates df_margin by copying from self.df_signal_summary and updates it by calculating the margin.
        '''

        import math
        # Initial buying power and margin setup
        
        margin = self.margin_init
        buying_power = margin/ self.margin_ratio

        # Calculate margin for each stock pair
        df_margin = self.df_signal_summary.copy()
        df_margin = df_margin[df_margin['signal'].isin([1, -1])]

        for index, row in df_margin.iterrows(): # https://www.w3schools.com/python/pandas/ref_df_iterrows.asp
            # Calculate the number of units for each stock pair
            stock1_units = math.floor((0.5 * buying_power) / row["stock1_start_price"])
            stock2_units = math.floor((0.5 * buying_power) / row["stock2_start_price"])
            
            # Calculate commissions for buying and selling
            if row["signal"] == 1:
                commision_buy = min(max(stock1_units * 0.005, 1), 0.5 * buying_power * 0.01)
                commision_sell = min(max(stock2_units * 0.005, 1), 0.5 * buying_power * 0.01) + 0.000008 * 0.5 * buying_power + 0.000166 * stock2_units
                total_commission = commision_buy + commision_sell
            else:
                commision_buy = min(max(stock2_units * 0.005, 1), 0.5 * buying_power * 0.01)
                commision_sell = min(max(stock1_units * 0.005, 1), 0.5 * buying_power * 0.01) + 0.000008 * 0.5 * buying_power + 0.000166 * stock1_units
                total_commission = commision_buy + commision_sell

            # Calculate margin based on signal
            if row["signal"] == 1: # Buy stock1 and sell stock2
                margin += ((row["stock1_final_price"] * 0.9997 - row["stock1_start_price"] * 1.0003) * stock1_units - 
                           (row["stock2_final_price"] * 1.0003 - row["stock2_start_price"] * 0.9997) * stock2_units) - total_commission
            else:
                margin += ((row["stock2_final_price"] * 0.9997 - row["stock2_start_price"] * 1.0003) * stock2_units - 
                           (row["stock1_final_price"] * 1.0003 - row["stock1_start_price"] * 0.9997) * stock1_units) - total_commission

            # Update margin and buying power for each iteration
            df_margin.loc[index, "margin"] = margin
            buying_power = margin / self.margin_ratio
            self.margin = margin
            
        self.df_margin = df_margin
  

    def trading_summary(self):
        """
        Provides a summary of the pair trading strategy.

        Attributes:
            - self.df_summary (DataFrame): The DataFrame obtained from margin_calculation.

        Returns: 
        Returns a dictionary containing the following information:
            - 'pair': The pair being analyzed.
            - 'window': The number of days used to calculate the moving average.
            - 'zscore_threshold': The threshold value used to determine the trading signal.
            - 'margin': The margin after trading.
        """
        self.zscore_calculation()
        self.signal_calculation()
        self.signal_summary()
        self.margin_calculation()
        trading_result = {
            'pair': (self.stock1, self.stock2),
            'window': self.window,
            'zscore_threshold': self.zscore_threshold,
            'margin': self.margin
        }
        return trading_result

class PairTradingIntraDay(PairTrading):
    
    def __init__(self, pair, df_whole_intraday, df_whole, window, zscore_threshold, margin_init, margin_ratio):
        """
        - PairTradingFinancialAnalysis 부모 클래스로부터 속성과 메서드를 상속받아 초기화합니다.   
        - 부모 클래스의 메서드는 zscore_calculation --> signal_calculation --> signal_summary --> margin_calculation --> trading_summary입니다.  
        - 자식 클래스는 부모 클래스의 zscore_calculation 메서드를 오버라이딩합니다. 이를 다형성이라 합니다. 다형성은 동일한 이름의 메서드가 서로 다른 클래스에서 서로 다른 기능을 하는 것을 의미합니다.

        """
        super().__init__(pair, df_whole, window, zscore_threshold, margin_init, margin_ratio)
        self.df_pair_intraday = df_whole_intraday.loc[:, pair].copy()

    def zscore_calculation(self):
        """
        주어진 주식 Pair에 윈도우를 기반으로 가격 비율의 이동 평균, 이동 표준 편차를 기반으로 zscore를 계산합니다.
        
        Attributes (속성): 
            - self.pair (튜플): 분석할 주식 쌍의 주식 심볼을 포함하는 튜플입니다.
            - self.df_pair (DataFrame): 분석할 주식에 대한 주식 가격을 포함하는 DataFrame입니다.
            - self.window (정수): 이동 평균과 이동 편차을 계산하는 데 사용되는 일수 (days)입니다.

        Reseults (결과):
            - 현재 가격 비율 from IntraDay Data과 주어진 윈도우의 과거 from 1-d interval Data로부터의 이동 평균, 이동 표준 편차를 추가하여 zscore를 계산하여 self.df를 업데이트합니다.
        """
        self.df_pair_intraday["Day"] = self.df_pair_intraday.index.date
        self.df_pair_intraday["ratio_intraday"] = np.log(self.df_pair_intraday[self.stock1]/self.df_pair_intraday[self.stock2])

        temp_df = self.df_pair.copy()   # 1-d interval data                                              
        temp_df["Day"] = temp_df.index.date
        temp_df["ratio"] = np.log(temp_df[self.stock1] / temp_df[self.stock2])
        temp_df["ma"] = temp_df["ratio"].rolling(window=self.window).mean().shift(1)
        temp_df["msd"] = temp_df["ratio"].rolling(window=self.window).std().shift(1)
        
        merged_data = pd.merge(self.df_pair_intraday, temp_df, on="Day", how="left", suffixes=("", "_1d"))
        merged_data.index = self.df_pair_intraday.index
        merged_data["zscore"] = (merged_data["ratio_intraday"] - merged_data["ma"])/merged_data["msd"]
        
        self.df_pair = merged_data # update the df_pair with the merged data to 5m interval data
        
class PairTradingUpdatePosition(PairTradingIntraDay):
    # PairTradingReadtime <- PairTradingIntraDay <- PairTrading

    def __init__(self, pair, df_whole_intraday, df_whole, margin_init, margin_ratio, df_pairs_wt_paras, ib):   
        """
        Initializes an instance of the PairTrading class.
        
        Parameters:
        - pair (str): The pair of stocks being traded.
        - df_whole_intraday (DataFrame): The intraday data for the entire trading period.
        - df_whole (DataFrame): The historical data for the entire trading period.
        - margin_init (float): The initial margin for trading.
        - margin_ratio (float): The margin ratio for trading.
        - df_pairs_wt_paras (DataFrame): The optimal parameters for the pair trading strategy.
        - ib: API connection
        """
        
        window = int(df_pairs_wt_paras[df_pairs_wt_paras.index == pair]["median_window"].iloc[-1])
        zscore_threshold = df_pairs_wt_paras[df_pairs_wt_paras.index == pair]["median_zscore_threshold"].iloc[-1]
        
        super().__init__(pair, df_whole_intraday, df_whole, window, zscore_threshold, margin_init, margin_ratio)
        
        self.trading_summary()
        self.signal = int(self.df_signal_summary["signal"].iloc[-1])
        self.df_current_status = pd.DataFrame(ib.positions())
        
        self.stock1_current_numbers = 0
        self.stock2_current_numbers = 0
        self.update_current_position()
        
        self.buying_power = margin_init/margin_ratio
        self.stock1_future_numbers = 0
        self.stock2_future_numbers = 0
        self.update_future_position()
        
        
    def update_current_position(self):
        """
        Updates the current position of the stocks in the pair trading strategy.

        This method retrieves the current position of the stocks from the `df_current_status` DataFrame
        and updates the `stock1_current_numbers` and `stock2_current_numbers` attributes accordingly.
        if stock1 or stock2 is not included in the current open position which is get from the `df_current_status`, 
        the numbers `stock1_current_numbers` and `stock2_current_numbers` will be set as zero
        """
        if len(self.df_current_status) != 0:    
            self.df_current_status['symbol'] = self.df_current_status['contract'].apply(lambda x: x.symbol)
            if (self.df_current_status["symbol"] == self.stock1).any():
                self.stock1_current_numbers = self.df_current_status[self.df_current_status["symbol"] == self.stock1]["position"].iloc[-1]
            if (self.df_current_status["symbol"] == self.stock2).any():
                self.stock2_current_numbers = self.df_current_status[self.df_current_status["symbol"] == self.stock2]["position"].iloc[-1]
    
    
    def update_future_position(self):
        """
        Updates the future positions of stock1 and stock2 based on the current signal.

        If the signal is 1, it calculates the number of future contracts to buy for stock1 and the number of future contracts to sell for stock2.
        If the signal is -1, it calculates the number of future contracts to sell for stock1 and the number of future contracts to buy for stock2.
        If the signal is neither 1 nor -1, it sets the future positions of both stock1 and stock2 to 0.
        For the above calculation, `buying power` which is get from margin and margin ratio will be used

        Returns:
            None
        """
        if self.signal == 1:
            self.stock1_future_numbers = math.floor((0.5 * self.buying_power) / self.df_signal_summary.iloc[-1]['stock1_start_price'])
            self.stock2_future_numbers = -math.floor((0.5 * self.buying_power) / self.df_signal_summary.iloc[-1]['stock2_start_price'])
        elif self.signal == -1:
            self.stock1_future_numbers = -math.floor((0.5 * self.buying_power) / self.df_signal_summary.iloc[-1]['stock1_start_price'])
            self.stock2_future_numbers = math.floor((0.5 * self.buying_power) / self.df_signal_summary.iloc[-1]['stock2_start_price'])
        else:
            self.stock1_future_numbers = 0
            self.stock2_future_numbers = 0
      
    def update_position_summary(self):
        position_summary = {
            'pair': (self.stock1, self.stock2),
            'stock1_current_numbers': self.stock1_current_numbers,
            'stock2_current_numbers': self.stock2_current_numbers,
            'stock1_future_numbers': self.stock1_future_numbers,
            'stock2_future_numbers': self.stock2_future_numbers,
        }
        return position_summary