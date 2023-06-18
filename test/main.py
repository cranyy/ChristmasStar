import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import multiprocessing as mp
import platform
import logging
from tqdm import tqdm
import os
import json
import backtrader as bt
import plotly.graph_objects as go
from mls import neural_network_model, random_forest_model, linear_regression_model, lstm_model
import stoceky
import plotey




# Read config file
with open('config.json', 'r') as f:
    config = json.load(f)

# Use parameters from config in your functions
start_date = dt.datetime.strptime(config["start_date"], "%Y-%m-%d")
end_date = dt.datetime.strptime(config["end_date"], "%Y-%m-%d")
num_symbols = config["num_symbols"]

# For Linear Regression
lr_model_file = config["linear_regression"]["model_file"]

api_key = "XG8IFQJ9QVLQ5HE7"


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import keras


import backtrader as bt
import numpy as np

class PredictiveStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.bar_executed = 0
        self.orderid = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED for order #{self.orderid}, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed = len(self)
            else:
                self.log(f'SELL EXECUTED for order #{self.orderid}, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}, Profit: {order.executed.price - self.buyprice}')
                self.orderid += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        self.log(f'Close, {self.dataclose[0]}')
        if self.order:
            return

        if not self.position:
            if self.dataclose[0] < self.dataclose[-1]:
                if self.dataclose[-1] < self.dataclose[-2]:
                    self.log(f'BUY CREATE for order #{self.orderid}, {self.dataclose[0]}')
                    self.order = self.buy()
        else:
            if len(self) >= (self.bar_executed + 5):
                self.log(f'SELL CREATE for order #{self.orderid}, {self.dataclose[0]}')
                self.order = self.sell()






if __name__ == '__main__':
    if platform.system() == 'Windows':
        mp.set_start_method('spawn')
        mp.freeze_support()
    # Define logger
    logger = mp.log_to_stderr(logging.INFO)
    # Define global variable for S&P 500 list
    sp500_list = []
    # 1. Data Collection
    # Get the list of S&P 500 stocks
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    sp500_list = table[0]['Symbol'].tolist()[:num_symbols]

    # Add SPX, DOW, and Tech 100 tickers to the beginning of the list
    sp500_list.insert(0, 'SPY') 
    sp500_list.insert(1, 'DIA')  # Dow Jones Industrial Average (DOW) alternative
    sp500_list.insert(2, 'QQQ')  # NASDAQ 100 (Tech 100) alternative

    # Define date range for historical data
    start_date = dt.datetime(1950, 1, 1)
    end_date = dt.datetime.now()

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Symbol', 'LR_MSE', 'RF_MSE', 'NN_MSE', 'LR_r2', 'RF_r2', 'NN_r2', 'Today', 'Today_Close_Price', 'Tomorrow', 'Tomorrow_Close_Price_Prediction', '1d_Action', '1d_Action_3p', '1d_Action_5p', '1d_Action_10p', 'Tomorrow'])

    all_predictions_dfs = []
    predictions_df = pd.DataFrame(columns=[...])

    # Iterate through S&P 500 stocks and predict their prices
    for symbol in tqdm(sp500_list):
        print(f"Processing {symbol}...")
        predictions_df = pd.DataFrame(columns=[...])

        
        try:
            # Get the stock data from Yahoo Finance
            stock_data_yf = stoceky.get_stock_data(symbol, start_date, end_date)
            print(f"{symbol}: Got stock data from Yahoo Finance")

            # Get the stock data from Alpha Vantage
            stock_data_av = stoceky.get_alpha_vantage_time_series_data(symbol, api_key)
            print(f"{symbol}: {len(stock_data_av)} data points from Alpha Vantage")

            # Merge the two DataFrames and remove duplicates
            stock_data = stoceky.merge_dataframes(stock_data_yf, stock_data_av)
            print(f"{symbol}: Merged stock data from Yahoo Finance and Alpha Vantage")
            
            

            # Add features
            stock_data = stoceky.add_features(symbol, stock_data)
            print(f"{symbol}: Added features to stock data")

            # Train and evaluate the neural network model
            nn_mse, nn_r2, nn_future_prices, nn_future_prices_all_days, nn_y_test, nn_y_pred, nn_future_prices_6m, nn_future_prices_1y = neural_network_model(stock_data)
            print(f"{symbol}: Trained and evaluated neural network model")
            print(f"{symbol}: Neural Network MSE: {nn_mse}")
            print()

            
            # Train and evaluate the random forest model
            rf_mse, rf_r2, rf_future_prices, rf_y_test, rf_y_pred, rf_future_prices_6m, rf_future_prices_1y = random_forest_model(stock_data, symbol)
            print(f"{symbol}: Trained and evaluated random forest model")
            print(f"{symbol}: Random Forest MSE: {rf_mse}")
            print()

            # Train and evaluate the linear regression model
            lr_mse, lr_r2, lr_future_prices, y_test, y_pred, lr_future_prices_6m, lr_future_prices_1y = linear_regression_model(stock_data)
            print(f"{symbol}: Trained and evaluated linear regression model")
            print(f"{symbol}: Linear Regression MSE: {lr_mse}")

            # Train and evaluate the LSTM model
            lstm_mse, lstm_r2, lstm_future_prices, lstm_future_prices_all_days, lstm_y_test, lstm_y_pred, lstm_future_prices_6m, lstm_future_prices_1y = lstm_model(stock_data)
            print(f"{symbol}: Trained and evaluated LSTM model")
            print(f"{symbol}: LSTM MSE: {lstm_mse}")
            print()

            # Determine the best model based on the lowest MSE
            mse_scores = {'LR': lr_mse, 'RF': rf_mse, 'NN': nn_mse, 'LSTM': lstm_mse}
            best_model = min(mse_scores, key=mse_scores.get)
            print(f"{symbol}: Best model: {best_model}")

            current_price = stock_data_yf.tail(1)['Close'].iloc[0]
            print(f"{symbol}: Current price: {current_price}")
            print()

            
            # Predict tomorrow's close price
            tomorrow_price = lr_future_prices[-2] if best_model == 'LR' else (rf_future_prices[-2] if best_model == 'RF' else nn_future_prices[-2])
            print(f"{symbol}: Tomorrow's price: {tomorrow_price}")
            print()
            
            future_prices_7d = lr_future_prices[-7] if best_model == 'LR' else (rf_future_prices[-7] if best_model == 'RF' else nn_future_prices[-7])
            future_prices_1m = lr_future_prices[-30] if best_model == 'LR' else (rf_future_prices[-30] if best_model == 'RF' else nn_future_prices[-30])

            print(f"{symbol}: Added predictions to DataFrame")
            if predictions_df is not None:
                predictions_df.loc[0, 'RF_MSE'] = rf_mse
                predictions_df.loc[0, 'RF_r2'] = rf_r2
                predictions_df.loc[0, 'Tomorrow_Close_Price_Prediction_RF'] = rf_future_prices[-2]
                predictions_df.loc[0, '7d_Close_Price_Prediction_RF'] = rf_future_prices[-7]
                predictions_df.loc[0, '1m_Close_Price_Prediction_RF'] = rf_future_prices[-30]
                predictions_df.loc[0, 'NN_MSE'] = nn_mse
                predictions_df.loc[0, 'NN_r2'] = nn_r2
                predictions_df.loc[0, 'Tomorrow_Close_Price_Prediction_NN'] = nn_future_prices[-2]
                predictions_df.loc[0, '7d_Close_Price_Prediction_NN'] = nn_future_prices[-7]
                predictions_df.loc[0, '1m_Close_Price_Prediction_NN'] = nn_future_prices[-30]
                # Update the predictions DataFrame with the best model's predictions
                predictions_df.loc[0, 'Best_Model'] = best_model
                predictions_df.loc[0, 'Tomorrow_Close_Price_Prediction_Best_Model'] = lr_future_prices[-2] if best_model == 'LR' else (rf_future_prices[-2] if best_model == 'RF' else nn_future_prices[-2])
                predictions_df.loc[0, '7d_Close_Price_Prediction_Best_Model'] = lr_future_prices[-7] if best_model == 'LR' else (rf_future_prices[-7] if best_model == 'RF' else nn_future_prices[-7])
                predictions_df.loc[0, '1m_Close_Price_Prediction_Best_Model'] = lr_future_prices[-30] if best_model == 'LR' else (rf_future_prices[-30] if best_model == 'RF' else nn_future_prices[-30])

                predictions_df.loc[0, '6m_Close_Price_Prediction_Best_Model'] = lr_future_prices_6m[-1] if best_model == 'LR' else (rf_future_prices_6m[-1] if best_model == 'RF' else nn_future_prices_6m[-1])
                predictions_df.loc[0, '1y_Close_Price_Prediction_Best_Model'] = lr_future_prices_1y[-1] if best_model == 'LR' else (rf_future_prices_1y[-1] if best_model == 'RF' else nn_future_prices_1y[-1])
                
                    # Create a cerebro entity
                # Create a cerebro entity
                cerebro = bt.Cerebro()

                # Add a strategy
                cerebro.addstrategy(PredictiveStrategy)

                # Load data
                data = bt.feeds.PandasData(dataname=stock_data)
                cerebro.adddata(data)

                # Set our desired cash start
                cerebro.broker.setcash(1000.0)

                # Print out the starting conditions
                print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

                # Run over everything
                cerebro.run()

                # Print out the final result
                print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
            
            # Get today's date and tomorrow's date

            today_date_date = stock_data.index[-1].strftime("%Y-%m-%d")
            tomorrow_date = (stock_data.index[-1] + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
            
            #plotey.save_plot(stock_data, y_test, y_pred, rf_y_pred, symbol, 'Linear Regression and Random Forest', tomorrow_price, future_prices_7d, future_prices_1m, rf_future_prices, nn_future_prices, nn_y_test, nn_y_pred, rf_future_prices_6m, rf_future_prices_1y, nn_future_prices_6m, nn_future_prices_1y, nn_future_prices_all_days)

            # Create a new DataFrame for the stock's predictions
            predictions_df = pd.DataFrame({
                                            'Symbol': [symbol],
                                            'LR_MSE': [lr_mse],
                                            'RF_MSE': [rf_mse],
                                            'NN_MSE': [nn_mse],
                                            'lstm_MSE': [lstm_mse],
                                            'lstm_r2': [lstm_r2],
                                            'rf_r2': [rf_r2],
                                            'LR_r2': [lr_r2],
                                            'Best_Model': [best_model],
                                            'Today': [today_date_date],
                                            'Today_Close_Price': [current_price],
                                            'Tomorrow': [tomorrow_date],
                                            'Tomorrow_Close_Price_Prediction_Best_Model': [lr_future_prices[-2] if best_model == 'LR' else (rf_future_prices[-2] if best_model == 'RF' else nn_future_prices[-2])],
                                            '7d_Close_Price_Prediction_Best_Model': [lr_future_prices[-7] if best_model == 'LR' else (rf_future_prices[-7] if best_model == 'RF' else nn_future_prices[-7])],
                                            '1m_Close_Price_Prediction_Best_Model': [lr_future_prices[-30] if best_model == 'LR' else (rf_future_prices[-30] if best_model == 'RF' else nn_future_prices[-30])],
                                            '6m_Close_Price_Prediction_Best_Model': [lr_future_prices_6m[-1] if best_model == 'LR' else (rf_future_prices_6m[-1] if best_model == 'RF' else nn_future_prices_6m[-1])],
                                            '1y_Close_Price_Prediction_Best_Model': [lr_future_prices_1y[-1] if best_model == 'LR' else (rf_future_prices_1y[-1] if best_model == 'RF' else nn_future_prices_1y[-1])],
                                            'Tomorrow_Close_Price_Prediction_RF': [rf_future_prices[-1]],
                                            '7d_Close_Price_Prediction_RF': [rf_future_prices[-7]],
                                            '1m_Close_Price_Prediction_RF': [rf_future_prices[-30]],
                                            '6m_Close_Price_Prediction_RF': [rf_future_prices_6m[-1]],
                                            '1y_Close_Price_Prediction_RF': [rf_future_prices_1y[-1]],
                                            'Tomorrow_Close_Price_Prediction_NN': [nn_future_prices[-1]],
                                            '7d_Close_Price_Prediction_NN': [nn_future_prices[-7]],
                                            '1m_Close_Price_Prediction_NN': [nn_future_prices[-30]],
                                            '6m_Close_Price_Prediction_NN': [nn_future_prices_6m[-1]],
                                            '1y_Close_Price_Prediction_NN': [nn_future_prices_1y[-1]]
                                            
                                    })
                                            



        except Exception as e:
            print(f"Error processing {symbol}: {e}")

        finally:
            tqdm.write("")  # Add empty line for readability

            if predictions_df is not None:
    # Append the stock's predictions DataFrame to the list of all predictions DataFrames
                all_predictions_dfs.append(predictions_df)

    # Concatenate all predictions DataFrames
    combined_predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)

    # Save the combined predictions DataFrame to a CSV file with a timestamp in its name
    csv_filename = plotey.generate_timestamped_filename("predictions", "csv")
    combined_predictions_df.to_csv(csv_filename, index=False)
    print(f"Saved combined predictions to {csv_filename}")


    
