import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
import platform
import logging
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import ta
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
import joblib
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import os
from sklearn.metrics import mean_squared_error, r2_score
import random
import json
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ParameterGrid
from ta.trend import PSARIndicator, AroonIndicator, DPOIndicator
from ta.momentum import PercentagePriceOscillator
from ta.volume import MFIIndicator
import stoceky


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
def feature_selection(stock_data, method="SelectKBest", n_features=7):
    X = stock_data.drop("Close", axis=1)
    y = stock_data["Close"]

    if method == "RFE":
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X, y)
        selected_columns = X.columns[selector.support_]

    elif method == "SelectKBest":
        selector = SelectKBest(score_func=f_regression, k=n_features)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]

    elif method == "Lasso":
        lasso = Lasso(alpha=0.1)
        lasso.fit(X, y)
        selected_columns = X.columns[lasso.coef_ != 0]

    return selected_columns

def data_has_changed(stock_data_old, stock_data_new):
    return not stock_data_old.equals(stock_data_new)

from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

def score_hyperparams(model, X, y, params, cv_splits):
    model.set_params(**params)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv_splits)
    return params, scores.mean()

class TqdmSplits(KFold):
    def __init__(self, n_splits=5, *args, **kwargs):
        super().__init__(n_splits, *args, **kwargs)

    def split(self, X, y=None, groups=None):
        iterator = super().split(X, y, groups)
        return tqdm(iterator, total=self.n_splits)

def custom_cv_splits(n_splits=3):
    return TqdmSplits(n_splits=n_splits)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import keras

def neural_network_model(stock_data, retrain=False):
    model_file = 'neural_network_model.h5'
    
    if os.path.isfile(model_file) and not retrain:
        model = keras.models.load_model(model_file)
        print(f"Loaded neural network model from {model_file}")
    
    stock_data = stock_data.dropna()
    selected_features = feature_selection(stock_data, method="SelectKBest", n_features=7)
    X = stock_data[selected_features]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the neural network model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=2)

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    last_30_days = stock_data[X.columns].tail(30)
    last_30_days_scaled = scaler.transform(last_30_days)
    future_prices = model.predict(last_30_days_scaled).flatten()

    model.save(model_file)

    return mse, r2, future_prices, y_test, y_pred

def random_forest_model(stock_data, retrain=False):
    model_file = 'random_forest_model.pkl'
    
    if os.path.isfile(model_file) and not retrain:
        model = joblib.load(model_file)
        print(f"Loaded random forest model from {model_file}")
    
    stock_data = stock_data.dropna()
    selected_features = feature_selection(stock_data, method="SelectKBest", n_features=7)
    X = stock_data[selected_features]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=42)
    rf_params = {'n_estimators': [50, 100],
                 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(estimator=model, param_grid=rf_params, scoring='neg_mean_squared_error', cv=custom_cv_splits(5), n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"{symbol}: GridSearchCV completed for random forest model")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    last_30_days = stock_data[X.columns].tail(30)
    last_30_days_scaled = scaler.transform(last_30_days)
    future_prices = best_model.predict(last_30_days_scaled)

    joblib.dump(best_model, model_file)

    return mse, r2, future_prices, y_test, y_pred


def linear_regression_model(stock_data, retrain=False):
    model_file = 'linear_regression_model.pkl'
    if os.path.isfile(model_file) and not retrain:
        model = joblib.load(model_file)
        print(f"Loaded linear regression model from {model_file}")
    stock_data = stock_data.dropna()
    selected_features = feature_selection(stock_data, method="SelectKBest", n_features=10)
    X = stock_data[selected_features]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = Lasso(max_iter=10000)  # Increase the number of iterations
    lasso_params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}
    pipeline = Pipeline([('model', model)])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=lasso_params, scoring='neg_mean_squared_error', cv=custom_cv_splits(5))
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    last_30_days = stock_data[X.columns].tail(30)
    last_30_days_scaled = scaler.transform(last_30_days)
    future_prices = best_model.predict(last_30_days_scaled)
    joblib.dump(best_model, model_file)
    return mse, r2, future_prices, y_test, y_pred


    
    

    
def get_action(current_price, future_price, threshold=0.03):
    if future_price > current_price * (1 + threshold):
        return 'BUY'
    elif future_price < current_price * (1 - threshold):
        return 'SELL'
    else:
        return 'UNKNOWN'

import time

import plotly.graph_objs as go
import plotly.io as pio

def save_plot(stock_data, y_test, y_pred, rf_y_pred, symbol, model_type, tomorrow_price, future_prices_7d, future_prices_1m):
    # Create a Scatter trace for actual prices
    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    actual_prices_trace = go.Scatter(
        x=stock_data.index,
        y=stock_data["Close"],
        mode='lines',
        name='Actual prices'
    )

    # Create a Scatter trace for predicted prices
    predicted_prices_trace = go.Scatter(
        x=y_test.index,
        y=y_pred,
        mode='markers',
        name='Predicted prices'
    )

    # Create a Scatter trace for tomorrow predicted price
    tomorrow_predicted_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=1)],
        y=[tomorrow_price],
        mode='markers',
        name='Tomorrow predicted price',
        text=['Tomorrow predicted price']
    )

    # Create a Scatter trace for 7d predicted price
    future_prices_7d_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=7)],
        y=[future_prices_7d],
        mode='markers',
        name='7d predicted price',
        text=['7d predicted price']
    )

    # Create a Scatter trace for 30d predicted price
    future_prices_1m_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=30)],
        y=[future_prices_1m],
        mode='markers',
        name='30d predicted price',
        text=['30d predicted price']
    )
    rf_predicted_prices_trace = go.Scatter(
        x=y_test.index,
        y=rf_y_pred,
        mode='lines+markers',
        name='Random Forest Predicted prices',
        marker=dict(size=6),
        line=dict(width=1)
)
    # Create a DataFrame for the random forest predicted prices
    rf_predicted_prices_df = pd.DataFrame({'Date': y_test.index, 'Price': rf_y_pred})
    rf_predicted_prices_df.set_index('Date', inplace=True)
    # Interpolate the random forest predicted prices
    rf_predicted_prices_interpolated = rf_predicted_prices_df.reindex(stock_data.index).interpolate(method='time')

    # Create a Scatter trace for random forest predicted prices
    rf_predicted_prices_trace = go.Scatter(
        x=rf_predicted_prices_interpolated.index,
        y=rf_predicted_prices_interpolated['Price'],
        mode='lines+markers',
        name='Random Forest Predicted prices'
    
    
    )
    # Create a Scatter trace for RF tomorrow predicted price
    rf_tomorrow_predicted_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=1)],
        y=[rf_future_prices[-2]],
        mode='markers',
        name='RF Tomorrow predicted price',
        text=['RF Tomorrow predicted price']
    )
    
    rf_future_prices_7d_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=7)],
        y=[rf_future_prices],
        mode='markers',
        name='RF 7d predicted price',
        text=['RF 7d predicted price']
)

    # Create a Scatter trace for RF 30d predicted price
    rf_future_prices_1m_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=30)],
        y=[rf_future_prices[-30]],
        mode='markers',
        name='RF 30d predicted price',
        text=['RF 30d predicted price']
        )
    nn_predicted_prices_trace = go.Scatter(
        x=nn_y_test.index,
        y=nn_y_pred,
        mode='markers',
        name='Neural Network Predicted prices'
    )

    nn_tomorrow_predicted_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=1)],
        y=[nn_future_prices[-2]],
        mode='markers',
        name='NN Tomorrow predicted price',
        text=['NN Tomorrow predicted price']
    )

    nn_future_prices_7d_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=7)],
        y=[nn_future_prices[-7]],
        mode='markers',
        name='NN 7d predicted price',
        text=['NN 7d predicted price']
    )

    nn_future_prices_1m_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=30)],
        y=[nn_future_prices[-30]],
        mode='markers',
        name='NN 30d predicted price',
        text=['NN 30d predicted price']
    )
    # Create a layout for the plot
    layout = go.Layout(
        title=f"Compare predicted prices to actual prices for {symbol} ({model_type})",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price')
    )

    # Create a Figure object
    # Create a Figure object with the Random Forest model's predictions
    fig = go.Figure(data=[actual_prices_trace, predicted_prices_trace, tomorrow_predicted_trace, future_prices_7d_trace, future_prices_1m_trace, rf_predicted_prices_trace, rf_tomorrow_predicted_trace, rf_future_prices_7d_trace, rf_future_prices_1m_trace, nn_predicted_prices_trace, nn_tomorrow_predicted_trace, nn_future_prices_7d_trace, nn_future_prices_1m_trace], layout=layout)

    # Save the figure as a static image
    pio.write_image(fig, f'Figures/{symbol}_{model_type}_fig.png')
    pyo.plot(fig, filename=f'Figures/{symbol}_{model_type}_fig.html', auto_open=False)

def generate_timestamped_filename(prefix, extension):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def merge_dataframes(stock_data_yf, stock_data_av):
    # Convert the timezone of the Yahoo Finance data to a timezone-naive index
    stock_data_yf.index = stock_data_yf.index.tz_localize(None)

    # Convert the timezone of the Alpha Vantage data to a timezone-naive index
    stock_data_av.index = stock_data_av.index.tz_localize(None)

    # Merge the two DataFrames and remove duplicates
    merged_data = pd.concat([stock_data_yf, stock_data_av], axis=1)
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    return merged_data

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
    
    # Iterate through S&P 500 stocks and predict their prices
    for symbol in tqdm(sp500_list):
        print(f"Processing {symbol}...")
        predictions_df = None
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
            nn_mse, nn_r2, nn_future_prices, nn_y_test, nn_y_pred = neural_network_model(stock_data)
            print(f"{symbol}: Trained and evaluated neural network model")
            print(f"{symbol}: Neural Network MSE: {nn_mse}")

            # Train and evaluate the random forest model
            rf_mse, rf_r2, rf_future_prices, rf_y_test, rf_y_pred = random_forest_model(stock_data)
            print(f"{symbol}: Trained and evaluated random forest model")
            print(f"{symbol}: Random Forest MSE: {rf_mse}")
            print()

            # Train and evaluate the linear regression model
            lr_mse, lr_r2, lr_future_prices, y_test, y_pred = linear_regression_model(stock_data)
            print(f"{symbol}: Trained and evaluated linear regression model")
            print(f"{symbol}: Linear Regression MSE: {lr_mse}")

            best_model = 'LR' if lr_mse < rf_mse else 'RF'
            print(f"{symbol}: Best model: {best_model}")
            print()

            current_price = stock_data_yf.tail(1)['Close'].iloc[0]
            print(f"{symbol}: Current price: {current_price}")
            print()

            action_1d = get_action(current_price, lr_future_prices[-1])
            action_3p = get_action(current_price, lr_future_prices[-1], threshold=0.03)
            action_5p = get_action(current_price, lr_future_prices[-1], threshold=0.05)
            action_10p = get_action(current_price, lr_future_prices[-1], threshold=0.1)

            # Predict tomorrow's close price
            tomorrow_price = lr_future_prices[-2] if dt.datetime.now().hour >= 16 else lr_future_prices[-1]
            print(f"{symbol}: Tomorrow's price: {tomorrow_price}")
            print()
            
            future_prices_7d = lr_future_prices[-7]
            future_prices_1m = lr_future_prices[-30]

            print(f"{symbol}: Added predictions to DataFrame")
            if predictions_df is not None:
                predictions_df.loc[0, 'RF_MSE'] = rf_mse
                predictions_df.loc[0, 'RF_r2'] = rf_r2
                predictions_df.loc[0, 'Tomorrow_Close_Price_Prediction_RF'] = rf_future_prices[-1]
                predictions_df.loc[0, '7d_Close_Price_Prediction_RF'] = rf_future_prices[-7]
                predictions_df.loc[0, '1m_Close_Price_Prediction_RF'] = rf_future_prices[-30]
                predictions_df.loc[0, 'NN_MSE'] = nn_mse
                predictions_df.loc[0, 'NN_r2'] = nn_r2
                predictions_df.loc[0, 'Tomorrow_Close_Price_Prediction_NN'] = nn_future_prices[-1]
                predictions_df.loc[0, '7d_Close_Price_Prediction_NN'] = nn_future_prices[-7]
                predictions_df.loc[0, '1m_Close_Price_Prediction_NN'] = nn_future_prices[-30]
            
            # Get today's date and tomorrow's date

            today_date_date = stock_data.index[-1].strftime("%Y-%m-%d")
            tomorrow_date = (stock_data.index[-1] + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

            save_plot(stock_data, y_test, y_pred, rf_y_pred, symbol, 'Linear Regression and Random Forest', tomorrow_price, future_prices_7d, future_prices_1m)           
            print(f"{symbol}: Saved Plot")

            # Create a new DataFrame for the stock's predictions
            predictions_df = pd.DataFrame({'Symbol': [symbol],
                                            'LR_MSE': [lr_mse],
                                            'RF_MSE': [rf_mse],
                                            'rf_r2': [rf_r2],
                                            'LR_r2': [lr_r2],
                                            'Today': [today_date_date],
                                            'Today_Close_Price':[current_price],
                                            'Tomorrow': [tomorrow_date],
                                            'Tomorrow_Close_Price_Prediction': [lr_future_prices[-2] if best_model == 'LR' else rf_future_prices[-2]],
                                            '7d_Close_Price_Prediction': [lr_future_prices[-7] if best_model == 'LR' else rf_future_prices[-7]],
                                            '1m_Close_Price_Prediction': [lr_future_prices[-30] if best_model == 'LR' else rf_future_prices[-30]],
                                            '1d_Action_3p': [action_3p],
                                            '1d_Action_5p': [action_5p],
                                            '1d_Action_10p': [action_10p],
                                            'Tomorrow_Close_Price_Prediction_RF': [rf_future_prices[-1]],
                                            '7d_Close_Price_Prediction_RF': [rf_future_prices[-7]],
                                            '1m_Close_Price_Prediction_RF': [rf_future_prices[-30]],

                                            
                                            '1d_Action': [action_1d]})

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
    csv_filename = generate_timestamped_filename("predictions", "csv")
    combined_predictions_df.to_csv(csv_filename, index=False)
    print(f"Saved combined predictions to {csv_filename}")