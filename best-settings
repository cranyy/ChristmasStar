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
from sklearn.linear_model import LinearRegression
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ParameterGrid
import hyperfeatey 
import stoceky
import plotey
from hyperfeatey import feature_selection, score_hyperparams, custom_cv_splits



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
    model.fit(X_train, y_train, epochs=75, batch_size=32, verbose=2)

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    last_365_days = stock_data[X.columns].tail(365)
    last_365_days_scaled = scaler.transform(last_365_days)
    future_prices_all_days = model.predict(last_365_days_scaled).flatten()

    predictions_df = pd.DataFrame(index=range(1, 366), columns=['NN_1d_Close_Price_Prediction'])
    for i in range(1, 366):
        predictions_df.loc[i, 'NN_1d_Close_Price_Prediction'] = future_prices_all_days[-i]

    print("Neural Network predictions for every single day in the 365-day window:")
    for i in range(1, 366):
        print(f"{i} day(s) ahead: {future_prices_all_days[-i]}")

    last_180_days = stock_data[X.columns].tail(180)
    last_180_days_scaled = scaler.transform(last_180_days)
    future_prices_6m = model.predict(last_180_days_scaled).flatten()

    last_365_days = stock_data[X.columns].tail(365)
    last_365_days_scaled = scaler.transform(last_365_days)
    future_prices_1y = model.predict(last_365_days_scaled).flatten()

    last_30_days = stock_data[X.columns].tail(30)
    last_30_days_scaled = scaler.transform(last_30_days)
    future_prices = model.predict(last_30_days_scaled).flatten()

    model.save(model_file)

    return mse, r2, future_prices, future_prices_all_days, y_test, y_pred, future_prices_6m, future_prices_1y



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
    last_180_days = stock_data[X.columns].tail(180)
    last_180_days_scaled = scaler.transform(last_180_days)
    future_prices_6m = best_model.predict(last_180_days_scaled)

    last_365_days = stock_data[X.columns].tail(365)
    last_365_days_scaled = scaler.transform(last_365_days)
    future_prices_1y = best_model.predict(last_365_days_scaled)

    joblib.dump(best_model, model_file)

    return mse, r2, future_prices, y_test, y_pred, future_prices_6m, future_prices_1y


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
    model.fit(X_train, y_train)
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
    future_prices = model.predict(last_30_days_scaled).flatten()

    last_180_days = stock_data[X.columns].tail(180)
    last_180_days_scaled = scaler.transform(last_180_days)
    future_prices_6m = model.predict(last_180_days_scaled).flatten()

    last_365_days = stock_data[X.columns].tail(365)
    last_365_days_scaled = scaler.transform(last_365_days)
    future_prices_1y = model.predict(last_365_days_scaled).flatten()

    joblib.dump(best_model, model_file)

    return mse, r2, future_prices, y_test, y_pred, future_prices_6m, future_prices_1y
    


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

            
            # Train and evaluate the random forest model
            rf_mse, rf_r2, rf_future_prices, rf_y_test, rf_y_pred, rf_future_prices_6m, rf_future_prices_1y = random_forest_model(stock_data)
            print(f"{symbol}: Trained and evaluated random forest model")
            print(f"{symbol}: Random Forest MSE: {rf_mse}")
            print()

            # Train and evaluate the linear regression model
            lr_mse, lr_r2, lr_future_prices, y_test, y_pred, lr_future_prices_6m, lr_future_prices_1y = linear_regression_model(stock_data)
            print(f"{symbol}: Trained and evaluated linear regression model")
            print(f"{symbol}: Linear Regression MSE: {lr_mse}")

            # Determine the best model based on the lowest MSE
            mse_scores = {'LR': lr_mse, 'RF': rf_mse, 'NN': nn_mse}
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
                
            
            # Get today's date and tomorrow's date

            today_date_date = stock_data.index[-1].strftime("%Y-%m-%d")
            tomorrow_date = (stock_data.index[-1] + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
            
            plotey.save_plot(stock_data, y_test, y_pred, rf_y_pred, symbol, 'Linear Regression and Random Forest', tomorrow_price, future_prices_7d, future_prices_1m, rf_future_prices, nn_future_prices, nn_y_test, nn_y_pred, rf_future_prices_6m, rf_future_prices_1y, nn_future_prices_6m, nn_future_prices_1y, nn_future_prices_all_days)

            # Create a new DataFrame for the stock's predictions
            predictions_df = pd.DataFrame({
                                            'Symbol': [symbol],
                                            'LR_MSE': [lr_mse],
                                            'RF_MSE': [rf_mse],
                                            'NN_MSE': [nn_mse],
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
