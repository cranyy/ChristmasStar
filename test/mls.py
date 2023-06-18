# mls.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras
from hyperfeatey import feature_selection, score_hyperparams, custom_cv_splits
import numpy as np
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

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

# Existing imports...

from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import os

# Existing functions...

def lstm_model(stock_data, retrain=False):
    model_file = 'lstm_model.h5'
    
    if os.path.isfile(model_file) and not retrain:
        model = keras.models.load_model(model_file)
        print(f"Loaded LSTM model from {model_file}")
    
    stock_data = stock_data.dropna()
    selected_features = feature_selection(stock_data, method="SelectKBest", n_features=7)
    X = stock_data[selected_features]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])

    # Save the model
    model.save(model_file)

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Rolling window prediction for the next 365 days
    future_prices_all_days = []
    current_input = X_test[-1].reshape((1, 1, X_test.shape[2]))
    for _ in range(365):
        future_price = model.predict(current_input)[0][0]
        future_prices_all_days.append(future_price)
        current_input = np.append(current_input[0][0][1:], future_price).reshape((1, 1, X_test.shape[2]))

    # Create future_prices for 6 months and 1 year
    future_prices_6m = future_prices_all_days[:180]
    future_prices_1y = future_prices_all_days

    # Create future_prices for the next day, 6 months and 1 year
    future_prices = [future_prices_all_days[0], future_prices_6m[-1], future_prices_1y[-1]]

    return mse, r2, future_prices, future_prices_all_days, y_test, y_pred, future_prices_6m, future_prices_1y





def random_forest_model(stock_data, symbol, retrain=False):
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