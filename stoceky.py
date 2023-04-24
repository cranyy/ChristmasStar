import pandas as pd
import ta
from ta.trend import PSARIndicator, AroonIndicator, DPOIndicator
from ta.momentum import PercentagePriceOscillator
from ta.volume import MFIIndicator
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData


api_key = "XG8IFQJ9QVLQ5HE7"



def add_RSI(stock_data, period=14):
    rsi = ta.momentum.RSIIndicator(close=stock_data["Close"], window=period)
    stock_data["RSI"] = rsi.rsi()

def add_MACD(stock_data, short_period=12, long_period=26, signal_period=9):
    macd = ta.trend.MACD(close=stock_data["Close"], window_slow=long_period, window_fast=short_period, window_sign=signal_period)
    stock_data["MACD"] = macd.macd()
    stock_data["MACD_Signal"] = macd.macd_signal()

def add_Bollinger_Bands(stock_data, period=20, std_dev=2):
    bollinger = ta.volatility.BollingerBands(close=stock_data["Close"], window=period, window_dev=std_dev)
    stock_data["Bollinger_High"] = bollinger.bollinger_hband()
    stock_data["Bollinger_Low"] = bollinger.bollinger_lband()

def add_Stochastic_Oscillator(stock_data, period=14, smooth_period=3):
    stoch = ta.momentum.StochasticOscillator(close=stock_data["Close"], high=stock_data["High"], low=stock_data["Low"], window=period, smooth_window=smooth_period)
    stock_data["Stoch_Oscillator"] = stoch.stoch()

def add_Chaikin_Money_Flow(stock_data, period=20):
    cmf = ta.volume.ChaikinMoneyFlowIndicator(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], volume=stock_data["Volume"], window=period)
    stock_data["Chaikin_MF"] = cmf.chaikin_money_flow()

def add_Average_True_Range(stock_data, period=14):
    atr = ta.volatility.AverageTrueRange(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=period)
    stock_data["ATR"] = atr.average_true_range()

def add_Commodity_Channel_Index(stock_data, period=20):
    cci = ta.trend.CCIIndicator(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], window=period)
    stock_data["CCI"] = cci.cci()

def add_Rate_of_Change(stock_data, period=12):
    roc = ta.momentum.ROCIndicator(close=stock_data["Close"], window=period)
    stock_data["ROC"] = roc.roc()

def add_Triple_Exponential_Moving_Average(stock_data, period=10):
    ema = ta.trend.EMAIndicator(close=stock_data["Close"], window=period)
    stock_data["TEMA"] = ema.ema_indicator()

def add_WilliamsR(stock_data, period=14):
    wr = ta.momentum.WilliamsRIndicator(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], lbp=period)
    stock_data["WilliamsR"] = wr.williams_r()

def add_Volume_Range(stock_data):
    stock_data["Volume_Range"] = stock_data["High"] - stock_data["Low"]


def add_OBV(stock_data):
    obv = ta.volume.OnBalanceVolumeIndicator(close=stock_data["Close"], volume=stock_data["Volume"])
    stock_data["OBV"] = obv.on_balance_volume()

def add_Parabolic_SAR(stock_data, step=0.02, max_step=0.2):
    psar = PSARIndicator(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], step=step, max_step=max_step)
    stock_data["Parabolic_SAR"] = psar.psar()

def add_Aroon_Oscillator(stock_data, period=25):
    aroon = AroonIndicator(close=stock_data["Close"], window=period)
    stock_data["Aroon_Oscillator"] = aroon.aroon_indicator()

def add_Detrended_Price_Oscillator(stock_data, period=20):
    dpo = DPOIndicator(close=stock_data["Close"], window=period)
    stock_data["DPO"] = dpo.dpo()

def add_Money_Flow_Index(stock_data, period=14):
    mfi = MFIIndicator(high=stock_data["High"], low=stock_data["Low"], close=stock_data["Close"], volume=stock_data["Volume"], window=period)
    stock_data["MFI"] = mfi.money_flow_index()

def add_Percentage_Price_Oscillator(stock_data, slow_period=26, fast_period=12, signal_period=9):
    ppo = PercentagePriceOscillator(close=stock_data["Close"], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
    stock_data["PPO"] = ppo.ppo()
    stock_data["PPO_Signal"] = ppo.ppo_signal()

def add_features(symbol, stock_data):

    stock_data.index = stock_data.index.tz_localize(None)
    num_technical_indicators = 0
    # Add 7-day rolling mean
    stock_data['7_day_mean'] = stock_data['Close'].rolling(window=7).mean()
    num_technical_indicators += 1

    # Add 30-day rolling mean
    stock_data['30_day_mean'] = stock_data['Close'].rolling(window=30).mean()
    num_technical_indicators += 1

    # Add 365-day rolling mean
    stock_data['365_day_mean'] = stock_data['Close'].rolling(window=365).mean()
    num_technical_indicators += 1
    # Add new features
    stock_data['Day_of_week'] = stock_data.index.dayofweek
    num_technical_indicators += 1
    stock_data['Day_of_month'] = stock_data.index.day
    num_technical_indicators += 1
    stock_data['Month'] = stock_data.index.month
    num_technical_indicators += 1

    # Add RSI
    add_RSI(stock_data) 
    num_technical_indicators += 1

    # Add MACD
    add_MACD(stock_data)
    num_technical_indicators += 1
    # Add Bollinger Bands
    add_Bollinger_Bands(stock_data)
    num_technical_indicators += 1
    add_Stochastic_Oscillator(stock_data)
    num_technical_indicators += 1
    add_Chaikin_Money_Flow(stock_data)
    num_technical_indicators += 1
    add_Average_True_Range(stock_data)
    num_technical_indicators += 1
    add_Commodity_Channel_Index(stock_data)
    num_technical_indicators += 1
    add_Rate_of_Change(stock_data)
    num_technical_indicators += 1
    add_Triple_Exponential_Moving_Average(stock_data)
    num_technical_indicators += 1
    add_WilliamsR(stock_data)
    num_technical_indicators += 1
    add_Parabolic_SAR(stock_data)
    num_technical_indicators += 1
    add_Aroon_Oscillator(stock_data)
    num_technical_indicators += 1
    add_Detrended_Price_Oscillator(stock_data)
    num_technical_indicators += 1
    add_Money_Flow_Index(stock_data)
    num_technical_indicators += 1
     # Add Percentage Price Oscillator only for SPY
    if symbol == 'SPY':
        add_Percentage_Price_Oscillator(stock_data)
        num_technical_indicators += 1
        print("Added PPO for SPY")
     # Add 1-hour and 6-hour rolling mean
    stock_data_1h = stock_data.resample('1H').mean()
    stock_data_6h = stock_data.resample('6H').mean()

    stock_data['1_hour_mean'] = stock_data_1h['Close'].rolling(window=1).mean()
    stock_data['6_hour_mean'] = stock_data_6h['Close'].rolling(window=1).mean()

    stock_data['1_hour_mean'] = stock_data['1_hour_mean'].interpolate(method='time')
    stock_data['6_hour_mean'] = stock_data['6_hour_mean'].interpolate(method='time')
    num_technical_indicators += 2  # Increment the count for the new features

    # Add Volume Range and OBV features
    add_Volume_Range(stock_data)
    num_technical_indicators += 1

    add_OBV(stock_data)
    num_technical_indicators += 1

    print(f"Added {num_technical_indicators} technical indicators to stock data")
    return stock_data

def get_company_overview(symbol, api_key):
    fd = FundamentalData(key=api_key)
    data, _ = fd.get_company_overview(symbol=symbol)
    return data

def get_alpha_vantage_time_series_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    data = data.rename(columns={'4. close': 'Close_av'})
    return data

def get_stock_data(ticker, start, end):
    # Download daily stock data
    daily_stock_data = yf.download(ticker, start=start, end=end)
    daily_stock_data.index = daily_stock_data.index.tz_localize(None)
    print(f"{ticker}: {len(daily_stock_data)} daily data points from Yahoo Finance")

    # Download hourly stock data for the last 730 days
    end_date_hourly = end
    start_date_hourly = end - pd.DateOffset(days=730)
    hourly_stock_data = yf.download(ticker, start=start_date_hourly, end=end_date_hourly, interval='1h')
    hourly_stock_data.index = hourly_stock_data.index.tz_localize(None)
    print(f"{ticker}: {len(hourly_stock_data)} hourly data points from Yahoo Finance")

    # Merge daily and hourly stock data
    stock_data = pd.concat([daily_stock_data, hourly_stock_data])
    # Remove duplicates and sort by date
    stock_data = stock_data.loc[~stock_data.index.duplicated(keep='last')].sort_index()
    return stock_data

def merge_dataframes(stock_data_yf, stock_data_av):
    # Convert the timezone of the Yahoo Finance data to a timezone-naive index
    stock_data_yf.index = stock_data_yf.index.tz_localize(None)

    # Convert the timezone of the Alpha Vantage data to a timezone-naive index
    stock_data_av.index = stock_data_av.index.tz_localize(None)

    # Merge the two DataFrames and remove duplicates
    merged_data = pd.concat([stock_data_yf, stock_data_av], axis=1)
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
    return merged_data

    
