import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as pyo
import os
import pandas as pd
import time

def save_plot(stock_data, y_test, y_pred, rf_y_pred, symbol, model_type, tomorrow_price, future_prices_7d, future_prices_1m, rf_future_prices, nn_future_prices, nn_y_test, nn_y_pred):
    # rest of the code

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
    # Create a DataFrame for the random forest predicted prices
    nn_predicted_prices_df = pd.DataFrame({'Date': y_test.index, 'Price': nn_y_pred})
    nn_predicted_prices_df.set_index('Date', inplace=True)
    # Interpolate the random forest predicted prices
    nn_predicted_prices_interpolated = nn_predicted_prices_df.reindex(stock_data.index).interpolate(method='time')
    nn_predicted_prices_trace = go.Scatter(
        x=nn_predicted_prices_interpolated.index,
        y=nn_predicted_prices_interpolated['Price'],
        mode='lines+markers',
        name='Neural Network Predicted prices'
    )

    nn_tomorrow_predicted_trace = go.Scatter(
        x=[stock_data.index[-1] + pd.DateOffset(days=1)],
        y=[nn_future_prices[-2]],
        mode='lines+markers',
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