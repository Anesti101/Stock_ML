"""Interactive Stock Market Dashboard.

A Dash-based web application for visualizing stock data, technical indicators,
and model predictions.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from data_prep import fetch_data_yfinance, validate_stock_data
except ImportError:
    print("Warning: data_prep module not found. Using mock functions.")
    fetch_data_yfinance = None
    validate_stock_data = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_moving_averages(data: pd.DataFrame, windows: list = [5, 10, 20, 50]) -> pd.DataFrame:
    """Calculate simple moving averages for given windows.
    
    Args:
        data: DataFrame with stock data
        windows: List of window sizes for moving averages
    
    Returns:
        DataFrame with added MA columns
    """
    df = data.copy()
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with stock data
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> tuple:
    """Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with stock data
        window: Rolling window size
        num_std: Number of standard deviations
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle_band = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return middle_band, upper_band, lower_band


def create_candlestick_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create candlestick chart with volume.
    
    Args:
        data: DataFrame with OHLCV data
        ticker: Stock ticker symbol
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_technical_indicators_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create chart with technical indicators.
    
    Args:
        data: DataFrame with stock data and indicators
        ticker: Stock ticker symbol
    
    Returns:
        Plotly figure object
    """
    # Calculate indicators
    df = calculate_moving_averages(data)
    middle_band, upper_band, lower_band = calculate_bollinger_bands(df)
    
    fig = go.Figure()
    
    # Close price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Close Price',
        line=dict(color='white', width=2)
    ))
    
    # Moving averages
    colors = ['cyan', 'yellow', 'magenta', 'green']
    for i, window in enumerate([5, 10, 20, 50]):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'SMA_{window}'],
            name=f'SMA {window}',
            line=dict(color=colors[i], width=1, dash='dash')
        ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=upper_band,
        name='BB Upper',
        line=dict(color='rgba(250, 128, 114, 0.5)', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=lower_band,
        name='BB Lower',
        line=dict(color='rgba(250, 128, 114, 0.5)', width=1),
        fill='tonexty',
        fillcolor='rgba(250, 128, 114, 0.1)'
    ))
    
    fig.update_layout(
        title=f'{ticker} Technical Indicators',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_rsi_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create RSI indicator chart.
    
    Args:
        data: DataFrame with stock data
        ticker: Stock ticker symbol
    
    Returns:
        Plotly figure object
    """
    rsi = calculate_rsi(data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=rsi,
        name='RSI',
        line=dict(color='cyan', width=2)
    ))
    
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title=f'{ticker} RSI (14)',
        yaxis_title='RSI',
        xaxis_title='Date',
        template='plotly_dark',
        height=300,
        yaxis_range=[0, 100]
    )
    
    return fig


def create_volume_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create volume chart.
    
    Args:
        data: DataFrame with stock data
        ticker: Stock ticker symbol
    
    Returns:
        Plotly figure object
    """
    colors = ['red' if close < open_ else 'green' 
              for close, open_ in zip(data['Close'], data['Open'])]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=f'{ticker} Trading Volume',
        yaxis_title='Volume',
        xaxis_title='Date',
        template='plotly_dark',
        height=300
    )
    
    return fig


def calculate_summary_stats(data: pd.DataFrame) -> dict:
    """Calculate summary statistics for stock data.
    
    Args:
        data: DataFrame with stock data
    
    Returns:
        Dictionary with summary statistics
    """
    current_price = data['Close'].iloc[-1]
    price_change = current_price - data['Close'].iloc[0]
    price_change_pct = (price_change / data['Close'].iloc[0]) * 100
    
    return {
        'current_price': f"${current_price:.2f}",
        'price_change': f"${price_change:.2f}",
        'price_change_pct': f"{price_change_pct:.2f}%",
        'high': f"${data['High'].max():.2f}",
        'low': f"${data['Low'].min():.2f}",
        'avg_volume': f"{data['Volume'].mean():.0f}",
        'volatility': f"{data['Close'].pct_change().std() * 100:.2f}%"
    }


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Stock Market Dashboard"

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("📊 Stock Market Dashboard", style={'textAlign': 'center', 'color': 'white'}),
        html.Hr()
    ]),
    
    html.Div([
        html.Div([
            html.Label("Stock Ticker:", style={'color': 'white', 'fontWeight': 'bold'}),
            dcc.Input(
                id='ticker-input',
                type='text',
                value='AAPL',
                placeholder='Enter ticker symbol (e.g., AAPL)',
                style={'width': '200px', 'marginRight': '10px'}
            ),
            
            html.Label("Period:", style={'color': 'white', 'fontWeight': 'bold', 'marginLeft': '20px'}),
            dcc.Dropdown(
                id='period-dropdown',
                options=[
                    {'label': '1 Month', 'value': '1mo'},
                    {'label': '3 Months', 'value': '3mo'},
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '2 Years', 'value': '2y'},
                    {'label': '5 Years', 'value': '5y'}
                ],
                value='1y',
                style={'width': '150px', 'display': 'inline-block', 'marginLeft': '10px'}
            ),
            
            html.Button(
                'Load Data',
                id='load-button',
                n_clicks=0,
                style={
                    'marginLeft': '20px',
                    'padding': '10px 20px',
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                }
            ),
        ], style={'padding': '20px', 'textAlign': 'center'}),
    ]),
    
    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center', 'padding': '10px'}),
    
    html.Div(id='summary-stats', style={'padding': '20px'}),
    
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            html.Div([
                dcc.Graph(id='candlestick-chart'),
                dcc.Graph(id='technical-chart'),
                dcc.Graph(id='rsi-chart'),
                dcc.Graph(id='volume-chart')
            ])
        ]
    ),
    
    # Store for data
    dcc.Store(id='stock-data-store')
    
], style={'backgroundColor': '#1e1e1e', 'minHeight': '100vh', 'padding': '20px'})


@app.callback(
    [Output('stock-data-store', 'data'),
     Output('error-message', 'children')],
    [Input('load-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('period-dropdown', 'value')]
)
def load_stock_data(n_clicks, ticker, period):
    """Load stock data when button is clicked."""
    if n_clicks == 0:
        return None, ""
    
    if not ticker:
        return None, "Please enter a ticker symbol"
    
    try:
        # Calculate date range
        end_date = datetime.now()
        period_days = {
            '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825
        }
        start_date = end_date - timedelta(days=period_days.get(period, 365))
        
        # Fetch data
        if fetch_data_yfinance:
            data = fetch_data_yfinance(
                ticker.upper(),
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        else:
            # Mock data for testing when yfinance not available
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            data = pd.DataFrame({
                'Open': np.random.uniform(100, 110, len(dates)),
                'High': np.random.uniform(110, 120, len(dates)),
                'Low': np.random.uniform(90, 100, len(dates)),
                'Close': np.random.uniform(100, 110, len(dates)),
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        
        # Convert to JSON-serializable format
        data_dict = {
            'index': data.index.strftime('%Y-%m-%d').tolist(),
            'data': data.to_dict('list')
        }
        
        logger.info(f"Successfully loaded data for {ticker}")
        return data_dict, ""
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, f"Error loading data: {str(e)}"


@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('technical-chart', 'figure'),
     Output('rsi-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('summary-stats', 'children')],
    [Input('stock-data-store', 'data')],
    [State('ticker-input', 'value')]
)
def update_charts(data_dict, ticker):
    """Update all charts when data changes."""
    if not data_dict:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark')
        return empty_fig, empty_fig, empty_fig, empty_fig, ""
    
    # Reconstruct DataFrame
    data = pd.DataFrame(data_dict['data'])
    data.index = pd.to_datetime(data_dict['index'])
    
    # Create charts
    candlestick = create_candlestick_chart(data, ticker.upper())
    technical = create_technical_indicators_chart(data, ticker.upper())
    rsi = create_rsi_chart(data, ticker.upper())
    volume = create_volume_chart(data, ticker.upper())
    
    # Calculate summary stats
    stats = calculate_summary_stats(data)
    
    summary = html.Div([
        html.H3(f"{ticker.upper()} Summary Statistics", style={'color': 'white', 'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H4("Current Price", style={'color': 'gray'}),
                    html.H2(stats['current_price'], style={'color': 'white'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("Change", style={'color': 'gray'}),
                    html.H2(stats['price_change'], style={'color': 'green' if '-' not in stats['price_change'] else 'red'}),
                    html.P(stats['price_change_pct'], style={'color': 'gray'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("High", style={'color': 'gray'}),
                    html.H2(stats['high'], style={'color': 'white'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("Low", style={'color': 'gray'}),
                    html.H2(stats['low'], style={'color': 'white'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("Avg Volume", style={'color': 'gray'}),
                    html.H2(stats['avg_volume'], style={'color': 'white'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("Volatility", style={'color': 'gray'}),
                    html.H2(stats['volatility'], style={'color': 'white'})
                ], className='stat-card'),
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
                'gap': '20px'
            })
        ])
    ])
    
    return candlestick, technical, rsi, volume, summary


if __name__ == '__main__':
    logger.info("Starting Stock Market Dashboard...")
    logger.info("Navigate to http://localhost:8050")
    app.run_server(debug=True, host='0.0.0.0', port=8050)