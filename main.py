import yfinance as yf
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Stock_ML", layout="wide")

ticker = st.text_input("Ticker", "AAPL")

# Intervals & periods (allow intraday)
interval = st.selectbox("Interval", ["1m","5m","15m","30m","1h","1d","1wk","1mo"], index=4)
intraday = {"1m","2m","5m","15m","30m","60m","90m","1h"}
period = st.selectbox("Period",
                      ["5d","7d","30d","60d"] if interval in intraday else ["6mo","1y","5y","10y","max"],
                      index=2 if interval in intraday else 1)

df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, actions=True)

if df.empty:
    st.warning("No data returned. Try a shorter period or a different interval.")
else:
    df = df.reset_index()
    time_col = "Date" if "Date" in df.columns else "Datetime"   # <- key fix
    st.dataframe(df.tail(250))
    fig = px.line(df, x=time_col, y="Close", title=f"{ticker} Close — {interval} over {period}")
    st.plotly_chart(fig, use_container_width=True)
