
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

st.set_page_config(page_title="📈 DAX Intraday Watchlist Bot", layout="wide")

DAX_TICKERS = ["ADS.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DAI.DE", "DBK.DE", "DB1.DE", "DPW.DE", "DTE.DE", 
               "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "LIN.DE", "MRK.DE", "MUV2.DE", 
               "PUM.DE", "RWE.DE", "SAP.DE", "SIE.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"]

def add_features(df):
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA30'] = df['Close'].rolling(30).mean()

    try:
        close_series = pd.Series(df['Close'].values, index=df.index)
        if close_series.isna().sum() == 0 and len(close_series) >= 20:
            rsi_calc = RSIIndicator(close=close_series, window=14)
            df['RSI'] = rsi_calc.rsi()
        else:
            df['RSI'] = np.nan
    except Exception:
        df['RSI'] = np.nan

    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Return1'] = df['Close'].pct_change()
    df['Return5'] = df['Close'].pct_change(5)
    df['Volume_Change'] = df['Volume'].pct_change()
    return df

def create_target(df, horizon=6, threshold=0.003):
    future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Target'] = (future_returns > threshold).astype(int)
    return df

def get_prediction_df(ticker, confidence_threshold):
    df = yf.download(ticker, interval="15m", period="5d", progress=False)
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    df = add_features(df)
    df = create_target(df)
    df.dropna(inplace=True)

    features = ['SMA10', 'SMA30', 'RSI', 'Momentum', 'Volatility', 'Return1', 'Return5', 'Volume_Change']
    X = df[features]
    y = df['Target']

    if len(X) < 10 or X.isnull().values.any() or y.isnull().any():
        return pd.DataFrame()

    if X.isnull().values.any():
        return pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    df_result = df.iloc[-len(X_test):].copy()
    df_result['Prob_Up'] = probs
    df_result['HighConfidence'] = df_result['Prob_Up'] >= confidence_threshold
    df_result['BuySignal'] = df_result['HighConfidence'].astype(int)

    return df_result

st.sidebar.title("🔧 Einstellungen")
selected_ticker = st.sidebar.selectbox("Wähle eine DAX-Aktie", DAX_TICKERS)
confidence_threshold = st.sidebar.slider("Vertrauens-Schwelle (%)", 50, 100, 90) / 100

st.title("📋 DAX Intraday Watchlist Dashboard")
st.markdown(f"**Ticker:** {selected_ticker} | **Confidence ≥ {int(confidence_threshold*100)}%**")

with st.spinner("🔄 Daten werden analysiert..."):
    df_result = get_prediction_df(selected_ticker, confidence_threshold)

if df_result.empty:
    st.error("Keine Daten oder Signale gefunden.")
else:
    st.subheader("📈 Kursverlauf & Kaufpunkte")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_result.index, df_result['Close'], label='Kurs', color='blue')
    ax.scatter(df_result[df_result['BuySignal'] == 1].index, df_result[df_result['BuySignal'] == 1]['Close'],
               label='💰 Buy Signal', color='green', marker='^', s=100)
    ax.set_title(f"{selected_ticker} - Kurs & Kauf-Signale")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("📊 Watchlist-Tabelle")
    st.dataframe(df_result[df_result['BuySignal'] == 1][['Close', 'Prob_Up']].rename(columns={
        'Close': 'Kurs (€)', 'Prob_Up': 'Vertrauen (%)'
    }).round(3))
