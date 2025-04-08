
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📋 DAX Watchlist", layout="wide")

DAX_TICKERS = ["ADS.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DAI.DE", "DBK.DE", "DB1.DE", "DPW.DE", "DTE.DE", 
               "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "LIN.DE", "MRK.DE", "MUV2.DE", 
               "PUM.DE", "RWE.DE", "SAP.DE", "SIE.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"]

def add_features(df):
    try:
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA30'] = df['Close'].rolling(30).mean()
        df['RSI'] = RSIIndicator(close=pd.Series(df['Close'].values, index=df.index), window=14).rsi()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Return1'] = df['Close'].pct_change()
        df['Return5'] = df['Close'].pct_change(5)
        df['Volume_Change'] = df['Volume'].pct_change()
    except:
        return pd.DataFrame()
    return df

def create_target(df, horizon=6, threshold=0.003):
    future_returns = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Target'] = (future_returns > threshold).astype(int)
    return df

def analyze_ticker(ticker, confidence_threshold):
    df = yf.download(ticker, interval="15m", period="5d", progress=False)
    if df.empty or len(df) < 30:
        return []

    df = add_features(df)
    df = create_target(df)
    df.dropna(inplace=True)

    features = ['SMA10', 'SMA30', 'RSI', 'Momentum', 'Volatility', 'Return1', 'Return5', 'Volume_Change']
    if any(f not in df.columns for f in features) or df[features].isnull().values.any() or df['Target'].isnull().any():
        return []

    if len(df) < 20:
        return []

    X = df[features]
    y = df['Target']
    if len(X) < 10:
        return []

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    df_result = df.iloc[-len(X_test):].copy()
    df_result['Prob_Up'] = probs
    df_result['HighConfidence'] = df_result['Prob_Up'] >= confidence_threshold
    df_result['Ticker'] = ticker

    signals = df_result[df_result['HighConfidence']].copy()
    return signals[['Ticker', 'Close', 'Prob_Up']]

st.title("📋 DAX High-Confidence Watchlist")
confidence_threshold = st.sidebar.slider("Vertrauens-Schwelle (%)", 50, 100, 90) / 100

st.write(f"Zeige Signale mit Confidence ≥ {int(confidence_threshold * 100)}%")

results = []

with st.spinner("🔄 DAX-Aktien werden analysiert..."):
    for ticker in DAX_TICKERS:
        signals = analyze_ticker(ticker, confidence_threshold)
        if len(signals):
            results.append(signals)

if results:
    watchlist_df = pd.concat(results)
    watchlist_df = watchlist_df.rename(columns={"Close": "Kurs (€)", "Prob_Up": "Vertrauen (%)"})
    watchlist_df["Vertrauen (%)"] = (watchlist_df["Vertrauen (%)"] * 100).round(2)
    watchlist_df = watchlist_df.sort_values("Vertrauen (%)", ascending=False)
    st.success(f"{len(watchlist_df)} High-Confidence Signale gefunden.")
    st.dataframe(watchlist_df.reset_index().rename(columns={"index": "Zeitpunkt"}))
else:
    st.warning("📭 Keine Signale gefunden. Versuche einen niedrigeren Schwellenwert oder später erneut.")
