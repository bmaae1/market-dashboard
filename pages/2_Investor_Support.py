import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
import pandas_ta as ta
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Investor Decision Support", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.block-container { padding: 0rem 1rem 0rem 1rem !important; }
[data-testid="stHeader"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebarNavLink"] {
    font-size: 28px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:48px; margin-bottom:0;'>📈 Investor Decision Support</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px; margin-top:0; margin-bottom:0; color:#94A3B8;'>AI-powered next-5-day return prediction based on historical market data</p>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 1, 0.7, 1])

with col1:
    st.markdown("**Select Market**")
    market = st.selectbox(
        label="",
        options=["S&P 500 (^GSPC)", "FTSE 100 (^FTSE)"],
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Prediction Date**")
    pred_date = st.date_input(
        label="",
        value=datetime.date.today(),
        label_visibility="collapsed"
    )

st.markdown("""
<style>
div[data-testid="stButton"] > button {
    background-color: #7C3AED !important;
    color: white !important;
    border: none !important;
    font-size: 24px !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    height: 40px !important;
}
div[data-testid="stButton"] > button:hover {
    background-color: #6D28D9 !important;
}
</style>
""", unsafe_allow_html=True)

with col3:
    st.markdown("**&nbsp;**", unsafe_allow_html=True)
    predict_btn = st.button("📈 Predict", use_container_width=True)

if predict_btn:
    with st.spinner("Downloading market data and predicting..."):

        # Map market selection to yfinance ticker
        ticker = "^GSPC" if "S&P" in market else "^FTSE"
        model_path = "XGBoost_SP500_Stage1_model.json" if "S&P" in market else "XGBoost_FTSE100_Stage1_model.json"

        # Download last 120 days
        end_date = pred_date + datetime.timedelta(days=10)
        raw = yf.download(ticker, start=end_date - datetime.timedelta(days=120), end=end_date, interval="1d", auto_adjust=True)

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        raw = raw.dropna()

        # Calculate technical indicators
        def calculate_indicators(df):
            df = df.copy()
            close = df['Close'].squeeze()
            high  = df['High'].squeeze()
            low   = df['Low'].squeeze()
            vol   = df['Volume'].squeeze()

            df['RSI'] = ta.rsi(close, length=14)
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_hist'] = macd.iloc[:, 1]
            bb = ta.bbands(close, length=20, std=2)
            df['BB_upper'] = bb.iloc[:, 2]
            df['BB_middle'] = bb.iloc[:, 1]
            df['BB_lower'] = bb.iloc[:, 0]
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            adx = ta.adx(high, low, close, length=20)
            df['ADX20'] = adx.iloc[:, 0]
            df['DI_plus'] = adx.iloc[:, 1]
            df['DI_minus'] = adx.iloc[:, 2]
            df['OBV'] = ta.obv(close, vol)
            stoch = ta.stoch(high, low, close, k=14, d=3)
            df['Stoch_K'] = stoch.iloc[:, 0]
            df['ATR'] = ta.atr(high, low, close, length=14)
            df['EMA_9'] = ta.ema(close, length=9)
            df['EMA_21'] = ta.ema(close, length=21)
            df['EMA_crossover'] = (df['EMA_9'] > df['EMA_21']).astype(int)
            df['return'] = close.pct_change() * 100
            return df

        df_ind = calculate_indicators(raw)
        df_ind = df_ind.dropna()

        # Lag and rolling features
        LAG_PERIODS = [1, 3, 5]
        ROLLING_WINDOWS = [5, 10]

        if "S&P" in market:
            LAG_FEATURES = ['return', 'Volume', 'RSI', 'MACD_hist', 'MACD']
            ROLLING_FEATURES = ['return', 'Volume', 'RSI']
        else:
            LAG_FEATURES     = ['RSI', 'MACD', 'MACD_hist']
            ROLLING_FEATURES = ['RSI']

        for feat in LAG_FEATURES:
            if feat in df_ind.columns:
                for lag in LAG_PERIODS:
                    df_ind[f'{feat}_lag{lag}'] = df_ind[feat].shift(lag)

        for feat in ROLLING_FEATURES:
            if feat in df_ind.columns:
                for w in ROLLING_WINDOWS:
                    df_ind[f'{feat}_roll{w}_mean'] = df_ind[feat].rolling(w).mean()

        df_ind = df_ind.dropna()
        pred_date_ts = pd.Timestamp(pred_date)
        df_ind = df_ind[df_ind.index <= pred_date_ts]
        
        # Features list
        if "S&P" in market:
            features = ['Volume', 'RSI', 'MACD_hist', 'BB_width', 'ADX20',
                        'DI_plus', 'DI_minus', 'OBV', 'Stoch_K', 'ATR', 'EMA_crossover', 'return',
                        'return_lag1', 'return_lag3', 'return_lag5',
                        'Volume_lag1', 'Volume_lag3', 'Volume_lag5',
                        'RSI_lag1', 'RSI_lag3', 'RSI_lag5',
                        'MACD_hist_lag1', 'MACD_hist_lag3', 'MACD_hist_lag5',
                        'MACD_lag1', 'MACD_lag3', 'MACD_lag5',
                        'return_roll5_mean', 'return_roll10_mean',
                        'Volume_roll5_mean', 'Volume_roll10_mean',
                        'RSI_roll5_mean', 'RSI_roll10_mean']
        else:
            features = ['RSI', 'MACD', 'MACD_hist', 'BB_width', 'ADX20',
            'DI_plus', 'DI_minus', 'OBV', 'Stoch_K', 'EMA_crossover',
            'RSI_lag1', 'RSI_lag3', 'RSI_lag5',
            'MACD_lag1', 'MACD_lag3', 'MACD_lag5',
            'MACD_hist_lag1', 'MACD_hist_lag3', 'MACD_hist_lag5',
            'RSI_roll5_mean', 'RSI_roll10_mean']

        # Load model
        model = XGBRegressor()
        model.load_model(model_path)

        # Predict
        pred_date_ts = pd.Timestamp(pred_date)
        if pred_date_ts in df_ind.index:
            X_latest = df_ind[features].loc[[pred_date_ts]].values
        else:
            nearest = df_ind.index[df_ind.index.get_indexer([pred_date_ts], method='nearest')[0]]
            X_latest = df_ind[features].loc[[nearest]].values
            st.warning(f"Date not available, using nearest trading day: {nearest.strftime('%d %b %Y')}")
        predicted_return = model.predict(X_latest)[0]

        # Signal
        if predicted_return > 0.3:
            signal = "BUY"
            signal_color = "#00C853"
        elif predicted_return < -0.3:
            signal = "SELL"
            signal_color = "#FF1744"
        else:
            signal = "HOLD"
            signal_color = "#FFD600"


    # Prediction summary cards (outside spinner, inside if predict_btn)
    st.markdown("<h3 style='color:#7C3AED; margin-top:20px;'>Prediction Summary</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    return_color = "#00C853" if predicted_return > 0 else "#FF1744"
    with c1:
        with st.container(border=True):
            st.markdown("**Predicted Next 5-Day Return**")
            st.markdown(f"<h2 style='color:{return_color};'>{predicted_return:+.2f}%</h2>", unsafe_allow_html=True)
            st.caption("Average return over next 5 trading days")

    with c2:
        with st.container(border=True):
            st.markdown("**Recommendation**")
            st.markdown(f"<h2 style='color:{signal_color};'>{signal}</h2>", unsafe_allow_html=True)
            if signal == "BUY":
                st.caption("Model suggests a positive outlook")
            elif signal == "SELL":
                st.caption("Model suggests a negative outlook")
            else:
                st.caption("Model suggests a neutral outlook")








    # chart
    hover_style = dict(
    bgcolor="rgba(0,0,0,0.8)",
    font=dict(
        size=20,
        family="Calibri",
        color="white"
    )
)
    import plotly.graph_objects as go
    col1, col2 =  st.columns([2,1.2])
    

    with col1:
        st.markdown("<h3 style='color:#7C3AED; margin-top:20px;'>Price Chart (Last 30 Days)</h3>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw.index[-30:],
            y=raw['Close'].squeeze()[-30:],
            mode='lines',
            line=dict(color='#6366F1', width=2),
            name='Close Price'
        ))
        fig.update_layout(
            hoverlabel=hover_style,
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='#FAFAFA'),
            xaxis=dict(showgrid=True, gridcolor='#1E2130'),
            yaxis=dict(showgrid=True, gridcolor='#1E2130'),
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(font=dict(color='#FAFAFA')),
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)










    # disclaimer
    st.markdown("""
<div style='background-color:#1A1F2E; border:1px solid #2E3250; border-radius:10px; padding:1rem; margin-top:1rem;'>
<p style='font-size:26px; color:#94A3B8; margin:0;'>
⚠️ <b style='color:#FAFAFA;'>Disclaimer:</b> Predictions are generated by a machine learning model trained on historical market data (2019–2026). 
They are not financial advice and should not be used as the sole basis for investment decisions. 
Past performance does not guarantee future results. Always consult a qualified financial advisor before investing.
</p>
</div>
""", unsafe_allow_html=True)


















    # table
    with col2:
        st.markdown("<h3 style='color:#7C3AED; margin-top:20px;'>Recent Predictions (Last 10 Days)</h3>", unsafe_allow_html=True)
    
        raw['log_return'] = np.log(raw['Close'].squeeze() / raw['Close'].squeeze().shift(1)) * 100
        raw['actual_5day_return'] = (
            raw['log_return'].shift(-1) + raw['log_return'].shift(-2) +
            raw['log_return'].shift(-3) + raw['log_return'].shift(-4) +
            raw['log_return'].shift(-5)
        ) / 5
    
        # Get 10 trading days before selected date
        pred_date_ts = pd.Timestamp(pred_date)
        available_dates = df_ind.index[df_ind.index < pred_date_ts]
        last_10 = df_ind[features].loc[available_dates[-10:]].iloc[::-1]  # descending
    
        rows = []
        for date, row in last_10.iterrows():
            pred = model.predict(row.values.reshape(1, -1))[0]
            if pred > 0.3:
                rec, rec_color = "BUY", "#00C853"
            elif pred < -0.3:
                rec, rec_color = "SELL", "#FF1744"
            else:
                rec, rec_color = "HOLD", "#FFD600"
    
            actual = raw['actual_5day_return'].loc[date] if date in raw.index else None
            actual_str = f"{actual:+.2f}" if actual is not None and not np.isnan(actual) else "-"
            error = pred - actual if actual is not None and not np.isnan(actual) else None
            error_str = f"{error:+.2f}" if error is not None else "-"
            pred_color = "#00C853" if pred > 0 else "#FF1744"
            actual_color = "#00C853" if actual is not None and not np.isnan(actual) and actual > 0 else "#FF1744"
            error_color = "#FF4444" if error is not None and error < 0 else "#00C853"
    
            rows.append(f"<tr><td>{date.strftime('%d %b %Y')}</td><td style='color:{pred_color}'>{pred:+.2f}</td><td style='color:{actual_color if actual_str != '-' else '#FAFAFA'}'>{actual_str}</td><td style='color:{error_color if error_str != '-' else '#FAFAFA'}'>{error_str}</td></tr>")
    
        st.markdown(f"""
        <table style="width:100%; border-collapse:collapse; font-size:30px;">
        <thead style="position:sticky; top:0; background:#0E1117;">
        <tr><th style="color:#94A3B8; font-weight:normal; padding:8px 4px; border-bottom:1px solid #1E2130;">Date</th>
        <th style="color:#94A3B8; font-weight:normal; padding:8px 4px; border-bottom:1px solid #1E2130;">Predicted (%)</th>
        <th style="color:#94A3B8; font-weight:normal; padding:8px 4px; border-bottom:1px solid #1E2130;">Actual (%)</th>
        <th style="color:#94A3B8; font-weight:normal; padding:8px 4px; border-bottom:1px solid #1E2130;">Error (%)</th>
        </thead>
        <tbody>
        {''.join(rows)}
        </tbody>
        </table>
        """, unsafe_allow_html=True)

   