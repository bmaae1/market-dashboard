# 1. Imports
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 2. Page config (must be first st. command)
st.set_page_config(page_title="Market Insight Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# 3. CSS styling
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.block-container { padding: 0rem 1rem 0rem 1rem !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 0rem !important; }
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stSidebarCollapsedControl"] { display: block !important; }
[data-testid="collapsedControl"] { display: block !important; }
#root > div:first-child { margin-top: 0 !important; }[data-testid="stSidebarNavLink"] {
    font-size: 28px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# 4. Load data
sp500 = pd.read_csv("sp500_features_final.csv", parse_dates=["Date"], index_col="Date")
ftse100 = pd.read_csv("ftse_features_final.csv", parse_dates=["Date"], index_col="Date")


# 5. Title and dropdown
col1, col2 = st.columns([3,1.5])
with col1:
    st.markdown("<h1 style='font-size:48px; margin-bottom:0;'>Market Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:20px;  margin-top:0;  margin-bottom:0;'>FTSE 100 & S&P 500 Market Prediction Dashboard</p>", unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown("""
        <style>
        .method-title { font-size: 24px !important; font-weight: bold; color: #FAFAFA; margin:0; }
        .method-text { font-size: 18px !important; color: #94A3B8; margin-top: 6px; }
        </style>
        <p class='method-title'>Methodology</p>
        <div style='display:flex; gap:2rem;' class='method-text'>
            <div>
                ✓ Stage 1: Technical indicators only<br>
                ✓ Stage 2: Technical + VIX + FinBERT sentiment
            </div>
            <div>
                ✓ Sentiment: Financial news encoded using FinBERT<br>
                ✓ Models: SVR · XGBoost · RF
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# Large selectbox via CSS
st.markdown("""
<style>
div[data-testid="stSelectbox"] > div > div {
    font-size: 18px !important;
    min-height: 50px !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    st.markdown("### Select Market Index")
    
    market = st.selectbox(
        label="",
        options=["S&P 500", "FTSE 100"],
        label_visibility="collapsed"
    )

    df = sp500.copy() if market == "S&P 500" else ftse100.copy()





# Key Metrics card
with col3:
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
    
        with m1:
            st.metric("Dataset", market)
        with m2:
            start = df.index.min().strftime("%b %Y")
            end = df.index.max().strftime("%b %Y")
            st.metric("Time Period", f"{start} – {end}")
        with m3:
            st.metric("Observations", f"{len(df):,}")
        with m4:
            st.metric("Current Model", "XGBoost")







# Define once at the top
hover_style = dict(
    bgcolor="rgba(0,0,0,0.8)",
    font=dict(
        size=20,
        family="Calibri",
        color="white"
    )
)



col1, col2, col3 = st.columns([3, 2, 1])  # candlestick takes 2/3, gauge takes 1/3

with col1:
    with st.container(border=True):

        # Use full dataset, remove rows with missing OHLC
        df_filtered = df.dropna(subset=["Open", "High", "Low", "Close"])
    
        fig = go.Figure()

        # Upper Bollinger Band 
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered["BB_upper"],
            line=dict(color="rgba(173,216,230,0.6)", width=1),
            name="Upper Band",
        ))
        
        # Middle Bollinger Band
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered["BB_middle"],
            line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash"),
            name="Middle Band",
        ))
        
        # Lower Bollinger Band with fill
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered["BB_lower"],
            line=dict(color="rgba(173,216,230,0.6)", width=1),
            fill="tonexty",
            fillcolor="rgba(173,216,230,0.05)",
            name="Lower Band",
        ))
        
        # Candlestick trace — add last so it appears first in hover
        fig.add_trace(go.Candlestick(
            x=df_filtered.index,
            open=df_filtered["Open"],
            high=df_filtered["High"],
            low=df_filtered["Low"],
            close=df_filtered["Close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            name="",
        ))
    
        fig.update_layout(
            hoverlabel=hover_style,
            title=f"{market} — Candlestick Chart with Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            hovermode="x unified",
            height=410,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(
                autorange=True,          # y-axis adjusts to visible candles
                fixedrange=False,        # allows y-axis to rescale on zoom/scroll
            ),
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    autorange=True,      # rangeslider adjusts with data
                ),
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # remove weekends
                ],
                autorange=True,
            ),
        )
    
        st.plotly_chart(fig, use_container_width=True)




















# sentiment visualization
with col2:
    with st.container(border=True):
        st.markdown("### Price and Sentiment Trends")

        if market == "FTSE 100":
            # Price vs Guardian sentiment chart
            fig_sent = go.Figure()

            # FTSE close price
            fig_sent.add_trace(go.Scatter(
                x=df.index,
                y=df["Close"],
                name="FTSE 100 Close",
                line=dict(color="steelblue", width=1),
                yaxis="y1",
            ))

            # 30-day Guardian sentiment MA
            sentiment_ma = df["guardian_sentiment"].rolling(30).mean()
            fig_sent.add_trace(go.Scatter(
                x=df.index,
                y=sentiment_ma,
                name="30-Day Sentiment MA",
                line=dict(color="red", width=1.5),
                yaxis="y2",
                fill="tozeroy",
                fillcolor="rgba(255,0,0,0.1)",
            ))

        elif market == "S&P 500":
            # Filter to news coverage period
            df_sent = df.loc["2019-01-01":"2024-03-04"]

            fig_sent = go.Figure()

            # S&P 500 close price
            fig_sent.add_trace(go.Scatter(
                x=df_sent.index,
                y=df_sent["Close"],
                name="S&P 500 Close",
                line=dict(color="steelblue", width=1),
                yaxis="y1",
            ))

            # 30-day S&P news sentiment MA
            sentiment_ma = df_sent["sp500_sentiment"].rolling(30).mean()
            fig_sent.add_trace(go.Scatter(
                x=df_sent.index,
                y=sentiment_ma,
                name="30-Day Sentiment MA",
                line=dict(color="red", width=1.5),
                yaxis="y2",
                fill="tozeroy",
                fillcolor="rgba(255,0,0,0.1)",
            ))

        fig_sent.update_layout(
            hoverlabel=hover_style,
            title=f"{market} Price vs News Sentiment Score",
            template="plotly_dark",
            hovermode="x unified",
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(title="Close Price", color="steelblue"),
            yaxis2=dict(
                title="Sentiment Score",
                overlaying="y",  # overlay on same chart
                side="right",
                color="red",
                range=[-0.5, 0.25],
            ),
        )

        st.plotly_chart(fig_sent, use_container_width=True)























# sentiment gauge
with col3:
    with st.container(border=True):
        st.markdown("#### Sentiment Summary Statistics")
        table_css = "<style>table {width:100% !important; height:300px !important;} td, th {text-align:left !important; font-weight:bold;}</style>"

        if market == "FTSE 100":
            guardian_stats = df["guardian_sentiment"].describe().drop("count").round(4)
            st.markdown("**Guardian News Sentiment**")
            st.markdown(table_css + guardian_stats.to_frame().to_html(bold_rows=False), unsafe_allow_html=True)

        elif market == "S&P 500":
            sp500_stats = df["sp500_sentiment"].describe().drop("count").round(4)
            st.markdown("**S&P 500 News Sentiment**")
            st.markdown(table_css + sp500_stats.to_frame().to_html(bold_rows=False), unsafe_allow_html=True)























# technical indicators
col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    with st.container(border=True):
        st.markdown("### Technical Indicators")

        # Filter missing values
        df_tech = df.dropna(subset=["RSI", "ADX20", "MACD", "MACD_signal", "MACD_hist"])

        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_tech.index,
            y=df_tech["RSI"],
            name="RSI",
            line=dict(color="#26a69a", width=1.2),
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(
            hoverlabel=hover_style,
            title=f"{market} — RSI",
            template="plotly_dark",
            hovermode="x unified",
            height=250,
            margin=dict(l=40, r=40, t=40, b=20),
            yaxis=dict(title="RSI", range=[0, 100]),
            xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]),
        )

        # ADX chart
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(
            x=df_tech.index,
            y=df_tech["ADX20"],
            name="ADX",
            line=dict(color="#f5a623", width=1.2),
        ))
        fig_adx.add_hline(y=25, line_dash="dash", line_color="grey", annotation_text="Trend Threshold (25)")
        fig_adx.update_layout(
            hoverlabel=hover_style,
            title=f"{market} — ADX",
            template="plotly_dark",
            hovermode="x unified",
            height=250,
            margin=dict(l=40, r=40, t=40, b=20),
            yaxis=dict(title="ADX"),
            xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]),
        )

        # MACD chart
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_tech.index,
            y=df_tech["MACD"],
            name="MACD",
            line=dict(color="#26a69a", width=1.2),
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_tech.index,
            y=df_tech["MACD_signal"],
            name="Signal",
            line=dict(color="#ef5350", width=1.2),
        ))
        # MACD histogram as bar chart
        fig_macd.add_trace(go.Bar(
            x=df_tech.index,
            y=df_tech["MACD_hist"],
            name="Histogram",
            marker_color=df_tech["MACD_hist"].apply(
                lambda x: "#26a69a" if x >= 0 else "#ef5350"  # green positive, red negative
            ),
        ))
        fig_macd.update_layout(
            hoverlabel=hover_style,
            title=f"{market} — MACD",
            template="plotly_dark",
            hovermode="x unified",
            height=250,
            margin=dict(l=40, r=40, t=40, b=20),
            yaxis=dict(title="MACD"),
            xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]),
        )

        st.plotly_chart(fig_rsi, use_container_width=True)
        st.plotly_chart(fig_adx, use_container_width=True)
        st.plotly_chart(fig_macd, use_container_width=True)













    


















# model results
# Row 3 - Right side: Model Performance
with col3:
    with st.container(border=True):
        st.markdown("### Model Performance")

        # Metric selection
        metric_choice = st.selectbox("Select Metric", ["RMSE", "R²"])

        # All regression model results per market
        model_results = {
            "S&P 500": {
                "XGBoost": {"Stage 1": {"RMSE": 0.4505, "R²": -0.7879}, "Stage 2": {"RMSE": 0.3922, "R²": -0.2807}},
            },
            "FTSE 100": {
               "XGBoost": {"Stage 1": {"RMSE": 0.3381, "R²": -0.0127}, "Stage 2": {"RMSE": 0.3353, "R²": 0.0433}},
            },
        }

        # Get data for selected market
        data = model_results[market]
        models = list(data.keys())

        # Build grouped bar chart - all models side by side
        fig_perf = go.Figure()

        # Stage 1 bars
        fig_perf.add_trace(go.Bar(
            name="Stage 1",
            x=models,
            y=[data[m]["Stage 1"][metric_choice] for m in models],
            marker_color="#26a69a",  # teal
        ))

        # Stage 2 bars
        fig_perf.add_trace(go.Bar(
            name="Stage 2",
            x=models,
            y=[data[m]["Stage 2"][metric_choice] for m in models],
            marker_color="#f5a623",  # orange
        ))

        fig_perf.update_layout(
            hoverlabel=hover_style,
            title=f"{metric_choice} — Stage 1 vs Stage 2 ({market})",
            template="plotly_dark",
            barmode="group",         # all models grouped side by side
            hovermode="x unified",
            height=282,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(title=metric_choice),
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",  # anchor to right
            x=1,              # push to far right
        ),        )

        st.plotly_chart(fig_perf, use_container_width=True)





































# classification
# Classification performance card
with col3:
    with st.container(border=True):
        st.markdown("### Random Forest Classification Performance")
    
        # Classification results per market
        clf_results = {
            "S&P 500": {
                "Stage 1": {"Accuracy": 42.6, "F1": 43.0},
                "Stage 2": {"Accuracy": 59.1, "F1": 58.0},
            },
            "FTSE 100": {
                "Stage 1": {"Accuracy": 45.5, "F1": 46.1},
                "Stage 2": {"Accuracy": 57.6, "F1": 54.23},
            },
        }
    
        data = clf_results[market]
        metrics = ["Accuracy", "F1"]
    
        fig_clf = go.Figure()
    
        # Stage 1 bars
        fig_clf.add_trace(go.Bar(
            name="Stage 1",
            x=metrics,
            y=[data["Stage 1"][m] for m in metrics],
            marker_color="#26a69a",  # teal
        ))
    
        # Stage 2 bars
        fig_clf.add_trace(go.Bar(
            name="Stage 2",
            x=metrics,
            y=[data["Stage 2"][m] for m in metrics],
            marker_color="#f5a623",  # orange
        ))
    
        fig_clf.update_layout(
            hoverlabel=hover_style,
            title=f"Next-day Direction Classification (UP/DOWN) — {market}",
            template="plotly_dark",
            barmode="group",
            hovermode="x unified",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(title="Score"),
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",  # anchor to right
            x=1,              # push to far right
        ),        )
    
        st.plotly_chart(fig_clf, use_container_width=True)





























# actual vs predicted
sp500_preds = pd.read_csv('XGBoost_SP500_Stage2_predictions.csv', parse_dates=["Date"], index_col="Date")
ftse_100_preds = pd.read_csv('XGBoost_FTSE100_Stage2_predictions.csv', parse_dates=["Date"], index_col="Date")

# Filter predictions by selected market
preds = sp500_preds if market == "S&P 500" else ftse_100_preds
with col2:
    with st.container(border=True):
        st.markdown("### XGBoost — Actual vs Predicted Returns (Technical & Bahevioural Indicators)")
    
        fig_pred = go.Figure()
    
        # Actual return — solid blue line
        fig_pred.add_trace(go.Scatter(
            x=preds.index,
            y=preds["actual"],
            name="Actual",
            line=dict(color="#4c9be8", width=1.2),
        ))
    
        # Predicted return — dashed yellow line
        fig_pred.add_trace(go.Scatter(
            x=preds.index,
            y=preds["predicted"],
            name="Predicted",
            line=dict(color="#f5a623", width=1.2, dash="dash"),
        ))
    
        # Zero reference line
        fig_pred.add_hline(
            y=0,
            line_dash="dot",
            line_color="grey",
            line_width=1,
            annotation_text="Zero",
            annotation_position="bottom right",
        )
    
        fig_pred.update_layout(
            hoverlabel=hover_style,
            title=f"XGBoost Stage 2 — Actual vs Predicted ({market})",
            template="plotly_dark",
            hovermode="x unified",
            height=735,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(
                title="Date",
                rangebreaks=[dict(bounds=["sat", "mon"])],  # remove weekends
            ),
            yaxis=dict(title="Next Day Return (%)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        st.markdown(
    "<p style='font-size:22px; font-weight:bold;'>Prediction target: Next-day return (%)</p>",
    unsafe_allow_html=True
)
        
