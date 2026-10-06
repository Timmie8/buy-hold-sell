import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Importeer de StockAnalyzer als deze aanwezig is, anders fallback
try:
    from ta_engine import StockAnalyzer
    has_ta_engine = True
except ImportError:
    has_ta_engine = False

# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Live Trading Dashboard & Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Live Trading Dashboard & Multi-Timeframe Scanner")

# Initialize Analyzer indien beschikbaar
if has_ta_engine:
    analyzer = StockAnalyzer()

# ─────────────────────────────────────────────────────────────
# SIDEBAR / INSTELLINGEN
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Instellingen Scanner & Dashboard")

# Ticker selectie voor de Scanner
default_tickers = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, INTC, PLTR"
ticker_input = st.sidebar.text_area(
    "Scanner Tickers (gescheiden door komma)", default_tickers, height=100
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Grafiek / AI Single Stock View")
selected_ticker = st.sidebar.text_input("Gedetailleerde Analyse Ticker", value="AAPL").upper()
timeframe = st.sidebar.selectbox("Timeframe Grafiek", options=["1d", "15m", "5m"], index=0)
period = st.sidebar.selectbox("Historie Periode", options=["1y", "6mo", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("🤖 ML & Tuning")
forecast_horizon = st.sidebar.slider("ML Voorspellingshorizon (candles)", 1, 10, 3)
enable_grid_search = st.sidebar.checkbox("Schakel GridSearchCV in", value=True)

SPY_TICKER = "SPY"
ATR_MAX_PCT = 3.0

# ─────────────────────────────────────────────────────────────
# BEREKENINGEN & INDICATOREN
# ─────────────────────────────────────────────────────────────
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Berekent de RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_trading_score(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent de indicatoren en 5 regels exact zoals in Pine Script v5 + RSI Breakout."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Moving Averages
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA15"] = df["Close"].ewm(span=15, adjust=False).mean()

    # VWAP
    df["HLC3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["PV"] = df["HLC3"] * df["Volume"]
    dates = df.index.date
    cum_pv = df.groupby(dates)["PV"].cumsum()
    cum_vol = df.groupby(dates)["Volume"].cumsum()
    df["VWAP"] = np.where(cum_vol != 0, cum_pv / cum_vol, df["HLC3"])

    # Volume SMA 20
    df["VolSMA20"] = df["Volume"].rolling(window=20).mean()

    # RSI
    df["RSI"] = calculate_rsi(df["Close"], 14)

    # Regels
    df["Rule1"] = (df["EMA5"] > df["EMA15"]).astype(int)
    df["Rule2"] = (df["Close"] > df["VWAP"]).astype(int)
    df["Rule3"] = (df["Volume"] > df["VolSMA20"]).astype(int)
    df["Rule4"] = (df["Close"] > df["EMA5"]).astype(int)
    df["Rule5"] = (df["EMA15"] > df["EMA15"].shift(1)).astype(int)

    df["Score"] = df["Rule1"] + df["Rule2"] + df["Rule3"] + df["Rule4"] + df["Rule5"]

    # RSI Breakout detectie (> 55 en net gekruist)
    df["RSI_Cross_55"] = (df["RSI"] > 55) & (df["RSI"].shift(1) <= 55)

    return df

def calculate_stable_rs_score(df_stock: pd.DataFrame, df_spy: pd.DataFrame, atr_max_pct: float = 3.0) -> int:
    """Berekent de Stable Relative Strength Score (0 - 100)."""
    try:
        combined = pd.DataFrame({
            "stock_close": df_stock["Close"],
            "stock_high": df_stock["High"],
            "stock_low": df_stock["Low"],
            "stock_volume": df_stock["Volume"],
            "spy_close": df_spy["Close"]
        }).dropna()

        if len(combined) < 22:
            return 0

        stock_ret = combined["stock_close"] / combined["stock_close"].shift(1)
        spy_ret = combined["spy_close"] / combined["spy_close"].shift(1)
        rs_ratio = stock_ret / spy_ret

        latest_rs = rs_ratio.iloc[-1]
        rs_score = 35 if latest_rs >= 1.0 else (20 if latest_rs >= 0.99 else 0)

        ema9 = combined["stock_close"].ewm(span=9, adjust=False).mean()
        ema21 = combined["stock_close"].ewm(span=21, adjust=False).mean()
        close_last = combined["stock_close"].iloc[-1]

        if close_last > ema9.iloc[-1] and ema9.iloc[-1] > ema21.iloc[-1]:
            trend_score = 25
        elif close_last > ema21.iloc[-1]:
            trend_score = 15
        else:
            trend_score = 0

        prev_close = combined["stock_close"].shift(1)
        tr1 = combined["stock_high"] - combined["stock_low"]
        tr2 = (combined["stock_high"] - prev_close).abs()
        tr3 = (combined["stock_low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        atr_pct = (atr14.iloc[-1] / close_last) * 100

        vol_score = 25 if atr_pct <= atr_max_pct else (15 if atr_pct <= (atr_max_pct + 1.5) else 0)

        sma_vol20 = combined["stock_volume"].rolling(window=20).mean()
        rvol = (combined["stock_volume"] / sma_vol20).iloc[-1]
        rvol_score = 15 if rvol >= 1.2 else (10 if rvol >= 1.0 else 0)

        return int(rs_score + trend_score + vol_score + rvol_score)
    except Exception:
        return 0

def calculate_daily_rvol_and_diff(df_stock: pd.DataFrame):
    """Berekent de RVOL score en de afwijking t.o.v. het 20-daags gemiddelde."""
    try:
        if len(df_stock) < 20:
            return 0, "N/A"

        avg_vol_20d = df_stock["Volume"].rolling(window=20).mean().iloc[-1]
        latest_vol = float(df_stock["Volume"].iloc[-1])

        if avg_vol_20d == 0:
            return 0, "N/A"

        rvol = latest_vol / avg_vol_20d
        rvol_score = 100 if rvol >= 3.0 else (75 if rvol >= 2.0 else (50 if rvol >= 1.5 else (25 if rvol >= 1.0 else 0)))

        diff_pct = ((latest_vol - avg_vol_20d) / avg_vol_20d) * 100
        return rvol_score, f"{diff_pct:+.1f}%"
    except Exception:
        return 0, "N/A"

def get_signal_badge(score: int) -> str:
    """Vertaalt de score naar een compact signaal."""
    mapping = {
        5: "🚀 Strong Buy (5)",
        4: "📈 Buy (4)",
        3: "⚖️ Hold (3)",
        2: "📉 Sell (2)",
        1: "🔴 Strong Sell (1)",
        0: "🔴 Strong Sell (0)",
    }
    return mapping.get(score, "N/A")

# ─────────────────────────────────────────────────────────────
# SECTION 1: SCANNER TABEL BEREKENEN
# ─────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if st.sidebar.button("🚀 Run Multi-Timeframe Scan", type="primary") or "scanned" not in st.session_state:
    st.session_state["scanned"] = True
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        spy_df_daily = yf.download(SPY_TICKER, period="60d", interval="1d", progress=False)
        if isinstance(spy_df_daily.columns, pd.MultiIndex):
            spy_df_daily.columns = spy_df_daily.columns.get_level_values(0)
    except Exception:
        spy_df_daily = pd.DataFrame()

    timeframes = [("1d", "60d", "1D"), ("1h", "60d", "1H"), ("15m", "7d", "15M")]

    for i, ticker_item in enumerate(tickers):
        status_text.text(f"Bezig met analyseren van {ticker_item}...")
        ticker_data = {"Ticker": ticker_item, "Prijs ($)": "N/A"}
        total_score_sum = 0
        stock_daily_df = pd.DataFrame()

        for interval, period_tf, label in timeframes:
            try:
                data = yf.download(ticker_item, period=period_tf, interval=interval, progress=False)

                if not data.empty and len(data) >= 20:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    if interval == "1d":
                        stock_daily_df = data.copy()

                    df_calc = calculate_trading_score(data)
                    latest = df_calc.iloc[-1]

                    score_val = int(latest["Score"])
                    
                    # Controleer op RSI Breakout
                    rsi_breakout = latest.get("RSI_Cross_55", False) or (latest.get("RSI", 0) > 55 and df_calc["RSI"].iloc[-2] <= 55)
                    
                    # Gele driehoek toevoegen bij breakout
                    score_display = f"⚠️ {score_val}" if rsi_breakout else f"{score_val}"

                    ticker_data["Prijs ($)"] = round(float(latest["Close"]), 2)
                    ticker_data[f"Score {label}"] = score_display
                    ticker_data[f"Signaal {label}"] = get_signal_badge(score_val)
                    total_score_sum += score_val
                else:
                    ticker_data[f"Score {label}"] = "0"
                    ticker_data[f"Signaal {label}"] = "Geen data"
            except Exception:
                ticker_data[f"Score {label}"] = "0"
                ticker_data[f"Signaal {label}"] = "Fout"

        if not stock_daily_df.empty and not spy_df_daily.empty:
            ticker_data["RS Score (0-100)"] = calculate_stable_rs_score(stock_daily_df, spy_df_daily, ATR_MAX_PCT)
        else:
            ticker_data["RS Score (0-100)"] = 0

        if not stock_daily_df.empty:
            rvol_daily_score, volume_diff = calculate_daily_rvol_and_diff(stock_daily_df)
            ticker_data["RVOL 1D Score"] = rvol_daily_score
            ticker_data["Volume vs Gem. (1D)"] = volume_diff
        else:
            ticker_data["RVOL 1D Score"] = 0
            ticker_data["Volume vs Gem. (1D)"] = "N/A"

        ticker_data["Totale Matrix Score"] = total_score_sum
        results.append(ticker_data)

        progress_bar.progress((i + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(by="Totale Matrix Score", ascending=False)
        display_df = res_df.drop(columns=["Totale Matrix Score"])

        def highlight_scores(val):
            val_str = str(val)
            clean_val = int(val_str.replace("⚠️", "").strip()) if val_str.replace("⚠️", "").strip().isdigit() else 0
            
            style = ""
            if clean_val >= 4:
                style = "background-color: #28a745; color: white; font-weight: bold;"
            elif clean_val == 3:
                style = "background-color: #ffc107; color: black; font-weight: bold;"
            else:
                style = "background-color: #dc3545; color: white; font-weight: bold;"
                
            if "⚠️️" in val_str:
                style += " border: 2px solid #ffcc00;"
            return style

        def highlight_rs_score(val):
            if isinstance(val, int):
                if val >= 70:
                    return "background-color: #28a745; color: white; font-weight: bold;"
                elif val >= 50:
                    return "background-color: #fd7e14; color: white; font-weight: bold;"
                else:
                    return "background-color: #dc3545; color: white; font-weight: bold;"
            return ""

        def highlight_rvol_score(val):
            if isinstance(val, int):
                if val >= 75:
                    return "background-color: #28a745; color: white; font-weight: bold;"
                elif val >= 50:
                    return "background-color: #ffc107; color: black; font-weight: bold;"
                elif val >= 25:
                    return "background-color: #fd7e14; color: white; font-weight: bold;"
                else:
                    return "background-color: #dc3545; color: white; font-weight: bold;"
            return ""

        def highlight_volume_diff(val):
            if isinstance(val, str) and val.endswith("%"):
                if val.startswith("+"):
                    return "background-color: #28a745; color: white; font-weight: bold;"
                elif val.startswith("-"):
                    return "background-color: #dc3545; color: white; font-weight: bold;"
            return ""

        st.subheader("📋 Multi-Timeframe Score Overzicht")
        st.caption("⚠️ = RSI Breakout detectie op dit tijdsframe (> 55 gekruist)")
        st.dataframe(
            display_df.style
            .map(highlight_scores, subset=["Score 1D", "Score 1H", "Score 15M"])
            .map(highlight_rs_score, subset=["RS Score (0-100)"])
            .map(highlight_rvol_score, subset=["RVOL 1D Score"])
            .map(highlight_volume_diff, subset=["Volume vs Gem. (1D)"]),
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────────────────────────────────────────
# SECTION 2: AI LIVE TRADING DASHBOARD (GESELECTEERDE TICKER)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header(f"📊 AI Live Trading Dashboard: {selected_ticker}")

if has_ta_engine:
    df_single = analyzer.get_stock_data(symbol=selected_ticker, timeframe=timeframe, period=period)
else:
    # Fallback m.b.v. yfinance
    df_single = yf.download(selected_ticker, period=period, interval=timeframe, progress=False)
    if not df_single.empty:
        if isinstance(df_single.columns, pd.MultiIndex):
            df_single.columns = df_single.columns.get_level_values(0)
        df_single["Timestamp"] = df_single.index
        df_single["RSI"] = calculate_rsi(df_single["Close"])

if df_single.empty:
    st.error(f"Geen data gevonden voor ticker '{selected_ticker}'. Controleer het symbool.")
else:
    if has_ta_engine:
        signals = analyzer.evaluate_signals(df_single)
        ml_res = analyzer.predict_ml_probability(
            df_single, 
            forecast_horizon=forecast_horizon, 
            use_grid_search=enable_grid_search
        )
    else:
        # Dummy/Fallback waarden indien ta_engine niet lokaal is geïnstalleerd
        rsi_val = round(float(df_single["RSI"].iloc[-1]), 2) if "RSI" in df_single else 50.0
        close_val = float(df_single["Close"].iloc[-1])
        prev_close = float(df_single["Close"].iloc[-2]) if len(df_single) > 1 else close_val
        change_pct = round(((close_val - prev_close) / prev_close) * 100, 2)

        signals = {
            "Price": close_val,
            "Change_Pct": change_pct,
            "RSI": rsi_val,
            "RSI_Overbought_Warning": rsi_val > 70 and df_single["RSI"].iloc[-1] < df_single["RSI"].iloc[-2],
            "RSI_Stijgend_Boven_70": rsi_val > 70 and df_single["RSI"].iloc[-1] >= df_single["RSI"].iloc[-2],
            "RSI_Above_55": rsi_val > 55,
            "RSI_Cross_55": rsi_val > 55 and df_single["RSI"].iloc[-2] <= 55,
            "STO_Status": "N/A",
            "MACD_Status": "N/A",
            "Action": "NEUTRAAL",
            "Reasons": ["Geen ta_engine.py geladen, basis yfinance analyse gebruikt."]
        }
        ml_res = {"up_prob": 50, "best_params": {}, "feature_importances": {"RSI": 1.0}}

    # 1. Status & kleur voor de RSI badge
    if signals.get('RSI_Overbought_Warning', False):
        rsi_status_text = f"OVERBOUGHT ({signals['RSI']})"
        bg_color = "#FF4B4B"
        text_color = "#FFFFFF"
    elif signals.get('RSI_Stijgend_Boven_70', False):
        rsi_status_text = f"🚀 STRONG > 70 ({signals['RSI']})"
        bg_color = "#28A745"
        text_color = "#FFFFFF"
    elif signals.get('RSI_Above_55', False):
        if signals.get('RSI_Cross_55', False):
            rsi_status_text = f"🔥 BREAKOUT > 55 ({signals['RSI']})"
        else:
            rsi_status_text = f"BULLISH > 55 ({signals['RSI']})"
        bg_color = "#28A745"
        text_color = "#FFFFFF"
    else:
        rsi_status_text = f"BEARISH < 55 ({signals['RSI']})"
        bg_color = "#6C757D"
        text_color = "#FFFFFF"

    # 2. Live Dashboard Balk
    st.markdown("### 📊 Live Dashboard & Signalen")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.metric("Laatste Prijs", f"${signals['Price']:.2f}", f"{signals['Change_Pct']}%")

    with m2:
        st.caption("RSI (14) Status")
        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                color: {text_color};
                padding: 8px 10px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 13px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                {rsi_status_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.metric("Slow-STO", signals['STO_Status'])

    with m4:
        st.metric("MACD", signals['MACD_Status'])

    with m5:
        st.metric("Advies Signaal", signals['Action'])

    with m6:
        st.metric("🤖 ML Kans (+{f}d)".format(f=forecast_horizon), f"{ml_res['up_prob']}%")

    st.markdown("---")

    # 3. Technische Grafiek & Indicatoren
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=('Koers & Candlesticks', 'Volume', 'RSI Indicator'),
        row_width=[0.2, 0.2, 0.6]
    )

    x_axis = df_single['Timestamp'] if 'Timestamp' in df_single else df_single.index

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=x_axis, open=df_single['Open'], high=df_single['High'], 
        low=df_single['Low'], close=df_single['Close'], name='Koers'
    ), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(
        x=x_axis, y=df_single['Volume'], name='Volume', 
        marker_color='lightblue'
    ), row=2, col=1)

    # RSI + Hulplijnen (30, 55, 70)
    fig.add_trace(go.Scatter(
        x=x_axis, y=df_single['RSI'], mode='lines', 
        name='RSI', line=dict(color='purple')
    ), row=3, col=1)

    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=55, line_dash="dash", line_color="green", annotation_text="Breakout (55)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=750, showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 4. ML Details & Onderbouwing
    col_reasons, col_ml = st.columns(2)

    with col_reasons:
        with st.expander("📋 Signaal Onderbouwing", expanded=True):
            for r in signals['Reasons']:
                st.write(r)

    with col_ml:
        with st.expander("🤖 Machine Learning Model Details", expanded=True):
            st.write(f"**Voorspelde kans op prijsstijging:** `{ml_res['up_prob']}%`")
            st.progress(ml_res['up_prob'] / 100)

            if enable_grid_search and "best_params" in ml_res and ml_res["best_params"]:
                st.markdown("**Gevonden Optimale Parameters (GridSearch):**")
                st.json(ml_res["best_params"])

            if 'feature_importances' in ml_res:
                st.markdown("**Top Gewichten Indicatoren:**")
                feat_df = pd.DataFrame(
                    list(ml_res['feature_importances'].items()), 
                    columns=['Indicator', 'Gewicht']
                ).sort_values(by='Gewicht', ascending=False)
                st.dataframe(feat_df.head(4), use_container_width=True, hide_index=True)
