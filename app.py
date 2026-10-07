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

# Session State initialisatie voor geselecteerde ticker
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "AAPL"

# ─────────────────────────────────────────────────────────────
# CONSTANTEN
# ─────────────────────────────────────────────────────────────
SPY_TICKER = "SPY"
ATR_MAX_PCT = 3.0
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}

# yfinance limieten
INTERVAL_PERIODS = {
    "1d":  ["1y", "6mo", "1mo"],
    "1h":  ["60d", "1mo", "7d"],
    "15m": ["1mo", "7d", "5d"],
    "5m":  ["1mo", "7d", "5d"],
    "1m":  ["7d", "5d", "1d"],
}

# ─────────────────────────────────────────────────────────────
# HULPFUNCTIES
# ─────────────────────────────────────────────────────────────
def style_map(styler, func, subset=None):
    """pandas >= 2.1 gebruikt Styler.map, oudere versies Styler.applymap."""
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maakt van een MultiIndex-kolomstructuur (yfinance) platte kolommen."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def download_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download met cache. Ongeldige combinaties geven een lege DataFrame."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return normalize_columns(df)
    except Exception:
        return pd.DataFrame()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI met Wilder-smoothing (EMA alpha = 1/period)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_trading_score(
    df: pd.DataFrame,
    interval: str = "1d",
    res_lookback: int = 10,
    vwap_rolling_days: int = 20,
) -> pd.DataFrame:
    """Indicatoren + Slow Stochastic + MACD + RSI breakout en regels."""
    if df is None or df.empty:
        raise ValueError("Geen data om indicatoren op te berekenen.")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Kolommen ontbreken in de data: {sorted(missing)}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Moving Averages
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA15"] = df["Close"].ewm(span=15, adjust=False).mean()

    # VWAP
    df["HLC3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["PV"] = df["HLC3"] * df["Volume"]
    is_daily = interval.lower() in ("1d", "1wk", "1mo")
    if is_daily:
        roll_pv = df["PV"].rolling(window=vwap_rolling_days, min_periods=1).sum()
        roll_vol = df["Volume"].rolling(window=vwap_rolling_days, min_periods=1).sum()
        df["VWAP"] = np.where(roll_vol != 0, roll_pv / roll_vol, df["HLC3"])
    else:
        dates = df.index.date
        cum_pv = df.groupby(dates)["PV"].cumsum()
        cum_vol = df.groupby(dates)["Volume"].cumsum()
        df["VWAP"] = np.where(cum_vol != 0, cum_pv / cum_vol, df["HLC3"])

    # Volume SMA 20
    df["VolSMA20"] = df["Volume"].rolling(window=20).mean()

    # RSI
    df["RSI"] = calculate_rsi(df["Close"], 14)

    # --- SLOW STOCHASTIC (14, 3, 3) ---
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    fast_k = 100 * ((df["Close"] - low_14) / (high_14 - low_14).replace(0, np.nan))
    df["Slow_K"] = fast_k.rolling(window=3).mean().fillna(50.0)  # %K
    df["Slow_D"] = df["Slow_K"].rolling(window=3).mean().fillna(50.0)  # %D
    df["Slow_Sto_Bullish"] = df["Slow_K"] > df["Slow_D"]

    # --- MACD (12, 26, 9) ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Bullish"] = df["MACD"] > df["MACD_Signal"]

    # Weerstand over N candles + 10% ruimte
    df["Res_ND"] = df["High"].rolling(window=res_lookback, min_periods=1).max()
    df["Res_ND_10Pct"] = df["Res_ND"] * 1.10

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
            "spy_close": df_spy["Close"],
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

        atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
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

        if not np.isfinite(avg_vol_20d) or avg_vol_20d == 0:
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


def fallback_ml_probability(df: pd.DataFrame, horizon: int = 3):
    empty = {"up_prob": None, "feature_importances": {}, "best_params": {},
             "note": "Onvoldoende data of sklearn ontbreekt voor de interne fallback."}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return empty

    feat = pd.DataFrame(index=df.index)
    feat["RSI"] = df["RSI"]
    feat["EMA5_EMA15"] = (df["EMA5"] - df["EMA15"]) / df["Close"]
    feat["Boven_VWAP"] = (df["Close"] > df["VWAP"]).astype(int)
    feat["Volume_Ratio"] = df["Volume"] / df["VolSMA20"]
    feat["Rendement_1"] = df["Close"].pct_change()
    feat["Rendement_5"] = df["Close"].pct_change(5)

    target = (df["Close"].shift(-horizon) > df["Close"]).astype(int)

    train = feat.copy()
    train["_y"] = target
    train = train.replace([np.inf, -np.inf], np.nan).dropna()

    latest_row = feat.replace([np.inf, -np.inf], np.nan).dropna()

    if len(train) < 60 or train["_y"].nunique() < 2 or latest_row.empty:
        return empty

    X, y = train.drop(columns="_y"), train["_y"]
    split = max(int(len(train) * 0.8), 30)
    if split >= len(train):
        split = len(train) - 1

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X.iloc[:split], y.iloc[:split])

    prob = float(model.predict_proba(latest_row.iloc[[-1]])[0][1])
    coefs = np.abs(model[-1].coef_[0])
    total = coefs.sum() if coefs.sum() > 0 else 1.0
    importances = {col: round(float(c / total), 3) for col, c in zip(X.columns, coefs)}

    return {
        "up_prob": int(round(prob * 100)),
        "feature_importances": importances,
        "best_params": {},
        "note": "Interne fallback (logistische regressie) — ta_engine.py niet geladen.",
    }


def run_scan(tickers, res_lookback, atr_max_pct, progress_cb=None) -> pd.DataFrame:
    spy_df_daily = download_data(SPY_TICKER, "60d", "1d")

    timeframes = [("1d", "60d", "1D"), ("1h", "60d", "1H"), ("15m", "7d", "15M")]
    results = []

    for i, ticker_item in enumerate(tickers):
        if progress_cb:
            progress_cb(i, len(tickers), ticker_item)

        ticker_data = {"Ticker": ticker_item, "Prijs ($)": "N/A"}
        total_score_sum = 0
        stock_daily_df = pd.DataFrame()

        for interval, period_tf, label in timeframes:
            data = download_data(ticker_item, period_tf, interval)

            if not data.empty and len(data) >= 20:
                if interval == "1d":
                    stock_daily_df = data.copy()

                try:
                    df_calc = calculate_trading_score(data, interval=interval, res_lookback=res_lookback)
                except ValueError:
                    ticker_data[f"Score {label}"] = "0"
                    ticker_data[f"Signaal {label}"] = "Fout"
                    continue

                latest = df_calc.iloc[-1]
                score_val = int(latest["Score"])

                rsi_breakout = bool(latest.get("RSI_Cross_55", False))
                score_display = f"⚠️ {score_val}" if rsi_breakout else f"{score_val}"

                ticker_data["Prijs ($)"] = round(float(latest["Close"]), 2)
                ticker_data[f"Score {label}"] = score_display
                ticker_data[f"Signaal {label}"] = get_signal_badge(score_val)
                total_score_sum += score_val
            else:
                ticker_data[f"Score {label}"] = "0"
                ticker_data[f"Signaal {label}"] = "Geen data"

        # --- WEERSTAND SCORE TOT DE TOP BEREKENING (OP DAGLEVEL) ---
        if not stock_daily_df.empty:
            if ticker_item == "QDEL":
                res_val = 15.50
            else:
                res_val = float(stock_daily_df["High"].rolling(window=res_lookback, min_periods=1).max().iloc[-1])
            
            latest_close = float(stock_daily_df["Close"].iloc[-1])
            dist_to_res_pct = ((res_val - latest_close) / latest_close) * 100 if latest_close else 0.0
            ticker_data["Weerstand tot top"] = f"{dist_to_res_pct:+.1f}%"
        else:
            ticker_data["Weerstand tot top"] = "N/A"

        if not stock_daily_df.empty and not spy_df_daily.empty:
            ticker_data["RS Score (0-100)"] = calculate_stable_rs_score(stock_daily_df, spy_df_daily, atr_max_pct)
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

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results).sort_values(by="Totale Matrix Score", ascending=False)
    return res_df.drop(columns=["Totale Matrix Score"]).reset_index(drop=True)


def highlight_scores(val):
    val_str = str(val)
    clean = val_str.replace("⚠️", "").strip()
    clean_val = int(clean) if clean.isdigit() else 0

    if clean_val >= 4:
        style = "background-color: #28a745; color: white; font-weight: bold;"
    elif clean_val == 3:
        style = "background-color: #ffc107; color: black; font-weight: bold;"
    else:
        style = "background-color: #dc3545; color: white; font-weight: bold;"

    if "⚠️" in val_str:
        style += " border: 2px solid #ffcc00;"
    return style


def _is_int(val) -> bool:
    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


def highlight_rs_score(val):
    if _is_int(val):
        if val >= 70:
            return "background-color: #28a745; color: white; font-weight: bold;"
        elif val >= 50:
            return "background-color: #fd7e14; color: white; font-weight: bold;"
        else:
            return "background-color: #dc3545; color: white; font-weight: bold;"
    return ""


def highlight_rvol_score(val):
    if _is_int(val):
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


def highlight_resistance(val):
    """Geeft een groene achtergrond als het percentage naar de weerstand groter is dan 6%."""
    if isinstance(val, str) and val.endswith("%"):
        try:
            clean_val = float(val.replace("%", "").replace("+", "").strip())
            if clean_val > 6.0:
                return "background-color: #28a745; color: white; font-weight: bold;"
        except ValueError:
            pass
    return ""


# ─────────────────────────────────────────────────────────────
# SIDEBAR / INSTELLINGEN
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Instellingen Scanner & Dashboard")

default_tickers = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, INTC, PLTR, QDEL"
ticker_input = st.sidebar.text_area(
    "Scanner Tickers (gescheiden door komma)", default_tickers, height=100
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Grafiek / AI Single Stock View")

if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "AAPL"

if "ticker_widget" not in st.session_state:
    st.session_state["ticker_widget"] = st.session_state["selected_ticker"]

if st.session_state.pop("_apply_clicked_ticker", False):
    st.session_state["ticker_widget"] = st.session_state["selected_ticker"]

st.sidebar.text_input("Gedetailleerde Analyse Ticker", key="ticker_widget")

if st.session_state["ticker_widget"].strip().upper() != st.session_state["selected_ticker"]:
    st.session_state["selected_ticker"] = st.session_state["ticker_widget"].strip().upper()

timeframe = st.sidebar.selectbox("Timeframe Grafiek", options=["1d", "1h", "15m", "5m"], index=0)
valid_periods = INTERVAL_PERIODS.get(timeframe, ["1y", "6mo", "1mo"])
period = st.sidebar.selectbox("Historie Periode", options=valid_periods, index=0)
st.sidebar.caption(
    f"ℹ️ yfinance-limiet: voor interval **{timeframe}** is **{valid_periods[0]}** "
    f"de langste geldige periode."
)

st.sidebar.markdown("---")
st.sidebar.header("🤖 ML & Tuning")
forecast_horizon = st.sidebar.slider("ML Voorspellingshorizon (candles)", 1, 10, 3)
enable_grid_search = st.sidebar.checkbox("Schakel GridSearchCV in", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Indicatoren")
res_lookback = st.sidebar.number_input(
    "Weerstand lookback (candles)", min_value=5, max_value=100, value=10, step=1,
)
vwap_rolling_days = st.sidebar.number_input(
    "VWAP rollend venster op dagdata (dagen)", min_value=5, max_value=60, value=20, step=1,
)

# ─────────────────────────────────────────────────────────────
# SECTION 1: SCANNER TABEL BEREKENEN & INTERACTIE
# ─────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if st.sidebar.button("🚀 Run Multi-Timeframe Scan", type="primary") or not st.session_state.get("scanned", False):
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def _cb(i, total, tkr):
        status_text.text(f"Bezig met analyseren van {tkr}...")
        progress_bar.progress((i + 1) / total)

    try:
        st.session_state["scan_results"] = run_scan(tickers, res_lookback, ATR_MAX_PCT, progress_cb=_cb)
        st.session_state["scanned"] = True
    except Exception as e:
        st.session_state["scan_results"] = pd.DataFrame()
        st.session_state["scanned"] = True
        st.sidebar.error(f"Scan mislukt: {e}")

    status_text.empty()
    progress_bar.empty()

scan_results = st.session_state.get("scan_results")

if st.session_state.get("scanned") and scan_results is not None:
    if scan_results.empty:
        st.warning("Geen resultaten. Controleer de tickers en probeer opnieuw.")
    else:
        st.subheader("📋 Multi-Timeframe Score Overzicht")
        st.caption("💡 **Tip:** Klik op een regel in de tabel om de grafiek en AI-analyse eronder direct te laden.")

        styler = scan_results.style
        styler = style_map(styler, highlight_scores, subset=["Score 1D", "Score 1H", "Score 15M"])
        styler = style_map(styler, highlight_rs_score, subset=["RS Score (0-100)"])
        styler = style_map(styler, highlight_rvol_score, subset=["RVOL 1D Score"])
        styler = style_map(styler, highlight_volume_diff, subset=["Volume vs Gem. (1D)"])
        styler = style_map(styler, highlight_resistance, subset=["Weerstand tot top"])

        selected_event = st.dataframe(
            styler,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="scan_table",
        )

        if selected_event and selected_event.selection.rows:
            clicked_ticker = scan_results.iloc[selected_event.selection.rows[0]]["Ticker"]
            if clicked_ticker != st.session_state["selected_ticker"]:
                st.session_state["selected_ticker"] = clicked_ticker
                st.session_state["_apply_clicked_ticker"] = True
                st.rerun()

# ─────────────────────────────────────────────────────────────
# SECTION 2: AI LIVE TRADING DASHBOARD (GESELECTEERDE TICKER)
# ─────────────────────────────────────────────────────────────
selected_ticker = st.session_state["selected_ticker"]

st.markdown("---")
st.header(f"📊 AI Live Trading Dashboard: {selected_ticker}")

if has_ta_engine:
    try:
        df_single = analyzer.get_stock_data(symbol=selected_ticker, timeframe=timeframe, period=period)
        df_single = normalize_columns(df_single)
    except Exception as e:
        df_single = pd.DataFrame()
        st.warning(f"ta_engine kon geen data ophalen: {e}")
else:
    df_single = download_data(selected_ticker, period, timeframe)
    if not df_single.empty:
        df_single["Timestamp"] = df_single.index

if df_single.empty:
    st.error(f"Geen data gevonden voor ticker '{selected_ticker}' op {timeframe}/{period}. Controleer het symbool.")
else:
    df_single = calculate_trading_score(
        df_single, interval=timeframe, res_lookback=int(res_lookback), vwap_rolling_days=int(vwap_rolling_days)
    )

    if has_ta_engine:
        signals = analyzer.evaluate_signals(df_single)
        ml_res = analyzer.predict_ml_probability(
            df_single,
            forecast_horizon=forecast_horizon,
            use_grid_search=enable_grid_search,
        )
    else:
        # Uitlezen van indicatoren en hun status op de laatste candle
        rsi_val = round(float(df_single["RSI"].iloc[-1]), 2)
        slow_k = float(df_single["Slow_K"].iloc[-1])
        slow_d = float(df_single["Slow_D"].iloc[-1])
        macd = float(df_single["MACD"].iloc[-1])
        macd_signal = float(df_single["MACD_Signal"].iloc[-1])

        # Bepalen van individuele indicator trends
        rsi_bullish = rsi_val > 50
        rsi_breakout = rsi_val > 55
        slow_sto_bullish = bool(df_single["Slow_Sto_Bullish"].iloc[-1])
        macd_bullish = bool(df_single["MACD_Bullish"].iloc[-1])

        # Hulpvariabelen voor statustekst
        sto_status_str = f"🟢 BULLISH ({slow_k:.1f})" if slow_sto_bullish else f"🔴 BEARISH ({slow_k:.1f})"
        macd_status_str = f"🟢 BULLISH" if macd_bullish else f"🔴 BEARISH"

        close_val = float(df_single["Close"].iloc[-1])
        prev_close = float(df_single["Close"].iloc[-2]) if len(df_single) > 1 else close_val
        change_pct = round(((close_val - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

        # --- NIEUWE LOGICA VOOR SIGNAAL ---
        reasons = []
        if rsi_bullish and slow_sto_bullish and macd_bullish:
            if rsi_breakout:
                action_signal = "🚀 STRONG BUY"
                reasons.append("RSI (>55 Breakout), Slow-STO (%K > %D) en MACD zijn allemaal BULLISH!")
            else:
                action_signal = "📈 BUY"
                reasons.append("RSI (>50), Slow-STO (%K > %D) en MACD zijn allemaal BULLISH.")
        else:
            action_signal = "NEUTRAAL / NO TRADE"
            if not rsi_bullish:
                reasons.append("RSI is nog niet bullish (<= 50).")
            if not slow_sto_bullish:
                reasons.append("Slow Stochastic is bearish (%K < %D).")
            if not macd_bullish:
                reasons.append("MACD is bearish (MACD < Signal).")

        signals = {
            "Price": close_val,
            "Change_Pct": change_pct,
            "RSI": rsi_val,
            "RSI_Overbought_Warning": rsi_val > 70 and df_single["RSI"].iloc[-1] < df_single["RSI"].iloc[-2],
            "RSI_Stijgend_Boven_70": rsi_val > 70 and df_single["RSI"].iloc[-1] >= df_single["RSI"].iloc[-2],
            "RSI_Above_55": rsi_breakout,
            "RSI_Cross_55": bool(df_single["RSI_Cross_55"].iloc[-1]),
            "STO_Status": sto_status_str,
            "MACD_Status": macd_status_str,
            "Action": action_signal,
            "Reasons": reasons,
        }
        ml_res = fallback_ml_probability(df_single, forecast_horizon)

    # ─────────────────────────────────────────────────────────────
    # INSTELLINGEN/BEREKENING VASTE WEERSTANDSLIJN (Bv. QDEL op $15.50)
    # ─────────────────────────────────────────────────────────────
    latest_close = float(df_single["Close"].iloc[-1])

    if selected_ticker == "QDEL":
        res_val = 15.50
    else:
        res_val = float(df_single["Res_ND"].iloc[-1]) if "Res_ND" in df_single else latest_close

    # Percentage berekenen van huidige prijs naar weerstandslijn
    dist_to_res_pct = ((res_val - latest_close) / latest_close) * 100 if latest_close else 0.0
    res_10pct_val = res_val * 1.10

    # 1. Status & kleur voor de RSI badge
    if signals.get("RSI_Overbought_Warning", False):
        rsi_status_text = f"OVERBOUGHT ({signals['RSI']})"
        bg_color = "#FF4B4B"
        text_color = "#FFFFFF"
    elif signals.get("RSI_Stijgend_Boven_70", False):
        rsi_status_text = f"🚀 STRONG > 70 ({signals['RSI']})"
        bg_color = "#28A745"
        text_color = "#FFFFFF"
    elif signals.get("RSI_Above_55", False):
        if signals.get("RSI_Cross_55", False):
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
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

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
            unsafe_allow_html=True,
        )

    with m3:
        st.metric("Slow-STO", signals.get("STO_Status", "N/A"))

    with m4:
        st.metric("MACD", signals.get("MACD_Status", "N/A"))

    with m5:
        st.metric("Advies Signaal", signals.get("Action", "N/A"))

    with m6:
        st.metric("Weerstand Lijn", f"${res_val:.2f}", f"{dist_to_res_pct:+.1f}% tot top")

    with m7:
        prob = ml_res.get("up_prob")
        st.metric(
            f"🤖 ML Kans (+{forecast_horizon}d)",
            f"{prob}%" if prob is not None else "N/A",
        )

    st.markdown("---")

    # 3. Technische Grafiek & Indicatoren
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Koers & Candlesticks", "Volume", "RSI Indicator"),
        row_width=[0.2, 0.2, 0.6],
    )

    x_axis = df_single["Timestamp"] if "Timestamp" in df_single else df_single.index

    fig.add_trace(go.Candlestick(
        x=x_axis, open=df_single["Open"], high=df_single["High"],
        low=df_single["Low"], close=df_single["Close"], name="Koers",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis, y=df_single["VWAP"], mode="lines",
        name=f"VWAP ({'rollend ' + str(int(vwap_rolling_days)) + 'd' if timeframe == '1d' else 'sessie'})",
        line=dict(color="blue", width=1.2),
    ), row=1, col=1)

    # RECHTE HORIZONTALE WEERSTANDSLIJN + PERCENTAGE ANNOTATIE
    fig.add_hline(
        y=res_val,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Weerstand ${res_val:.2f} ({dist_to_res_pct:+.1f}%)",
        annotation_position="top right",
        row=1, col=1,
    )

    # RECHTE HORIZONTALE WEERSTAND +10% ZONE LIJN
    fig.add_hline(
        y=res_10pct_val,
        line_dash="dot",
        line_color="orange",
        line_width=1.5,
        annotation_text=f"Weerstand +10% (${res_10pct_val:.2f})",
        annotation_position="top right",
        row=1, col=1,
    )

    fig.add_trace(go.Bar(
        x=x_axis, y=df_single["Volume"], name="Volume", marker_color="lightblue",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis, y=df_single["RSI"], mode="lines", name="RSI", line=dict(color="purple"),
    ), row=3, col=1)

    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=55, line_dash="dash", line_color="green", annotation_text="Breakout (55)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=750, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")

    # 4. ML Details & Onderbouwing
    col_reasons, col_ml = st.columns(2)

    with col_reasons:
        with st.expander("📋 Signaal Onderbouwing", expanded=True):
            for r in signals.get("Reasons", []):
                st.write(r)
            st.write(f"- **Vaste Weerstand:** ${res_val:.2f} (Afstand: {dist_to_res_pct:+.2f}%)")
            st.write(f"- **Weerstand +10% zone:** ${res_10pct_val:.2f}")
            st.write(f"- **VWAP:** ${float(df_single['VWAP'].iloc[-1]):.2f}")

    with col_ml:
        with st.expander("🤖 Machine Learning Model Details", expanded=True):
            prob = ml_res.get("up_prob")
            if prob is None:
                st.info(ml_res.get("note", "Geen ML-voorspelling beschikbaar."))
            else:
                st.write(f"**Voorspelde kans op prijsstijging:** `{prob}%`")
                st.progress(prob / 100)

            if enable_grid_search and ml_res.get("best_params"):
                st.markdown("**Gevonden Optimale Parameters (GridSearch):**")
                st.json(ml_res["best_params"])

            if ml_res.get("feature_importances"):
                st.markdown("**Top Gewichten Indicatoren:**")
                feat_df = pd.DataFrame(
                    list(ml_res["feature_importances"].items()),
                    columns=["Indicator", "Gewicht"],
                ).sort_values(by="Gewicht", ascending=False)
                st.dataframe(feat_df.head(4), width="stretch", hide_index=True)
