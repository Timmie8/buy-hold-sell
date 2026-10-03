import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Timeframe Trading Score Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Multi-Timeframe Auto Trading Score Scanner")
st.caption(
    "Scant elk aandeel direct op 1D, 1H en 15M (MA5/15 + VWAP + Volume), Stable RS Score (0-100) én Dagelijkse RVOL Score (1D vs 20D gemiddelde)."
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR / INSTELLINGEN
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Instellingen")

# Standaard tickerlijst
default_tickers = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, INTC, PLTR"
ticker_input = st.sidebar.text_area(
    "Tickers (gescheiden door komma)", default_tickers, height=140
)

# Benchmark Ticker en instellingen voor RS Score
SPY_TICKER = "SPY"
ATR_MAX_PCT = 3.0


# ─────────────────────────────────────────────────────────────
# TRADING SCORE BEREKENING (EXACT TRADINGVIEW MATCH)
# ─────────────────────────────────────────────────────────────
def calculate_trading_score(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent de indicatoren en 5 regels exact zoals in Pine Script v5."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # 1. Moving Averages (EMA 5 & EMA 15)
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA15"] = df["Close"].ewm(span=15, adjust=False).mean()

    # 2. VWAP (Anchored per sessie/dag zoals ta.vwap in Pine Script)
    df["HLC3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["PV"] = df["HLC3"] * df["Volume"]

    # Cumulatieve som per unieke dag resetten
    dates = df.index.date
    cum_pv = df.groupby(dates)["PV"].cumsum()
    cum_vol = df.groupby(dates)["Volume"].cumsum()

    # Deel door cumulatief volume (met veiligheidscheck voor 0)
    df["VWAP"] = np.where(cum_vol != 0, cum_pv / cum_vol, df["HLC3"])

    # 3. Volume SMA 20
    df["VolSMA20"] = df["Volume"].rolling(window=20).mean()

    # ─────────────────────────────────────────────────────────
    # REGELS (0 of 1)
    # ─────────────────────────────────────────────────────────
    df["Rule1"] = (df["EMA5"] > df["EMA15"]).astype(int)
    df["Rule2"] = (df["Close"] > df["VWAP"]).astype(int)
    df["Rule3"] = (df["Volume"] > df["VolSMA20"]).astype(int)
    df["Rule4"] = (df["Close"] > df["EMA5"]).astype(int)
    df["Rule5"] = (df["EMA15"] > df["EMA15"].shift(1)).astype(int)

    # Totale Score (0 t/m 5)
    df["Score"] = (
        df["Rule1"] + df["Rule2"] + df["Rule3"] + df["Rule4"] + df["Rule5"]
    )

    return df


def calculate_stable_rs_score(df_stock: pd.DataFrame, df_spy: pd.DataFrame, atr_max_pct: float = 3.0) -> int:
    """Berekent de Stable Relative Strength Score (0 - 100) exact volgens het Pine Script."""
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

        # --- 1. RELATIEVE STERKTE VS SPY ---
        stock_ret = combined["stock_close"] / combined["stock_close"].shift(1)
        spy_ret = combined["spy_close"] / combined["spy_close"].shift(1)
        rs_ratio = stock_ret / spy_ret

        latest_rs = rs_ratio.iloc[-1]
        if latest_rs >= 1.0:
            rs_score = 35
        elif latest_rs >= 0.99:
            rs_score = 20
        else:
            rs_score = 0

        # --- 2. TREND & INTRADAY STABILITEIT ---
        ema9 = combined["stock_close"].ewm(span=9, adjust=False).mean()
        ema21 = combined["stock_close"].ewm(span=21, adjust=False).mean()

        close_last = combined["stock_close"].iloc[-1]
        ema9_last = ema9.iloc[-1]
        ema21_last = ema21.iloc[-1]

        if close_last > ema9_last and ema9_last > ema21_last:
            trend_score = 25
        elif close_last > ema21_last:
            trend_score = 15
        else:
            trend_score = 0

        # --- 3. VOLATILITEIT / DOWNSIDE PROTECTION (ATR 14) ---
        prev_close = combined["stock_close"].shift(1)
        tr1 = combined["stock_high"] - combined["stock_low"]
        tr2 = (combined["stock_high"] - prev_close).abs()
        tr3 = (combined["stock_low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
        atr_val = atr14.iloc[-1]
        atr_pct = (atr_val / close_last) * 100

        if atr_pct <= atr_max_pct:
            vol_score = 25
        elif atr_pct <= (atr_max_pct + 1.5):
            vol_score = 15
        else:
            vol_score = 0

        # --- 4. RELATIEF VOLUME (RVOL) ---
        sma_vol20 = combined["stock_volume"].rolling(window=20).mean()
        rvol = (combined["stock_volume"] / sma_vol20).iloc[-1]

        if rvol >= 1.2:
            rvol_score = 15
        elif rvol >= 1.0:
            rvol_score = 10
        else:
            rvol_score = 0

        total_score = rs_score + trend_score + vol_score + rvol_score
        return int(total_score)

    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────
# BEREKENING: DAGELIJKE RVOL SCORE (1D VS 20D GEMIDDELD VOLUME)
# ─────────────────────────────────────────────────────────────
def calculate_daily_rvol_score(df_stock: pd.DataFrame) -> int:
    """
    Berekent de RVOL op dagniveau door het volume van de meest recente handelsdag
    te vergelijken met het gemiddelde dagvolume van de afgelopen 20 handelsdagen.
    """
    try:
        if len(df_stock) < 20:
            return 0

        # 20-daags gemiddeld volume berekenen
        avg_vol_20d = df_stock["Volume"].rolling(window=20).mean().iloc[-1]
        latest_vol = float(df_stock["Volume"].iloc[-1])

        if avg_vol_20d == 0:
            return 0

        # RVOL verhouding
        rvol = latest_vol / avg_vol_20d

        # Score toewijzen op basis van dagelijkse RVOL
        if rvol >= 3.0:
            return 100
        elif rvol >= 2.0:
            return 75
        elif rvol >= 1.5:
            return 50
        elif rvol >= 1.0:
            return 25
        return 0

    except Exception:
        return 0


def get_signal_badge(score: int) -> str:
    """Vertaalt de score naar een compact signaal met emoji."""
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
# SCANNER APPLICATIE LOGICA
# ─────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if (
    st.sidebar.button("🚀 Start Multi-Timeframe Scan", type="primary")
    or "scanned" not in st.session_state
):
    st.session_state["scanned"] = True
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Haal SPY dagkoersen op voor de Stable RS Score berekening
    try:
        spy_df_daily = yf.download(SPY_TICKER, period="60d", interval="1d", progress=False)
        if isinstance(spy_df_daily.columns, pd.MultiIndex):
            spy_df_daily.columns = spy_df_daily.columns.get_level_values(0)
    except Exception:
        spy_df_daily = pd.DataFrame()

    timeframes = [
        ("1d", "60d", "1D"),
        ("1h", "60d", "1H"),
        ("15m", "7d", "15M"),
    ]

    for i, ticker in enumerate(tickers):
        status_text.text(
            f"Bezig met analyseren van {ticker} (1D, 1H, 15M, RS Score & Dagelijkse RVOL)..."
        )
        ticker_data = {"Ticker": ticker, "Prijs ($)": "N/A"}
        total_score_sum = 0
        stock_daily_df = pd.DataFrame()

        for interval, period, label in timeframes:
            try:
                data = yf.download(
                    ticker, period=period, interval=interval, progress=False
                )

                if not data.empty and len(data) >= 20:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    if interval == "1d":
                        stock_daily_df = data.copy()

                    df = calculate_trading_score(data)
                    latest = df.iloc[-1]

                    score = int(latest["Score"])
                    signal = get_signal_badge(score)

                    # Update prijs
                    ticker_data["Prijs ($)"] = round(float(latest["Close"]), 2)

                    # Score en signaal per timeframe opslaan
                    ticker_data[f"Score {label}"] = score
                    ticker_data[f"Signaal {label}"] = signal
                    total_score_sum += score
                else:
                    ticker_data[f"Score {label}"] = 0
                    ticker_data[f"Signaal {label}"] = "Geen data"

            except Exception as e:
                ticker_data[f"Score {label}"] = 0
                ticker_data[f"Signaal {label}"] = "Fout"

        # Bereken de Stable RS Score
        if not stock_daily_df.empty and not spy_df_daily.empty:
            rs_score_val = calculate_stable_rs_score(stock_daily_df, spy_df_daily, ATR_MAX_PCT)
            ticker_data["RS Score (0-100)"] = rs_score_val
        else:
            ticker_data["RS Score (0-100)"] = 0

        # Bereken de Dagelijkse RVOL Score (1D vs 20D gemiddelde)
        if not stock_daily_df.empty:
            rvol_daily_score = calculate_daily_rvol_score(stock_daily_df)
            ticker_data["RVOL 1D Score"] = rvol_daily_score
        else:
            ticker_data["RVOL 1D Score"] = 0

        # Gemiddelde score voor sortering op totaalbeeld
        ticker_data["Totale Matrix Score"] = total_score_sum
        results.append(ticker_data)

        progress_bar.progress((i + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(
            by="Totale Matrix Score", ascending=False
        )

        # Verwijder de hulpsorteerkolom
        display_df = res_df.drop(columns=["Totale Matrix Score"])

        # Kleurstijlen voor tabelweergave op alle score kolommen
        def highlight_scores(val):
            if isinstance(val, int):
                if val >= 4:
                    return "background-color: #28a745; color: white; font-weight: bold;"
                elif val == 3:
                    return "background-color: #ffc107; color: black; font-weight: bold;"
                else:
                    return "background-color: #dc3545; color: white; font-weight: bold;"
            return ""

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

        st.subheader("📋 Multi-Timeframe Score Overzicht")
        st.dataframe(
            display_df.style
            .map(highlight_scores, subset=["Score 1D", "Score 1H", "Score 15M"])
            .map(highlight_rs_score, subset=["RS Score (0-100)"])
            .map(highlight_rvol_score, subset=["RVOL 1D Score"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Geen data gevonden voor de opgegeven tickers.")
