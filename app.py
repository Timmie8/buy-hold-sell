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
    "Scant elk aandeel direct op 1 Dag (1D), 1 Uur (1H) en 15 Minuten (15M) volgens de exacte Pine Script logica."
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

    timeframes = [
        ("1d", "60d", "1D"),
        ("1h", "60d", "1H"),
        ("15m", "7d", "15M"),
    ]

    for i, ticker in enumerate(tickers):
        status_text.text(
            f"Bezig met analyseren van {ticker} (1D, 1H en 15M)..."
        )
        ticker_data = {"Ticker": ticker, "Prijs ($)": "N/A"}
        total_score_sum = 0

        for interval, period, label in timeframes:
            try:
                data = yf.download(
                    ticker, period=period, interval=interval, progress=False
                )

                if not data.empty and len(data) >= 20:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

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

        st.subheader("📋 Multi-Timeframe Score Overzicht")
        st.dataframe(
            display_df.style.map(
                highlight_scores,
                subset=["Score 1D", "Score 1H", "Score 15M"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Geen data gevonden voor de opgegeven tickers.")
