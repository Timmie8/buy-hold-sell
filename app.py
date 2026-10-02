import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Trading Score Scanner", page_icon="📊", layout="wide"
)

st.title("📊 Auto Trading Score Scanner")
st.caption(
    "Exacte Python/Streamlit vertaling van de Pine Script MA5/MA15 + VWAP + Volume indicator."
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR / INSTELLINGEN
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Instellingen")

# Standaard tickerlijst
default_tickers = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, INTC, PLTR"
ticker_input = st.sidebar.text_area(
    "Tickers (gescheiden door komma)", default_tickers, height=120
)

# Timeframe / Interval keuzes
interval_choice = st.sidebar.selectbox(
    "Interval",
    options=["1d", "1h", "15m", "5m"],
    index=0,
    help="Kies het tijdsframe voor de indicatoren.",
)

# Period dynamisch bepalen afhankelijk van interval
period_choice = "60d" if interval_choice in ["1d", "1h"] else "7d"


# ─────────────────────────────────────────────────────────────
# TRADING SCORE BEREKENING (EXACT TRADINGVIEW MATCH)
# ─────────────────────────────────────────────────────────────
def calculate_trading_score(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent de indicatoren en 5 regels exact zoals in Pine Script v5."""
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

    # Rule 1: Trend (MA5 > MA15)
    df["Rule1"] = (df["EMA5"] > df["EMA15"]).astype(int)

    # Rule 2: VWAP position (Close > VWAP)
    df["Rule2"] = (df["Close"] > df["VWAP"]).astype(int)

    # Rule 3: Volume spike (Volume > VolSMA20)
    df["Rule3"] = (df["Volume"] > df["VolSMA20"]).astype(int)

    # Rule 4: Momentum continuation (Close > MA5)
    df["Rule4"] = (df["Close"] > df["EMA5"]).astype(int)

    # Rule 5: Structure (MA15 slope > 0 op vorige bar)
    df["Rule5"] = (df["EMA15"] > df["EMA15"].shift(1)).astype(int)

    # Totale Score (0 t/m 5)
    df["Score"] = (
        df["Rule1"] + df["Rule2"] + df["Rule3"] + df["Rule4"] + df["Rule5"]
    )

    return df


def get_signal_text(score: int) -> str:
    """Vertaalt de score naar de exacte tekst van de Pine Script switch statement."""
    mapping = {
        5: "Strong Buy 🚀",
        4: "Buy 📈",
        3: "Hold ⚖️",
        2: "Sell 📉",
        1: "Strong Sell 🔴",
        0: "Strong Sell 🔴",
    }
    return mapping.get(score, "N/A")


# ─────────────────────────────────────────────────────────────
# SCANNER APPLICATIE LOGICA
# ─────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if (
    st.sidebar.button("🚀 Start Scan", type="primary")
    or "scanned" not in st.session_state
):
    st.session_state["scanned"] = True
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Bezig met analyseren van {ticker}...")
        try:
            data = yf.download(
                ticker,
                period=period_choice,
                interval=interval_choice,
                progress=False,
            )

            if not data.empty and len(data) >= 20:
                # Eventuele MultiIndex kolomstructuur platmaken
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                df = calculate_trading_score(data)
                latest = df.iloc[-1]

                score = int(latest["Score"])
                signal = get_signal_text(score)

                results.append(
                    {
                        "Ticker": ticker,
                        "Prijs ($)": round(float(latest["Close"]), 2),
                        "Score": score,
                        "Signaal": signal,
                        "Rule 1 (MA5>15)": "✅" if latest["Rule1"] == 1 else "❌",
                        "Rule 2 (>VWAP)": "✅" if latest["Rule2"] == 1 else "❌",
                        "Rule 3 (>Vol)": "✅" if latest["Rule3"] == 1 else "❌",
                        "Rule 4 (>MA5)": "✅" if latest["Rule4"] == 1 else "❌",
                        "Rule 5 (MA15 ↑)": "✅" if latest["Rule5"] == 1 else "❌",
                    }
                )
        except Exception as e:
            st.error(f"Fout bij ophalen {ticker}: {e}")

        progress_bar.progress((i + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(
            by="Score", ascending=False
        )

        # Kleurstijlen voor tabelweergave
        def highlight_score(val):
            if val >= 4:
                return "background-color: #28a745; color: white; font-weight: bold;"
            elif val == 3:
                return "background-color: #ffc107; color: black; font-weight: bold;"
            else:
                return "background-color: #dc3545; color: white; font-weight: bold;"

        st.subheader("📋 Overzicht Aandelen Scores")
        st.dataframe(
            res_df.style.map(highlight_score, subset=["Score"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Geen data gevonden voor de opgegeven tickers.")
