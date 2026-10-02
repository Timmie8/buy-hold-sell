import pandas as pd
import streamlit as st
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Score Scanner", page_icon="📊", layout="wide"
)

st.title("📊 Auto Trading Score Scanner")
st.caption(
    "Gebaseerd op Pine Script: MA5/MA15 + VWAP + Volume Strategy (yfinance data)"
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR / CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Instellingen")

# Standaard tickers (aanpasbaar in de webapp)
default_tickers = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, INTC, PLTR"
ticker_input = st.sidebar.text_area("Tickers (gescheiden door komma)", default_tickers)

# Tijdsframe keuzes
interval_choice = st.sidebar.selectbox(
    "Interval", options=["1d", "1h", "15m", "5m"], index=0
)
period_choice = "60d" if interval_choice in ["1d", "1h"] else "7d"


# ─────────────────────────────────────────────────────────────
# LOGICA BEREKENING
# ─────────────────────────────────────────────────────────────
def calculate_trading_score(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA15"] = df["Close"].ewm(span=15, adjust=False).mean()

    df["HLC3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (df["HLC3"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["VolSMA20"] = df["Volume"].rolling(window=20).mean()

    # Regels (0 of 1)
    df["Rule1"] = (df["EMA5"] > df["EMA15"]).astype(int)
    df["Rule2"] = (df["Close"] > df["VWAP"]).astype(int)
    df["Rule3"] = (df["Volume"] > df["VolSMA20"]).astype(int)
    df["Rule4"] = (df["Close"] > df["EMA5"]).astype(int)
    df["Rule5"] = (df["EMA15"] > df["EMA15"].shift(1)).astype(int)

    df["Score"] = (
        df["Rule1"] + df["Rule2"] + df["Rule3"] + df["Rule4"] + df["Rule5"]
    )
    return df


def get_signal_text(score: int) -> str:
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
# MAIN SCANNER FUNCTION
# ─────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if st.sidebar.button("🚀 Start Scan", type="primary") or "scanned" not in st.session_state:
    st.session_state["scanned"] = True
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"Bezig met ophalen data voor {ticker}...")
        try:
            data = yf.download(
                ticker,
                period=period_choice,
                interval=interval_choice,
                progress=False,
            )

            if not data.empty and len(data) >= 20:
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
            st.error(f"Fout bij {ticker}: {e}")

        progress_bar.progress((i + 1) / len(tickers))

    status_text.empty()
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(
            by="Score", ascending=False
        )

        # Highlighten van resultaten op basis van score
        def highlight_score(val):
            if val >= 4:
                return "background-color: #d4edda; color: #155724; font-weight: bold;"
            elif val <= 2:
                return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
            return "background-color: #fff3cd; color: #856404;"

        st.subheader("Resultaten Overview")
        st.dataframe(
            res_df.style.map(highlight_score, subset=["Score"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Geen data gevonden voor de opgegeven tickers.")
