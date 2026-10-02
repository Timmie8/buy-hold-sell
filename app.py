import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# CONFIGURATIE & TICKERS
# ─────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "AMD",
    "INTC",
    "PLTR",
]
PERIOD = "60d"  # Voldoende historie voor MA15 / SMA20 berekeningen
INTERVAL = "1d"  # Dagsgrafiek ('1d', '1h', '15m', etc.)


# ─────────────────────────────────────────────────────────────
# BEREKENING PER AANDEEL
# ─────────────────────────────────────────────────────────────
def calculate_trading_score(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent de indicatoren en scores exact volgens de Pine Script logica."""
    # Indicatoren
    df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA15"] = df["Close"].ewm(span=15, adjust=False).mean()

    # VWAP berekening: (High + Low + Close) / 3 * Volume
    df["HLC3"] = (df["High"] + df["Low"] + df["Close"]) / 3
    # Cumulatieve VWAP (of dagelijkse reset bij intraday data)
    df["VWAP"] = (df["HLC3"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    df["VolSMA20"] = df["Volume"].rolling(window=20).mean()

    # Regels (0 of 1)
    # Rule 1: Trend (MA5 > MA15)
    df["Rule1"] = (df["EMA5"] > df["EMA15"]).astype(int)

    # Rule 2: VWAP position (Close > VWAP)
    df["Rule2"] = (df["Close"] > df["VWAP"]).astype(int)

    # Rule 3: Volume spike (Volume > VolSMA20)
    df["Rule3"] = (df["Volume"] > df["VolSMA20"]).astype(int)

    # Rule 4: Momentum continuation (Close > MA5)
    df["Rule4"] = (df["Close"] > df["EMA5"]).astype(int)

    # Rule 5: Structure (MA15 slope > 0)
    df["Rule5"] = (df["EMA15"] > df["EMA15"].shift(1)).astype(int)

    # Totale score
    df["Score"] = (
        df["Rule1"] + df["Rule2"] + df["Rule3"] + df["Rule4"] + df["Rule5"]
    )

    return df


def get_signal_text(score: int) -> str:
    """Vertaalt score naar tekst zoals in de Pine Script switch-statement."""
    mapping = {
        5: "Strong Buy",
        4: "Buy",
        3: "Hold",
        2: "Sell",
        1: "Strong Sell",
        0: "Strong Sell",  # Pine Script switch vangt 0 ook als N/A op, hier 'Strong Sell'
    }
    return mapping.get(score, "N/A")


# ─────────────────────────────────────────────────────────────
# MAIN SCANNER LOOP
# ─────────────────────────────────────────────────────────────
def run_scanner(tickers: list) -> pd.DataFrame:
    print(f"Bezig met scannen van {len(tickers)} aandelen...\n")
    results = []

    for ticker in tickers:
        try:
            # Data ophalen via yfinance
            data = yf.download(
                ticker, period=PERIOD, interval=INTERVAL, progress=False
            )

            if data.empty or len(data) < 20:
                print(f"⚠️ Geen of onvoldoende data voor {ticker}")
                continue

            # Flatten multi-index columns als yfinance die teruggeeft
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Score berekenen
            df = calculate_trading_score(data)
            latest = df.iloc[-1]  # Meest recente bar (vandaag/laatste slot)

            score = int(latest["Score"])
            signal = get_signal_text(score)

            results.append(
                {
                    "Ticker": ticker,
                    "Prijs ($)": round(latest["Close"], 2),
                    "Score (0-5)": score,
                    "Signaal": signal,
                    "Rule 1 (MA5>15)": "✅" if latest["Rule1"] == 1 else "❌",
                    "Rule 2 (>VWAP)": "✅" if latest["Rule2"] == 1 else "❌",
                    "Rule 3 (>Vol)": "✅" if latest["Rule3"] == 1 else "❌",
                    "Rule 4 (>MA5)": "✅" if latest["Rule4"] == 1 else "❌",
                    "Rule 5 (MA15 ↑)": "✅" if latest["Rule5"] == 1 else "❌",
                }
            )
        except Exception as e:
            print(f"Fout bij verwerken van {ticker}: {e}")

    # Resultaten omzetten naar DataFrame en sorteren op hoogste score
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Score (0-5)", ascending=False)

    return res_df


# ─────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scanner_results = run_scanner(TICKERS)

    # Nette output tonen in terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("=" * 70)
    print(" 📊 AUTOMATED TRADING SCORE SCANNER RESULTS")
    print("=" * 70)
    print(scanner_results.to_string(index=False))
