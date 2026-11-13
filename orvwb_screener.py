import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd

from orvwb_bot import (
    IST,
    LEDGER_FILE,
    WATCHLIST,
    get_angel_session,
    load_token_map,
    resolve_token,
    simulate_orvwb_v6,
)

SCREENER_FILE = "ORVWB_Screener_Ranked.csv"
LOOKBACK_DAYS = 99

CANDIDATE_SYMBOLS = [
    "MCX", "VEDL", "RELIANCE", "TRENT", "ADANIENT", "JINDALSTEL",
    "BHARTIARTL", "CDSL", "HAL", "DIXON", "BAJAJFINSV", "HDFCAMC",
    "ICICIBANK", "SBIN", "TATAMOTORS", "TATASTEEL", "HINDALCO", "DLF",
    "CHOLAFIN", "PFC", "RECLTD", "BEL", "BHEL", "POLYCAB",
    "PERSISTENT", "COFORGE", "LT", "M&M", "AXISBANK", "ZOMATO"
]


def fetch_screener_candles(smart_api, exchange: str, token: str, days_back: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Hardened 99-day fetcher for 60-ticker Screener runs with rate-limit backoff."""
    now_ist = datetime.now(IST)
    frames = []
    chunks = [(days_back, 50), (50, 0)]

    for start_d, end_d in chunks:
        from_dt = now_ist - timedelta(days=start_d)
        to_dt = now_ist - timedelta(days=end_d)
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": to_dt.strftime("%Y-%m-%d 15:30"),
        }
        for attempt in range(5):
            try:
                time.sleep(1.3 + attempt * 1.5)
                res = smart_api.getCandleData(params)
                if res and res.get("status") and res.get("data"):
                    chunk_df = pd.DataFrame(
                        res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                    )
                    frames.append(chunk_df)
                    break
                elif res and res.get("errorCode") in ("AB1004", "AB2001"):
                    time.sleep(4.5 + attempt * 2.0)
                else:
                    time.sleep(2.0)
            except Exception:
                time.sleep(3.5)

    if len(frames) < 2:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)
    df.sort_values("Timestamp", inplace=True)
    df.set_index("Timestamp", inplace=True)
    return df


def send_screener_telegram(df_ranked: pd.DataFrame):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id or df_ranked.empty:
        return

    active_set = {f"{x['symbol']}.{x['exchange']}" for x in WATCHLIST}
    qualified = df_ranked[df_ranked["Quality_Gate"] == "PASS"].head(12)

    lines = [f"🛰️ *ORVWB v6 Multi-Horizon Screener (99d + 45d + 30d)*"]
    lines.append("*Top 12 Institutional Leaders (Quality Gate: PF ≥ 1.50):*")
    for idx, (_, r) in enumerate(qualified.iterrows(), 1):
        badge = "✅" if r["Ticker"] in active_set else "🆕"
        lines.append(
            f"{idx}. {badge} `{r['Ticker']}`: 99d ₹{r['Total_Net_PnL_INR']:,.0f} | 45d ₹{r['Recent_45d_PnL_INR']:,.0f} | PF: {r['Profit_Factor']:.2f} | WR: {r['Win_Rate_Pct']:.1f}%"
        )

    new_challengers = [r["Ticker"] for _, r in qualified.iterrows() if r["Ticker"] not in active_set]
    if new_challengers:
        lines.append(f"\n🔄 *Verified Challengers (PF ≥ 1.50):* `{', '.join(new_challengers)}`")
    else:
        lines.append("\n🔒 *Roster Status:* All 12 active tickers remain optimal.")

    msg = "\n".join(lines)
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"Screener Telegram alert skipped: {e}")


if __name__ == "__main__":
    print(f"=== Running ORVWB v6 Multi-Horizon (99d/45d/30d) Screener ===")
    smart_api = get_angel_session()
    scrip_df = load_token_map()
    now_ist = datetime.now(IST)
    cutoff_45d = (now_ist - timedelta(days=45)).strftime("%Y-%m-%d")
    cutoff_30d = (now_ist - timedelta(days=30)).strftime("%Y-%m-%d")

    # Load Master Ledger as a guaranteed fallback against API throttling on active stocks
    ledger_fallback = pd.read_csv(LEDGER_FILE) if os.path.exists(LEDGER_FILE) else pd.DataFrame()
    rows = []

    for sym in CANDIDATE_SYMBOLS:
        for exch in ["NSE", "BSE"]:
            ticker_name = f"{sym}.{exch}"
            token = resolve_token(scrip_df, {"symbol": sym, "exchange": exch, "token": None})
            trades = []
            if token:
                candles = fetch_screener_candles(smart_api, exch, token, days_back=LOOKBACK_DAYS)
                if not candles.empty:
                    trades = simulate_orvwb_v6(candles, sym, exch)

            tdf = pd.DataFrame(trades) if trades else pd.DataFrame()

            # Fallback to local Master Ledger if Angel One throttled an active ticker
            if tdf.empty and not ledger_fallback.empty and ticker_name in ledger_fallback["Ticker"].values:
                print(f"⚠️ Using Master Ledger fallback for throttled active ticker: {ticker_name}")
                tdf = ledger_fallback[ledger_fallback["Ticker"] == ticker_name].copy()
                if "Base_1L_PnL" in tdf.columns:
                    tdf["Net_PnL_INR"] = tdf["Base_1L_PnL"].fillna(tdf["Net_PnL_INR"])

            if tdf.empty:
                continue

            tdf = tdf[~tdf["Status"].astype(str).str.startswith("OPEN")]
            if len(tdf) < 6:
                continue

            wins = tdf[tdf["Net_PnL_INR"] > 0]["Net_PnL_INR"].sum()
            losses = abs(tdf[tdf["Net_PnL_INR"] <= 0]["Net_PnL_INR"].sum())
            pf = round(wins / losses, 2) if losses > 0 else 99.0
            wr = round((tdf["Net_PnL_INR"] > 0).mean() * 100, 1)
            mean_r = round(tdf["Result_R"].mean(), 2)
            net_pnl = round(tdf["Net_PnL_INR"].sum(), 2)

            pnl_45d = round(tdf[tdf["Date"] >= cutoff_45d]["Net_PnL_INR"].sum(), 2)
            pnl_30d = round(tdf[tdf["Date"] >= cutoff_30d]["Net_PnL_INR"].sum(), 2)

            # Multi-Horizon Composite Score: 50% 99d Baseline + 30% 45d Regime + 20% 30d Velocity
            composite_score = round(0.50 * net_pnl + 0.30 * (pnl_45d * 1.6) + 0.20 * (pnl_30d * 2.2), 2)
            quality_pass = (pf >= 1.50) and (wr >= 52.5) and (net_pnl > 2000) and (pnl_45d > 0)

            print(f"{ticker_name} -> 99d: ₹{net_pnl:,.0f} | 45d: ₹{pnl_45d:,.0f} | 30d: ₹{pnl_30d:,.0f} | PF: {pf} | Gate: {'PASS' if quality_pass else 'REJECT'}")
            rows.append({
                "Symbol": sym,
                "Ticker": ticker_name,
                "Quality_Gate": "PASS" if quality_pass else "REJECT",
                "Trades_99d": len(tdf),
                "Win_Rate_Pct": wr,
                "Profit_Factor": pf,
                "Mean_R": mean_r,
                "Recent_30d_PnL_INR": pnl_30d,
                "Recent_45d_PnL_INR": pnl_45d,
                "Total_Net_PnL_INR": net_pnl,
                "Composite_Score_INR": composite_score,
            })

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df["Gate_Rank"] = (res_df["Quality_Gate"] == "PASS").astype(int)
        res_df.sort_values(
            by=["Gate_Rank", "Composite_Score_INR", "Profit_Factor"],
            ascending=[False, False, False],
            inplace=True,
        )
        dedup_df = res_df.drop_duplicates(subset=["Symbol"], keep="first").copy()
        dedup_df.drop(columns=["Symbol", "Gate_Rank"], inplace=True)
        dedup_df.to_csv(SCREENER_FILE, index=False)
        send_screener_telegram(dedup_df)
        print(f"Saved {SCREENER_FILE} with {len(dedup_df)} multi-horizon ranked companies!")
