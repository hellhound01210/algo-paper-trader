import os
import time
import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pyotp
from SmartApi import SmartConnect

from orvwb_bot import (
    IST,
    WATCHLIST,
    get_angel_session,
    load_token_map,
    resolve_token,
    fetch_5m_candles,
    simulate_orvwb_v6,
)

SCREENER_FILE = "ORVWB_Screener_Ranked.csv"
LOOKBACK_DAYS = 99

# 30-Stock High-Beta Liquid Institutional Universe (Tested on Both NSE & BSE)
CANDIDATE_SYMBOLS = [
    "MCX", "VEDL", "RELIANCE", "TRENT", "ADANIENT", "JINDALSTEL",
    "BHARTIARTL", "CDSL", "HAL", "DIXON", "BAJAJFINSV", "HDFCAMC",
    "ICICIBANK", "SBIN", "TATAMOTORS", "TATASTEEL", "HINDALCO", "DLF",
    "CHOLAFIN", "PFC", "RECLTD", "BEL", "BHEL", "POLYCAB",
    "PERSISTENT", "COFORGE", "LT", "M&M", "AXISBANK", "ZOMATO"
]


def send_screener_telegram(df_ranked: pd.DataFrame):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id or df_ranked.empty:
        return

    active_set = {f"{x['symbol']}.{x['exchange']}" for x in WATCHLIST}
    top12 = df_ranked.head(12)

    lines = [f"🛰️ *ORVWB v6 {LOOKBACK_DAYS}-Day Screener Intelligence*"]
    lines.append("*Top 12 Institutional Leaders (1 Per Company):*")
    for idx, (_, r) in enumerate(top12.iterrows(), 1):
        badge = "✅" if r["Ticker"] in active_set else "🆕"
        lines.append(
            f"{idx}. {badge} `{r['Ticker']}`: ₹{r['Total_Net_PnL_INR']:,.0f} | PF: {r['Profit_Factor']:.2f} | WR: {r['Win_Rate_Pct']:.1f}% ({int(r['Trades_99d'])}T)"
        )

    new_challengers = [r["Ticker"] for _, r in top12.iterrows() if r["Ticker"] not in active_set]
    if new_challengers:
        lines.append(f"\n🔄 *Rotation Candidates Detected:* `{', '.join(new_challengers)}`")
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
    print(f"=== Running ORVWB v6 {LOOKBACK_DAYS}-Day Institutional Screener ===")
    smart_api = get_angel_session()
    scrip_df = load_token_map()
    rows = []

    for sym in CANDIDATE_SYMBOLS:
        for exch in ["NSE", "BSE"]:
            token = resolve_token(scrip_df, {"symbol": sym, "exchange": exch, "token": None})
            if not token:
                continue
            candles = fetch_5m_candles(smart_api, exch, token, days_back=LOOKBACK_DAYS)
            if candles.empty:
                continue
            trades = simulate_orvwb_v6(candles, sym, exch)
            if not trades:
                continue

            tdf = pd.DataFrame(trades)
            # Exclude open floating mid-day rows from historical ranking
            tdf = tdf[~tdf["Status"].astype(str).str.startswith("OPEN")]
            if len(tdf) < 5:
                continue

            wins = tdf[tdf["Net_PnL_INR"] > 0]["Net_PnL_INR"].sum()
            losses = abs(tdf[tdf["Net_PnL_INR"] <= 0]["Net_PnL_INR"].sum())
            pf = round(wins / losses, 2) if losses > 0 else 99.0
            wr = round((tdf["Net_PnL_INR"] > 0).mean() * 100, 1)
            mean_r = round(tdf["Result_R"].mean(), 2)
            net_pnl = round(tdf["Net_PnL_INR"].sum(), 2)

            print(f"{sym}.{exch} -> {LOOKBACK_DAYS}d Trades: {len(tdf)} | WR: {wr}% | PF: {pf} | PnL: INR {net_pnl:,.2f}")
            rows.append({
                "Symbol": sym,
                "Ticker": f"{sym}.{exch}",
                "Trades_99d": len(tdf),
                "Win_Rate_Pct": wr,
                "Profit_Factor": pf,
                "Mean_R": mean_r,
                "Total_Net_PnL_INR": net_pnl,
            })

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        # Sort by Total Net PnL descending, keep best exchange per company
        res_df.sort_values(by=["Total_Net_PnL_INR", "Profit_Factor"], ascending=[False, False], inplace=True)
        dedup_df = res_df.drop_duplicates(subset=["Symbol"], keep="first").copy()
        dedup_df.drop(columns=["Symbol"], inplace=True)
        dedup_df.to_csv(SCREENER_FILE, index=False)
        send_screener_telegram(dedup_df)
        print(f"Saved {SCREENER_FILE} with {len(dedup_df)} ranked companies!")
