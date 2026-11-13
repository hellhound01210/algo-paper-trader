import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import numpy as np
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
STAGE1_TOP_N = 35

# Full 150-Stock Liquid Institutional Universe (Nifty 200 / F&O Leaders Across All Sectors)
INSTITUTIONAL_UNIVERSE = [
    # Commodities, Exchanges & Capital Markets
    "MCX", "CDSL", "BSE", "ANGELONE", "CAMSSERV", "KFINTECH", "IEX", "MANAPPURAM", "MUTHOOTFIN",
    # Metals & Mining
    "VEDL", "JINDALSTEL", "TATASTEEL", "HINDALCO", "JSWSTEEL", "NMDC", "SAIL", "NATIONALUM", "HINDCOPPER", "APLAPOLLO",
    # Banking & PSU Grinders
    "SBIN", "ICICIBANK", "HDFCBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB",
    "CANBK", "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK", "AUBANK", "BANDHANBNK", "RBLBANK", "KARURVYSYA",
    # NBFCs, Insurance & Asset Management
    "BAJAJFINSV", "BAJFINANCE", "CHOLAFIN", "HDFCAMC", "SHREECEM", "SHRIRAMFIN", "PFC", "RECLTD",
    "IRFC", "HUDCO", "LICHSGFIN", "LTF", "M&MFIN", "POONAWALLA", "SBICARD", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
    # Defense, Aerospace & Heavy Engineering
    "HAL", "BEL", "BHEL", "MAZDOCK", "COCHINSHIP", "BDL", "PARAS", "DATAPATTNS", "BEML",
    # Energy, Power & Oil/Gas
    "RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "TATAPOWER", "ADANIPOWER", "ADANIGREEN",
    "ADANIENSOL", "JSWENERGY", "NHPC", "SJVN", "TORNTPOWER", "CESC", "BPCL", "HPCL", "IOC", "GAIL", "PETRONET", "IGL",
    # Retail, Consumer & New-Age Tech
    "TRENT", "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY", "TITAN", "DMART", "INDHOTEL",
    "JUBLFOOD", "DEVYANI", "VBL", "GODREJCP", "DABUR", "MARICO", "COLPAL", "ITC", "HINDUNILVR", "NESTLEIND", "TATACONSUM",
    # EMS, Cables & Industrials
    "DIXON", "KAYNES", "AMBER", "SYRMA", "POLYCAB", "KEI", "HAVELLS", "VOLTAS", "BLUESTARCO",
    "CUMMINSIND", "SIEMENS", "ABB", "CGPOWER", "THERMAX", "ASTRAL", "SUPREMEIND", "PIIND",
    # Infrastructure, Ports & Real Estate
    "LT", "ADANIENT", "ADANIPORTS", "DLF", "GODREJPROP", "MACROTECH", "PRESTIGE", "OBEROIRLTY",
    "PHOENIXLTD", "NCC", "IRCON", "RVNL", "IRB", "GMRAIRPORT", "CONCOR", "ULTRACEMCO", "AMBUJACEM",
    # Automobiles & EV Supply Chain
    "TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "TVSMOTOR", "EICHERMOT", "HEROMOTOCO",
    "ASHOKLEY", "BHARATFORG", "SONACOMS", "MOTHERSON", "EXIDEIND", "AMARAJABAT", "BALKRISIND", "APOLLOTYRE",
    # IT, Software & Telecom
    "BHARTIARTL", "INDUSTOWER", "IDEA", "TATACOMM", "PERSISTENT", "COFORGE", "TCS", "INFY",
    "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "KPITTECH", "TATAELXSI", "OFSS", "CYIENT",
    # Pharma & Healthcare
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "BIOCON", "ZYDUSLIFE",
    "TORNTPHARM", "ALKEM", "LAURUSLABS", "GRANULES", "APOLLOHOSP", "MAXHEALTH", "FORTIS", "SYNGENE"
]

# Deduplicate universe while preserving order
INSTITUTIONAL_UNIVERSE = list(dict.fromkeys(INSTITUTIONAL_UNIVERSE))

# Tickers known to have superior ORVWB fill profiles on BSE (tested on both NSE and BSE in Stage 2)
DUAL_EXCHANGE_SYMBOLS = {"VEDL", "HDFCAMC", "CHOLAFIN", "RECLTD", "DIXON", "DLF", "BHEL", "PFC", "BEL", "M&M"}


def fetch_daily_regime_score(smart_api, scrip_df: pd.DataFrame, symbol: str) -> dict:
    """Stage 1: Fast 1-Day candle scan to rank intraday expansion, body efficiency, and turnover."""
    token = resolve_token(scrip_df, {"symbol": symbol, "exchange": "NSE", "token": None})
    exch = "NSE"
    if not token:
        token = resolve_token(scrip_df, {"symbol": symbol, "exchange": "BSE", "token": None})
        exch = "BSE"
    if not token:
        return {}

    now_ist = datetime.now(IST)
    from_dt = now_ist - timedelta(days=45)
    params = {
        "exchange": exch,
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
        "todate": now_ist.strftime("%Y-%m-%d 15:30"),
    }

    for attempt in range(3):
        try:
            time.sleep(0.55 + attempt * 0.8)
            res = smart_api.getCandleData(params)
            if res and res.get("status") and res.get("data"):
                df = pd.DataFrame(
                    res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                if len(df) < 12:
                    return {}
                c = df["Close"].astype(float)
                h = df["High"].astype(float)
                l = df["Low"].astype(float)
                o = df["Open"].astype(float)
                v = df["Volume"].astype(float)

                turnover_cr = float((c * v).tail(20).mean() / 1e7)
                # Filter out low-liquidity stocks (< ₹25 Cr daily turnover on NSE)
                if exch == "NSE" and turnover_cr < 25.0:
                    return {}

                range_pct = float(((h - l) / c.clip(lower=1.0) * 100.0).tail(20).mean())
                body_eff = float(( (c - o).abs() / (h - l).clip(lower=0.05) ).tail(20).mean())

                # Regime Score rewards wide daily range + strong directional closes + deep liquidity
                regime_score = round(range_pct * (0.5 + body_eff) * np.log1p(max(turnover_cr, 1.0)), 2)
                return {
                    "Symbol": symbol,
                    "Daily_Range_Pct": round(range_pct, 2),
                    "Body_Eff": round(body_eff, 2),
                    "Turnover_Cr": round(turnover_cr, 1),
                    "Regime_Score": regime_score,
                }
            elif res and res.get("errorCode") in ("AB1004", "AB2001"):
                time.sleep(2.5 + attempt * 1.5)
            else:
                break
        except Exception:
            time.sleep(1.5)
    return {}


def fetch_screener_candles(smart_api, exchange: str, token: str, days_back: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Stage 2: Hardened 99-day 5m fetcher for Top-35 finalists with rate-limit backoff."""
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
                time.sleep(1.25 + attempt * 1.5)
                res = smart_api.getCandleData(params)
                if res and res.get("status") and res.get("data"):
                    chunk_df = pd.DataFrame(
                        res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                    )
                    frames.append(chunk_df)
                    break
                elif res and res.get("errorCode") in ("AB1004", "AB2001"):
                    time.sleep(4.0 + attempt * 2.0)
                else:
                    time.sleep(2.0)
            except Exception:
                time.sleep(3.0)

    if len(frames) < 2:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)
    df.sort_values("Timestamp", inplace=True)
    df.set_index("Timestamp", inplace=True)
    return df


def send_screener_telegram(df_ranked: pd.DataFrame, scanned_count: int):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id or df_ranked.empty:
        return

    active_set = {f"{x['symbol']}.{x['exchange']}" for x in WATCHLIST}
    qualified = df_ranked[df_ranked["Quality_Gate"] == "PASS"].head(12)

    lines = [f"🛰️ *ORVWB v6 Two-Stage Institutional Screener*"]
    lines.append(f"• *Universe Scanned:* {scanned_count} Liquid F&O Stocks -> Top {len(df_ranked)} Deep-Audited")
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
        lines.append("\n🔒 *Roster Status:* All active tickers remain optimal.")

    msg = "\n".join(lines)
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"Screener Telegram alert skipped: {e}")


if __name__ == "__main__":
    print(f"=== Stage 1: Scanning {len(INSTITUTIONAL_UNIVERSE)} Institutional Stocks (Daily Regime Filter) ===")
    smart_api = get_angel_session()
    scrip_df = load_token_map()
    now_ist = datetime.now(IST)
    cutoff_45d = (now_ist - timedelta(days=45)).strftime("%Y-%m-%d")
    cutoff_30d = (now_ist - timedelta(days=30)).strftime("%Y-%m-%d")

    stage1_rows = []
    for sym in INSTITUTIONAL_UNIVERSE:
        info = fetch_daily_regime_score(smart_api, scrip_df, sym)
        if info:
            stage1_rows.append(info)

    stage1_df = pd.DataFrame(stage1_rows)
    active_symbols = [x["symbol"] for x in WATCHLIST]

    if not stage1_df.empty:
        stage1_df.sort_values(by="Regime_Score", ascending=False, inplace=True)
        top_symbols = stage1_df.head(STAGE1_TOP_N)["Symbol"].tolist()
        # Always include active watchlist symbols + top daily expansion leaders
        finalist_symbols = list(dict.fromkeys(active_symbols + top_symbols))
        print(f"Stage 1 Complete: Selected {len(finalist_symbols)} High-Expansion Finalists from {len(stage1_df)} liquid stocks.")
    else:
        finalist_symbols = list(dict.fromkeys(active_symbols + INSTITUTIONAL_UNIVERSE[:30]))

    print(f"\n=== Stage 2: Running 99-Day 5m ORVWB v6 Simulation on {len(finalist_symbols)} Finalists ===")
    ledger_fallback = pd.read_csv(LEDGER_FILE) if os.path.exists(LEDGER_FILE) else pd.DataFrame()
    active_exch_map = {x["symbol"]: x["exchange"] for x in WATCHLIST}
    rows = []

    for sym in finalist_symbols:
        # Test primary exchange + BSE only for dual-exchange candidates to cut API load by 45%
        exchanges_to_test = ["NSE", "BSE"] if sym in DUAL_EXCHANGE_SYMBOLS else ["NSE"]
        if sym in active_exch_map and active_exch_map[sym] not in exchanges_to_test:
            exchanges_to_test.append(active_exch_map[sym])

        for exch in exchanges_to_test:
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
        send_screener_telegram(dedup_df, len(INSTITUTIONAL_UNIVERSE))
        print(f"Saved {SCREENER_FILE} with {len(dedup_df)} multi-horizon ranked companies!")
