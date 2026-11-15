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

LEDGER_FILE = "ORVWB_v6_Elite_Ledger.csv"
SCREENER_FILE = "ORVWB_Screener_Ranked.csv"
IST = timezone(timedelta(hours=5, minutes=30))
LOOKBACK_DAYS = 99
MIN_SHARE_PRICE = 100.0

# Fallback 16-Stock Institutional Portfolio (Used if Screener CSV is missing or incomplete)
DEFAULT_WATCHLIST = [
    {"symbol": "POLICYBZR", "exchange": "NSE", "token": None},
    {"symbol": "MCX", "exchange": "NSE", "token": None},
    {"symbol": "JINDALSTEL", "exchange": "NSE", "token": None},
    {"symbol": "VEDL", "exchange": "BSE", "token": None},
    {"symbol": "LAURUSLABS", "exchange": "NSE", "token": None},
    {"symbol": "SHRIRAMFIN", "exchange": "NSE", "token": None},
    {"symbol": "PAYTM", "exchange": "NSE", "token": None},
    {"symbol": "RELIANCE", "exchange": "NSE", "token": None},
    {"symbol": "BHARTIARTL", "exchange": "NSE", "token": None},
    {"symbol": "BSE", "exchange": "NSE", "token": None},
    {"symbol": "BAJAJFINSV", "exchange": "NSE", "token": None},
    {"symbol": "TRENT", "exchange": "NSE", "token": None},
    {"symbol": "HDFCAMC", "exchange": "BSE", "token": None},
    {"symbol": "CDSL", "exchange": "NSE", "token": None},
    {"symbol": "ICICIBANK", "exchange": "NSE", "token": None},
    {"symbol": "PARAS", "exchange": "NSE", "token": None},
]


def load_dynamic_watchlist(default_list: list, min_stocks: int = 8, max_stocks: int = 16) -> list:
    """Automatically loads top Quality-Gate PASS stocks from ORVWB_Screener_Ranked.csv with circuit breakers."""
    if not os.path.exists(SCREENER_FILE):
        return default_list
    try:
        scr = pd.read_csv(SCREENER_FILE)
        if scr.empty or "Quality_Gate" not in scr.columns:
            return default_list

        passed = scr[
            (scr["Quality_Gate"] == "PASS")
            & (scr["Profit_Factor"] >= 1.50)
            & (scr["Win_Rate_Pct"] >= 52.5)
            & (scr["Recent_45d_PnL_INR"] > 0)
            & (scr["Recent_30d_PnL_INR"] >= 0)
            & (~scr["Ticker"].astype(str).str.startswith("IDEA."))
        ].head(max_stocks)

        if len(passed) < min_stocks:
            print(f"⚠️ Screener has only {len(passed)} passing stocks (< {min_stocks}). Using fallback watchlist.")
            return default_list

        dynamic_list = []
        for ticker_str in passed["Ticker"].astype(str):
            if "." in ticker_str:
                sym, exch = ticker_str.split(".", 1)
                dynamic_list.append({"symbol": sym.strip(), "exchange": exch.strip(), "token": None})

        return dynamic_list if len(dynamic_list) >= min_stocks else default_list
    except Exception as e:
        print(f"⚠️ Error loading dynamic watchlist ({e}). Using fallback watchlist.")
        return default_list


WATCHLIST = load_dynamic_watchlist(DEFAULT_WATCHLIST)

PSU_GRINDERS = ["SBIN", "BANKBARODA", "PNB", "CANBK", "NTPC", "POWERGRID", "COALINDIA"]
CAPITAL_PER_TRADE = 100000.0
COMMISSION_PCT = 0.0003

CONVICTION_CAP_MAP = {0: 70000.0, 1: 85000.0, 2: 115000.0, 3: 135000.0}
CONVICTION_LABEL_MAP = {
    0: "Tier C (0.70x)",
    1: "Tier B (0.85x)",
    2: "Tier A (1.15x)",
    3: "Tier A+ (1.35x)",
}


def get_angel_session():
    api_key = os.environ["ANGEL_API_KEY"].strip()
    client_code = os.environ["ANGEL_CLIENT_CODE"].strip()
    pin = os.environ["ANGEL_PIN"].strip()
    totp_secret = os.environ["ANGEL_TOTP_SECRET"].strip().replace(" ", "")

    for attempt in range(3):
        try:
            smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            data = smart_api.generateSession(client_code, pin, totp)
            if data and data.get("status"):
                return smart_api
            print(f"Login attempt {attempt + 1} failed: {data}. Retrying in 5s...")
        except Exception as e:
            print(f"Login exception on attempt {attempt + 1}: {e}")
        time.sleep(5)
    raise RuntimeError("Angel One Login Failed after 3 attempts.")


def load_token_map():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                scrip_list = json.loads(response.read().decode())
            return pd.DataFrame(scrip_list)
        except Exception as e:
            print(f"ScripMaster download attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    raise RuntimeError("Failed to download OpenAPIScripMaster.json")


def resolve_token(scrip_df: pd.DataFrame, item: dict) -> str:
    if item.get("token"):
        return str(item["token"])
    sym = item["symbol"]
    exch = item["exchange"]
    if exch == "NSE":
        sub = scrip_df[(scrip_df["exch_seg"] == "NSE") & (scrip_df["symbol"] == f"{sym}-EQ")]
    else:
        sub = scrip_df[
            (scrip_df["exch_seg"] == "BSE") &
            ((scrip_df["name"] == sym) | (scrip_df["symbol"] == sym))
        ]
    if sub.empty:
        return ""
    return str(sub.iloc[0]["token"])


def fetch_5m_candles(smart_api, exchange: str, token: str, days_back: int = LOOKBACK_DAYS) -> pd.DataFrame:
    now_ist = datetime.now(IST)
    frames = []
    chunks = [(days_back, 50), (50, 0)] if days_back > 55 else [(days_back, 0)]

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
        for attempt in range(4):
            try:
                time.sleep(1.0 + attempt * 1.2)
                res = smart_api.getCandleData(params)
                if res and res.get("status") and res.get("data"):
                    chunk_df = pd.DataFrame(
                        res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                    )
                    frames.append(chunk_df)
                    break
                elif res and res.get("errorCode") in ("AB1004", "AB2001"):
                    time.sleep(3.5 + attempt * 1.5)
                else:
                    time.sleep(1.5)
            except Exception:
                time.sleep(2.5)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)
    df.sort_values("Timestamp", inplace=True)
    df.set_index("Timestamp", inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["AVG_VOL"] = df["Volume"].rolling(20).mean()

    df["Date"] = df.index.date
    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["PV"] = hlc3 * df["Volume"]
    df["Cum_PV"] = df.groupby("Date")["PV"].cumsum()
    df["Cum_Vol"] = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = np.where(df["Cum_Vol"] > 0, df["Cum_PV"] / df["Cum_Vol"], hlc3)
    return df


def simulate_orvwb_v6(df: pd.DataFrame, symbol: str, exchange: str) -> list:
    if df.empty:
        return []
    df = compute_indicators(df)
    use_eng_a = not any(psu in symbol.upper() for psu in PSU_GRINDERS)
    engine_name = "Engine A (35/65)" if use_eng_a else "Engine B (2.5R Run)"
    trades = []

    for trade_date, day_df in df.groupby("Date"):
        or_bars = day_df.between_time("09:15", "09:25")
        if len(or_bars) < 3:
            continue

        or_high = float(or_bars["High"].max())
        or_low = float(or_bars["Low"].min())
        or_mid = (or_high + or_low) * 0.5
        or_width = or_high - or_low

        entry_end = "11:10" if use_eng_a else "11:25"
        active_bars = day_df.between_time("09:30", "15:15")
        if len(active_bars) < 1:
            continue

        pos = 0
        entry_price = active_stop = tp1_price = tp2_price = init_risk = 0.0
        tp1_hit = False
        entry_time = None
        realized_r = 0.0
        rem_weight = 1.0
        exit_reason = ""
        exit_price = 0.0
        last_close = 0.0

        for ts, row in active_bars.iterrows():
            t_str = ts.strftime("%H:%M")
            c, h, l, o, v = float(row["Close"]), float(row["High"]), float(row["Low"]), float(row["Open"]), float(row["Volume"])
            atr, vwap, ema20, avg_vol = float(row["ATR"]), float(row["VWAP"]), float(row["EMA20"]), float(row["AVG_VOL"])
            last_close = c

            if np.isnan(atr) or np.isnan(avg_vol) or atr <= 0:
                continue

            if pos != 0:
                if pos == 1:
                    if l <= active_stop:
                        realized_r += rem_weight * ((active_stop - entry_price) / init_risk)
                        exit_price = active_stop
                        exit_reason = "TP1 + BE Shield" if tp1_hit else "Stop Loss"
                        pos = 0
                        break
                    if use_eng_a and not tp1_hit and h >= tp1_price:
                        realized_r += 0.35 * 1.0
                        rem_weight = 0.65
                        active_stop = entry_price + (0.15 * init_risk)
                        tp1_hit = True
                    if h >= tp2_price:
                        realized_r += rem_weight * 2.5
                        exit_price = tp2_price
                        exit_reason = "Full TP2 (+2.5R)"
                        pos = 0
                        break
                elif pos == -1:
                    if h >= active_stop:
                        realized_r += rem_weight * ((entry_price - active_stop) / init_risk)
                        exit_price = active_stop
                        exit_reason = "TP1 + BE Shield" if tp1_hit else "Stop Loss"
                        pos = 0
                        break
                    if use_eng_a and not tp1_hit and l <= tp1_price:
                        realized_r += 0.35 * 1.0
                        rem_weight = 0.65
                        active_stop = entry_price - (0.15 * init_risk)
                        tp1_hit = True
                    if l <= tp2_price:
                        realized_r += rem_weight * 2.5
                        exit_price = tp2_price
                        exit_reason = "Full TP2 (-2.5R)"
                        pos = 0
                        break

                if t_str >= "15:10":
                    move = (c - entry_price) if pos == 1 else (entry_price - c)
                    realized_r += rem_weight * (move / init_risk)
                    exit_price = c
                    exit_reason = "15:15 EOD Square-Off"
                    pos = 0
                    break

            if pos == 0 and entry_time is None and t_str <= entry_end:
                # Hard Penny-Stock Guard: Never trigger entries below ₹100/share
                if c < MIN_SHARE_PRICE:
                    continue

                bar_range = max(h - l, 0.05)
                body_pct = abs(c - o) / bar_range

                if use_eng_a:
                    or_valid = 0 < or_width <= (3.4 * atr)
                    has_vol = v >= (1.10 * avg_vol)
                    strong_body = body_pct >= 0.45
                    bull_top = ((h - c) / bar_range) <= 0.35
                    bear_bot = ((c - l) / bar_range) <= 0.35

                    long_sig = or_valid and has_vol and strong_body and bull_top and (c > or_high) and (c > vwap) and (c > ema20)
                    short_sig = or_valid and has_vol and strong_body and bear_bot and (c < or_low) and (c < vwap) and (c < ema20)
                else:
                    has_vol = v > (1.20 * avg_vol)
                    long_sig = has_vol and (c > or_high) and (c > vwap)
                    short_sig = has_vol and (c < or_low) and (c < vwap)

                if long_sig:
                    pos = 1
                    entry_time = t_str
                    entry_price = c
                    raw_sl = min(or_mid - 0.75 * atr, l - 0.30 * atr)
                    active_stop = min(raw_sl, entry_price - 0.75 * atr) if use_eng_a else (or_low - 1.0 * atr)
                    init_risk = max(entry_price - active_stop, 0.05)
                    tp1_price = entry_price + 1.0 * init_risk
                    tp2_price = entry_price + 2.5 * init_risk
                elif short_sig:
                    pos = -1
                    entry_time = t_str
                    entry_price = c
                    raw_sl = max(or_mid + 0.75 * atr, h + 0.30 * atr)
                    active_stop = max(raw_sl, entry_price + 0.75 * atr) if use_eng_a else (or_high + 1.0 * atr)
                    init_risk = max(active_stop - entry_price, 0.05)
                    tp1_price = entry_price - 1.0 * init_risk
                    tp2_price = entry_price - 2.5 * init_risk

        if entry_time is not None and pos != 0 and exit_reason == "":
            move = (last_close - entry_price) if pos == 1 else (entry_price - last_close)
            realized_r += rem_weight * (move / init_risk)
            exit_price = last_close
            exit_reason = "OPEN (TP1 Hit, Trailing)" if tp1_hit else "OPEN (Floating)"

        if entry_time is not None and exit_reason != "":
            qty = max(int(CAPITAL_PER_TRADE // entry_price), 1)
            gross_pnl = realized_r * init_risk * qty
            fees = CAPITAL_PER_TRADE * COMMISSION_PCT * 2
            base_pnl = round(gross_pnl - fees, 2)
            trades.append({
                "Date": str(trade_date),
                "Ticker": f"{symbol}.{exchange}",
                "Engine": engine_name,
                "Side": "LONG" if (tp2_price > entry_price) else "SHORT",
                "Entry_Time": entry_time,
                "Entry_Price": round(entry_price, 2),
                "Stop_Loss": round(active_stop, 2),
                "Target_2.5R": round(tp2_price, 2),
                "Exit_Price": round(exit_price, 2),
                "Result_R": round(realized_r, 2),
                "Base_1L_PnL": base_pnl,
                "Net_PnL_INR": base_pnl,
                "Status": exit_reason,
            })
    return trades


def apply_adaptive_conviction_sizing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(by=["Date", "Entry_Time"]).reset_index(drop=True)

    if "Base_1L_PnL" not in df.columns:
        df["Base_1L_PnL"] = np.nan

    missing_base = df["Base_1L_PnL"].isna()
    if missing_base.any():
        init_risk = (df.loc[missing_base, "Target_2.5R"] - df.loc[missing_base, "Entry_Price"]).abs() / 2.5
        qty_1l = (CAPITAL_PER_TRADE // df.loc[missing_base, "Entry_Price"]).clip(lower=1)
        fees_1l = CAPITAL_PER_TRADE * COMMISSION_PCT * 2
        df.loc[missing_base, "Base_1L_PnL"] = (
            df.loc[missing_base, "Result_R"] * init_risk * qty_1l - fees_1l
        ).round(2)

    prior_3_sum = (
        df.groupby("Ticker")["Base_1L_PnL"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0.0)
    )
    expanding_mean_r = (
        df.groupby("Ticker")["Result_R"]
        .transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
        .fillna(0.35)
    )
    is_prime_time = df["Entry_Time"] <= "10:15"

    # Factor 4: Walk-Forward Intraday Market Breadth Synchronizer (up to current Entry_Time)
    same_side_cnt = []
    opp_side_cnt = []
    for idx, row in df.iterrows():
        sub = df[(df["Date"] == row["Date"]) & (df["Entry_Time"] <= row["Entry_Time"]) & (df.index != idx)]
        same_side_cnt.append(int((sub["Side"] == row["Side"]).sum()))
        opp_side_cnt.append(int((sub["Side"] != row["Side"]).sum()))

    same_s = pd.Series(same_side_cnt, index=df.index)
    opp_s = pd.Series(opp_side_cnt, index=df.index)
    breadth_adj = np.where((same_s >= 1) & (opp_s == 0), 1, np.where(opp_s >= 1, -1, 0))

    base_scores = (
        is_prime_time.astype(int)
        + (expanding_mean_r >= 0.30).astype(int)
        + (prior_3_sum <= 0.0).astype(int)
    )
    scores = (base_scores + breadth_adj).clip(0, 3)

    trade_cap = scores.map(CONVICTION_CAP_MAP)
    qty_flat = (CAPITAL_PER_TRADE // df["Entry_Price"]).clip(lower=1)
    qty_adaptive = (trade_cap // df["Entry_Price"]).astype(int)

    per_share_gross = (df["Base_1L_PnL"] + (CAPITAL_PER_TRADE * COMMISSION_PCT * 2)) / qty_flat
    adaptive_fees = trade_cap * COMMISSION_PCT * 2

    df["Conviction"] = scores.map(CONVICTION_LABEL_MAP)
    df["Net_PnL_INR"] = (per_share_gross * qty_adaptive - adaptive_fees).round(2)
    return df


def send_telegram_summary(today_str: str, today_df: pd.DataFrame, combined_df: pd.DataFrame, missing_feeds: list):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return

    roster_count = len(WATCHLIST)
    total_pnl = combined_df["Net_PnL_INR"].sum() if not combined_df.empty else 0.0
    win_rate = (combined_df["Net_PnL_INR"] > 0).mean() * 100 if not combined_df.empty else 0.0
    gw = combined_df[combined_df["Net_PnL_INR"] > 0]["Net_PnL_INR"].sum() if not combined_df.empty else 0.0
    gl = abs(combined_df[combined_df["Net_PnL_INR"] <= 0]["Net_PnL_INR"].sum()) if not combined_df.empty else 0.0
    pf = (gw / gl) if gl > 0 else 99.0

    max_dd = 0.0
    if not combined_df.empty:
        daily_eq = combined_df.groupby("Date")["Net_PnL_INR"].sum().cumsum()
        peak = daily_eq.cummax()
        max_dd = float((daily_eq - peak).min())

    lines = [f"📊 *ORVWB v6 Command Report ({today_str})*"]
    if today_df.empty:
        lines.append(f"• *Today's Action:* No setups triggered across the {roster_count}-stock watchlist today.")
    else:
        for _, r in today_df.iterrows():
            tier_str = r.get("Conviction", "Tier A")
            lines.append(
                f"• `{r['Ticker']}` [{tier_str}] ({r['Side']} @ {r['Entry_Time']} | Entry: ₹{r['Entry_Price']} | SL: ₹{r['Stop_Loss']}): {r['Result_R']:+.2f}R | ₹{r['Net_PnL_INR']:+.2f} [{r['Status']}]"
            )
        lines.append(f"• *Today's Net PnL:* ₹{today_df['Net_PnL_INR'].sum():,.2f}")

    lines.append(
        f"\n📈 *99d Adaptive Ledger Telemetry ({roster_count} Stocks):*\n"
        f"• *Total Trades:* {len(combined_df)} | *Win Rate:* {win_rate:.1f}% | *PF:* {pf:.2f}\n"
        f"• *Cumulative Net PnL:* ₹{total_pnl:,.2f} | *Max DD:* ₹{max_dd:,.2f}"
    )

    if missing_feeds:
        lines.append(f"⚠️ *API Watchdog:* Missing feed on {', '.join(missing_feeds)} (preserved from ledger)")

    msg = "\n".join(lines)

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"Telegram alert skipped: {e}")


if __name__ == "__main__":
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    roster_count = len(WATCHLIST)
    print(f"=== Running ORVWB v6 Master Bot ({roster_count} Dynamic Stocks | {LOOKBACK_DAYS}d Lookback) for IST Date: {today_str} ===")
    smart_api = get_angel_session()
    scrip_df = load_token_map()
    all_new_trades = []
    missing_feeds = []
    active_tickers = {f"{item['symbol']}.{item['exchange']}" for item in WATCHLIST}

    for item in WATCHLIST:
        sym, exch = item["symbol"], item["exchange"]
        token = resolve_token(scrip_df, item)
        if not token:
            print(f"Skipping {sym}.{exch}: Token not found")
            missing_feeds.append(f"{sym}.{exch}")
            continue
        candles = fetch_5m_candles(smart_api, exch, token, days_back=LOOKBACK_DAYS)
        if candles.empty:
            missing_feeds.append(f"{sym}.{exch}")
            continue
        latest_bar = str(candles.index.max())
        sim_trades = simulate_orvwb_v6(candles, sym, exch)
        today_trades = [t for t in sim_trades if t["Date"] == today_str]
        print(f"{sym}.{exch} (Latest bar: {latest_bar}) -> {LOOKBACK_DAYS}d trades: {len(sim_trades)} | Today ({today_str}): {len(today_trades)}")
        all_new_trades.extend(sim_trades)

    new_df = pd.DataFrame(all_new_trades)
    if not new_df.empty:
        if os.path.exists(LEDGER_FILE):
            old_df = pd.read_csv(LEDGER_FILE)
            old_df = old_df[old_df["Ticker"].isin(active_tickers)]
            old_df = old_df[~old_df["Status"].astype(str).str.startswith("OPEN")]
            if "Base_1L_PnL" not in old_df.columns:
                old_df["Base_1L_PnL"] = np.nan
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(
                subset=["Date", "Ticker", "Side", "Entry_Time"], keep="last", inplace=True
            )
        else:
            combined_df = new_df

        combined_df = apply_adaptive_conviction_sizing(combined_df)
        combined_df.to_csv(LEDGER_FILE, index=False)

        today_df = combined_df[combined_df["Date"] == today_str]
        send_telegram_summary(today_str, today_df, combined_df, missing_feeds)
        print(f"Saved {roster_count}-stock {LOOKBACK_DAYS}d adaptive ledger! Total trades: {len(combined_df)} (Today {today_str}: {len(today_df)} trades). Net PnL: INR {combined_df['Net_PnL_INR'].sum():,.2f}")
    else:
        print("No trades generated across watchlist.")
