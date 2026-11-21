import os
import sys
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

# Immutable Paper-Trading Window Start Date (prevents survivorship & lookahead bias across screener rotations)
PAPER_TRADE_START_DATE = "2026-11-10"

# Master Switch: Keep False during 60-Trading-Day Shadow-Live Testing!
LIVE_ORDER_EXECUTION = False

# Realistic Friction & Capital Settings
CAPITAL_PER_TRADE = 100000.0
LIVE_START_CAPITAL_1X = 50000.0
SLIPPAGE_PER_SIDE = 0.0005  # 0.05% per side = 0.10% round-trip slippage

# Fallback 16-Stock Institutional Portfolio
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

CONVICTION_CAP_MAP = {0: 70000.0, 1: 85000.0, 2: 115000.0, 3: 135000.0}
CONVICTION_LABEL_MAP = {
    0: "Tier C (0.70x)",
    1: "Tier B (0.85x)",
    2: "Tier A (1.15x)",
    3: "Tier A+ (1.35x)",
}


def calculate_angel_one_charges(buy_val: float, sell_val: float, exchange: str = "NSE", orders: int = 2) -> float:
    """
    Exact Angel One Equity Intraday Charges:
    - Brokerage: Lower of ₹20 or 0.1% per executed order (min ₹5/order)
    - STT: 0.025% on sell side
    - Exchange Txn: NSE 0.00307%, BSE 0.00375% on total turnover
    - SEBI: 0.0001% (₹10/Cr) on total turnover
    - Stamp Duty: 0.003% on buy side
    - GST: 18% on (Brokerage + Exchange Txn + SEBI)
    """
    turnover = buy_val + sell_val
    if turnover <= 0:
        return 0.0

    if orders == 3:
        b1 = max(5.0, min(20.0, buy_val * 0.001))
        b2 = max(5.0, min(20.0, (sell_val * 0.35) * 0.001))
        b3 = max(5.0, min(20.0, (sell_val * 0.65) * 0.001))
        brokerage = b1 + b2 + b3
    else:
        b1 = max(5.0, min(20.0, buy_val * 0.001))
        b2 = max(5.0, min(20.0, sell_val * 0.001))
        brokerage = b1 + b2

    stt = sell_val * 0.00025
    exch_rate = 0.0000307 if exchange == "NSE" else 0.0000375
    exch_txn = turnover * exch_rate
    sebi = turnover * 0.000001
    stamp = buy_val * 0.00003
    gst = 0.18 * (brokerage + exch_txn + sebi)

    return brokerage + stt + exch_txn + sebi + stamp + gst


def compute_real_trade_pnl(
    entry_price: float,
    per_share_gross: float,
    side: str,
    exchange: str,
    status: str,
    capital: float,
) -> float:
    qty = int(capital // entry_price)
    if qty < 1:
        return 0.0

    entry_val = qty * entry_price
    gross_pnl = per_share_gross * qty
    exit_val = (entry_val + gross_pnl) if side == "LONG" else (entry_val - gross_pnl)
    buy_val = entry_val if side == "LONG" else exit_val
    sell_val = exit_val if side == "LONG" else entry_val

    orders = 3 if ("TP1" in str(status) or "TP2" in str(status)) else 2
    charges = calculate_angel_one_charges(buy_val, sell_val, exchange, orders)
    slippage = (buy_val + sell_val) * SLIPPAGE_PER_SIDE
    return round(gross_pnl - charges - slippage, 2)


def merge_ledger_with_paper_lock(
    new_df: pd.DataFrame, active_tickers: set, today_str: str
) -> pd.DataFrame:
    """
    Merges newly simulated trades with existing ledger while locking all closed
    live paper trades (>= PAPER_TRADE_START_DATE):
    - Never deletes rotated-out tickers from the paper-trading window.
    - Never retroactively injects newly added tickers into past paper-trading dates.
    - Allows pre-paper backtest history (< PAPER_TRADE_START_DATE), today's live trades,
      and any prior dates with leftover OPEN rows to refresh cleanly.
    """
    if not os.path.exists(LEDGER_FILE):
        return new_df.copy()

    old_df = pd.read_csv(LEDGER_FILE)
    if old_df.empty:
        return new_df.copy()

    if "Base_1L_PnL" not in old_df.columns:
        old_df["Base_1L_PnL"] = np.nan
    if "Net_PnL_50k_1x" not in old_df.columns:
        old_df["Net_PnL_50k_1x"] = np.nan

    old_dates = old_df["Date"].astype(str)
    old_is_open = old_df["Status"].astype(str).str.startswith("OPEN")
    open_dates = set(old_dates[old_is_open])

    is_locked_paper = (old_dates >= PAPER_TRADE_START_DATE) & (old_dates != today_str) & (~old_is_open)
    is_active_backtest = (old_dates < PAPER_TRADE_START_DATE) & (old_df["Ticker"].isin(active_tickers))
    kept_old = old_df[is_locked_paper | is_active_backtest].copy()

    has_locked_paper = (old_dates >= PAPER_TRADE_START_DATE).any()
    if has_locked_paper and not new_df.empty:
        new_dates = new_df["Date"].astype(str)
        allowed_new = (
            (new_dates < PAPER_TRADE_START_DATE)
            | (new_dates == today_str)
            | (new_dates.isin(open_dates))
        )
        filtered_new = new_df[allowed_new].copy()
    else:
        filtered_new = new_df.copy()

    combined_df = pd.concat([kept_old, filtered_new], ignore_index=True)
    combined_df.drop_duplicates(subset=["Date", "Ticker", "Side", "Entry_Time"], keep="last", inplace=True)
    return combined_df


def get_angel_session():
    api_key = os.environ["ANGEL_API_KEY"].strip()
    client_code = os.environ["ANGEL_CLIENT_CODE"].strip()
    pin = os.environ["ANGEL_PIN"].strip()
    totp_secret = os.environ["ANGEL_TOTP_SECRET"].strip().replace(" ", "")

    for attempt in range(4):
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
    raise RuntimeError("Angel One Login Failed after 4 attempts.")


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
    base_sleep = 0.35 if days_back <= 10 else 1.0

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
                time.sleep(base_sleep + attempt * 1.0)
                res = smart_api.getCandleData(params)
                if res and res.get("status") and res.get("data"):
                    chunk_df = pd.DataFrame(
                        res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                    )
                    frames.append(chunk_df)
                    break
                elif res and res.get("errorCode") in ("AB1004", "AB2001"):
                    time.sleep(3.0 + attempt * 1.5)
                else:
                    time.sleep(1.2)
            except Exception:
                time.sleep(2.0)

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
            side_str = "LONG" if (tp2_price > entry_price) else "SHORT"
            per_share_gross = realized_r * init_risk
            base_pnl = compute_real_trade_pnl(entry_price, per_share_gross, side_str, exchange, exit_reason, CAPITAL_PER_TRADE)
            pnl_50k = compute_real_trade_pnl(entry_price, per_share_gross, side_str, exchange, exit_reason, LIVE_START_CAPITAL_1X)
            trades.append({
                "Date": str(trade_date),
                "Ticker": f"{symbol}.{exchange}",
                "Engine": engine_name,
                "Side": side_str,
                "Entry_Time": entry_time,
                "Entry_Price": round(entry_price, 2),
                "Stop_Loss": round(active_stop, 2),
                "Target_2.5R": round(tp2_price, 2),
                "Exit_Price": round(exit_price, 2),
                "Result_R": round(realized_r, 2),
                "Base_1L_PnL": base_pnl,
                "Net_PnL_INR": base_pnl,
                "Net_PnL_50k_1x": pnl_50k,
                "Status": exit_reason,
            })
    return trades


def apply_adaptive_conviction_sizing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(by=["Date", "Entry_Time"]).reset_index(drop=True)

    # Drop legacy 20k column if upgrading from an older ledger CSV
    if "Net_PnL_20k_1x" in df.columns:
        df.drop(columns=["Net_PnL_20k_1x"], inplace=True)

    # Recompute exact realistic friction (Angel One brokerage + taxes + 0.10% RT slippage) across all rows
    init_risk = (df["Target_2.5R"] - df["Entry_Price"]).abs() / 2.5
    per_share_gross = df["Result_R"] * init_risk

    base_1l_list = []
    for idx, r in df.iterrows():
        exch = "BSE" if ".BSE" in str(r["Ticker"]) else "NSE"
        base_1l_list.append(
            compute_real_trade_pnl(
                float(r["Entry_Price"]),
                float(per_share_gross.loc[idx]),
                str(r["Side"]),
                exch,
                str(r["Status"]),
                CAPITAL_PER_TRADE,
            )
        )
    df["Base_1L_PnL"] = base_1l_list

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

    df["Conviction"] = scores.map(CONVICTION_LABEL_MAP)
    trade_cap = scores.map(CONVICTION_CAP_MAP)

    net_adap_list = []
    net_50k_list = []
    for idx, r in df.iterrows():
        exch = "BSE" if ".BSE" in str(r["Ticker"]) else "NSE"
        net_adap_list.append(
            compute_real_trade_pnl(
                float(r["Entry_Price"]),
                float(per_share_gross.loc[idx]),
                str(r["Side"]),
                exch,
                str(r["Status"]),
                float(trade_cap.loc[idx]),
            )
        )
        net_50k_list.append(
            compute_real_trade_pnl(
                float(r["Entry_Price"]),
                float(per_share_gross.loc[idx]),
                str(r["Side"]),
                exch,
                str(r["Status"]),
                LIVE_START_CAPITAL_1X,
            )
        )

    df["Net_PnL_INR"] = net_adap_list
    df["Net_PnL_50k_1x"] = net_50k_list
    return df


def send_telegram_text(msg: str):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"Telegram alert skipped: {e}")


def send_telegram_summary(today_str: str, today_df: pd.DataFrame, combined_df: pd.DataFrame, missing_feeds: list):
    roster_count = len(WATCHLIST)
    total_pnl = combined_df["Net_PnL_INR"].sum() if not combined_df.empty else 0.0
    total_50k = combined_df["Net_PnL_50k_1x"].sum() if (not combined_df.empty and "Net_PnL_50k_1x" in combined_df.columns) else 0.0
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
    lines.append("_(Net of Angel One Brokerage, Taxes & 0.10% RT Slippage)_")
    if today_df.empty:
        lines.append(f"• *Today's Action:* No setups triggered across the {roster_count}-stock watchlist today.")
    else:
        for _, r in today_df.iterrows():
            tier_str = r.get("Conviction", "Tier A")
            pnl_50k_val = r.get("Net_PnL_50k_1x", 0.0)
            lines.append(
                f"• `{r['Ticker']}` [{tier_str}] ({r['Side']} @ {r['Entry_Time']} | Entry: ₹{r['Entry_Price']} | SL: ₹{r['Stop_Loss']}): "
                f"{r['Result_R']:+.2f}R | ₹{r['Net_PnL_INR']:+.2f} (50k 1x: ₹{pnl_50k_val:+.2f}) [{r['Status']}]"
            )
        lines.append(
            f"• *Today's Net PnL (₹1L Adap):* ₹{today_df['Net_PnL_INR'].sum():+,.2f}\n"
            f"• *Today's Net PnL (₹50k 1x):* ₹{today_df['Net_PnL_50k_1x'].sum():+,.2f}"
        )

    # Dedicated Locked Paper-Trading Window Telemetry
    if not combined_df.empty:
        paper_df = combined_df[combined_df["Date"].astype(str) >= PAPER_TRADE_START_DATE]
        if not paper_df.empty:
            p_days = paper_df["Date"].nunique()
            p_wr = (paper_df["Net_PnL_INR"] > 0).mean() * 100
            p_gw = paper_df[paper_df["Net_PnL_INR"] > 0]["Net_PnL_INR"].sum()
            p_gl = abs(paper_df[paper_df["Net_PnL_INR"] <= 0]["Net_PnL_INR"].sum())
            p_pf = (p_gw / p_gl) if p_gl > 0 else 99.0
            p_1l = paper_df["Net_PnL_INR"].sum()
            p_50k = paper_df["Net_PnL_50k_1x"].sum()
            lines.append(
                f"\n🔒 *Live Paper Window (Since {PAPER_TRADE_START_DATE} | {p_days}d):*\n"
                f"• *Trades:* {len(paper_df)} | *WR:* {p_wr:.1f}% | *PF:* {p_pf:.2f}\n"
                f"• *Net (₹1L Adap):* ₹{p_1l:+,.2f} | *Net (₹50k 1x):* ₹{p_50k:+,.2f}"
            )

    lines.append(
        f"\n📈 *99d Real-World Ledger Telemetry ({roster_count} Active Stocks):*\n"
        f"• *Total Trades:* {len(combined_df)} | *Win Rate:* {win_rate:.1f}% | *PF:* {pf:.2f}\n"
        f"• *Cumulative Net (₹1L Adap):* ₹{total_pnl:,.2f} | *Max DD:* ₹{max_dd:,.2f}\n"
        f"• *Cumulative Net (₹50k 1x):* ₹{total_50k:,.2f}"
    )

    if missing_feeds:
        lines.append(f"⚠️ *API Watchdog:* Missing feed on {', '.join(missing_feeds)} (preserved from ledger)")

    send_telegram_text("\n".join(lines))


def run_single_pass(smart_api, scrip_df: pd.DataFrame, today_str: str) -> tuple:
    all_new_trades = []
    missing_feeds = []
    active_tickers = {f"{item['symbol']}.{item['exchange']}" for item in WATCHLIST}

    for item in WATCHLIST:
        sym, exch = item["symbol"], item["exchange"]
        token = resolve_token(scrip_df, item)
        if not token:
            missing_feeds.append(f"{sym}.{exch}")
            continue
        candles = fetch_5m_candles(smart_api, exch, token, days_back=LOOKBACK_DAYS)
        if candles.empty:
            missing_feeds.append(f"{sym}.{exch}")
            continue
        sim_trades = simulate_orvwb_v6(candles, sym, exch)
        all_new_trades.extend(sim_trades)

    new_df = pd.DataFrame(all_new_trades)
    if new_df.empty:
        return pd.DataFrame(), pd.DataFrame(), missing_feeds

    combined_df = merge_ledger_with_paper_lock(new_df, active_tickers, today_str)
    combined_df = apply_adaptive_conviction_sizing(combined_df)
    combined_df.to_csv(LEDGER_FILE, index=False)
    today_df = combined_df[combined_df["Date"] == today_str]
    return today_df, combined_df, missing_feeds


def run_continuous_live_loop():
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    roster_count = len(WATCHLIST)

    # If triggered after 15:16 IST (e.g., manual EOD reconciliation), run a single pass to settle open trades
    if now_ist.weekday() < 5 and now_ist.strftime("%H:%M") >= "15:16":
        print(f"Post-close run detected (>= 15:16 IST) for {today_str}. Running EOD reconciliation pass...")
        smart_api = get_angel_session()
        scrip_df = load_token_map()
        today_df, combined_df, mf = run_single_pass(smart_api, scrip_df, today_str)
        if not combined_df.empty:
            send_telegram_summary(today_str, today_df, combined_df, mf)
        return

    print(f"=== Starting Continuous Shadow-Live Engine ({roster_count} Stocks) for {today_str} ===")

    smart_api = get_angel_session()
    scrip_df = load_token_map()

    # Step 1: Pre-Market 99-Day Historical Baseline Cache + Initial Today Cache
    token_cache = {}
    historical_trades = []
    today_trade_cache = {}  # Persistent intraday RAM cache so EOD API timeouts never drop morning trades
    missing_feeds = []

    for item in WATCHLIST:
        sym, exch = item["symbol"], item["exchange"]
        t_key = f"{sym}.{exch}"
        token = resolve_token(scrip_df, item)
        if not token:
            missing_feeds.append(t_key)
            continue
        token_cache[t_key] = (sym, exch, token)
        candles_99d = fetch_5m_candles(smart_api, exch, token, days_back=LOOKBACK_DAYS)
        if not candles_99d.empty:
            sim_t = simulate_orvwb_v6(candles_99d, sym, exch)
            historical_trades.extend([t for t in sim_t if t["Date"] != today_str])
            today_init = [t for t in sim_t if t["Date"] == today_str]
            if today_init:
                today_trade_cache[t_key] = today_init
            print(f"Pre-loaded 99d history for {t_key}: {len(sim_t)} setups.")
        else:
            missing_feeds.append(t_key)

    now_ist = datetime.now(IST)
    if now_ist.weekday() >= 5:
        print("Weekend run detected. Running single verification pass...")
        today_df, combined_df, mf = run_single_pass(smart_api, scrip_df, today_str)
        if not combined_df.empty:
            send_telegram_summary(today_str, today_df, combined_df, mf)
        return

    send_telegram_text(
        f"🟢 *ORVWB v6 Shadow-Live Engine Armed ({today_str})*\n"
        f"• *Active Watchlist:* {len(token_cache)} Institutional Leaders\n"
        f"• *Cost Model:* Exact Angel One Taxes + 0.10% RT Slippage (₹50k 1x)\n"
        f"• *Paper Lock:* Immutable since `{PAPER_TRADE_START_DATE}`\n"
        f"• *Mode:* Continuous 5-Minute Live Polling (`09:30` – `15:15 IST`)"
    )

    alerted_states = {}
    active_tickers = {f"{item['symbol']}.{item['exchange']}" for item in WATCHLIST}

    while True:
        now_ist = datetime.now(IST)
        hm_str = now_ist.strftime("%H:%M")

        if hm_str < "09:35":
            target_935 = now_ist.replace(hour=9, minute=35, second=2, microsecond=0)
            wait_s = max((target_935 - now_ist).total_seconds(), 1.0)
            print(f"[{now_ist.strftime('%H:%M:%S')}] Waiting {wait_s:.0f}s for 09:35:02 IST first breakout candle close...")
            time.sleep(min(wait_s, 30.0))
            continue

        if hm_str == "12:30" and now_ist.second < 15:
            try:
                smart_api = get_angel_session()
            except Exception as e:
                print(f"Midday session refresh warning: {e}")

        # Fast 5-Day Intraday Poll across all active stocks with Persistent Intraday Cache
        for t_key, (sym, exch, token) in token_cache.items():
            candles_5d = fetch_5m_candles(smart_api, exch, token, days_back=5)
            if candles_5d.empty:
                continue
            sim_recent = simulate_orvwb_v6(candles_5d, sym, exch)
            today_t = [t for t in sim_recent if t["Date"] == today_str]
            if today_t:
                today_trade_cache[t_key] = today_t

        today_live_trades = []
        for cached_list in today_trade_cache.values():
            today_live_trades.extend(cached_list)

        all_trades = historical_trades + today_live_trades
        new_df = pd.DataFrame(all_trades)
        if not new_df.empty:
            combined_df = merge_ledger_with_paper_lock(new_df, active_tickers, today_str)
            combined_df = apply_adaptive_conviction_sizing(combined_df)
            combined_df.to_csv(LEDGER_FILE, index=False)
            today_df = combined_df[combined_df["Date"] == today_str]

            new_events = []
            for _, r in today_df.iterrows():
                t_k = f"{r['Ticker']}_{r['Entry_Time']}"
                cur_status = str(r["Status"])
                if alerted_states.get(t_k) != cur_status:
                    alerted_states[t_k] = cur_status
                    new_events.append(
                        f"⚡ *Live Signal Update ({hm_str} IST)*\n"
                        f"• `{r['Ticker']}` [{r['Conviction']}] ({r['Side']} @ {r['Entry_Time']})\n"
                        f"• *Entry:* ₹{r['Entry_Price']} | *SL:* ₹{r['Stop_Loss']} | *TP2:* ₹{r['Target_2.5R']}\n"
                        f"• *Current Status:* `{cur_status}` ({r['Result_R']:+.2f}R | ₹{r['Net_PnL_INR']:+.2f} | 50k 1x: ₹{r['Net_PnL_50k_1x']:+.2f})"
                    )
            if new_events:
                send_telegram_text("\n\n".join(new_events))

        if hm_str >= "15:15":
            print("15:15 IST EOD Square-Off reached. Sending final Command Report...")
            today_df = combined_df[combined_df["Date"] == today_str] if not new_df.empty else pd.DataFrame()
            send_telegram_summary(today_str, today_df, combined_df, missing_feeds)
            break

        now_ist = datetime.now(IST)
        next_min = (now_ist.minute // 5 + 1) * 5
        if next_min >= 60:
            next_run = (now_ist + timedelta(hours=1)).replace(minute=0, second=2, microsecond=0)
        else:
            next_run = now_ist.replace(minute=next_min, second=2, microsecond=0)
        sleep_secs = max((next_run - now_ist).total_seconds(), 2.0)
        print(f"[{now_ist.strftime('%H:%M:%S')}] Scan complete. Today trades: {len(today_live_trades)}. Sleeping {sleep_secs:.1f}s until {next_run.strftime('%H:%M:%S')} IST...")
        time.sleep(sleep_secs)


if __name__ == "__main__":
    if "--live-loop" in sys.argv:
        run_continuous_live_loop()
    else:
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        print(f"=== Running ORVWB v6 Single-Pass Bot ({len(WATCHLIST)} Stocks) for {today_str} ===")
        smart_api = get_angel_session()
        scrip_df = load_token_map()
        today_df, combined_df, missing_feeds = run_single_pass(smart_api, scrip_df, today_str)
        if not combined_df.empty:
            send_telegram_summary(today_str, today_df, combined_df, missing_feeds)
