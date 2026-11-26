import os
import time
import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pyotp
import yfinance as yf
from SmartApi import SmartConnect

# ==============================================================================
# 1. CONFIGURATION (HYBRID: YFINANCE FULL F&O RVOL SCAN -> SMARTAPI TOP 9)
# ==============================================================================
LEDGER_FILE = "ORVWB_v6_Elite_Ledger.csv"
IST = timezone(timedelta(hours=5, minutes=30))

LOOKBACK_DAYS = 35          # ~24 trading days for 20-day 09:30 RVOL baseline (under 60d 5m cap)
TOP_N_RVOL = 9              # Select Top 9 stocks by 09:30 RVOL each day
MIN_RVOL_0930 = 1.25        # Minimum 09:15-09:25 relative volume to qualify
MIN_SHARE_PRICE = 100.0
YF_BATCH_SIZE = 45          # Batch size for yfinance 5m scan
SLEEP_SEC = 1.2             # Spacing for the 9 SmartAPI calls (~12s total)

# Immutable Paper-Trading Window Start Date (locks all closed paper trades since this date)
PAPER_TRADE_START_DATE = "2026-11-10"

# Realistic Friction & Capital Settings
CAPITAL_PER_TRADE = 100000.0
LIVE_START_CAPITAL_1X = 50000.0
MAX_DAILY_TRADES_CAP = 1
SLIPPAGE_PER_SIDE = 0.0005  # 0.05% per side = 0.10% round-trip slippage

CONVICTION_CAP_MAP = {0: 70000.0, 1: 85000.0, 2: 115000.0, 3: 135000.0}
CONVICTION_LABEL_MAP = {
    0: "Tier C (0.70x)",
    1: "Tier B (0.85x)",
    2: "Tier A (1.15x)",
    3: "Tier A+ (1.35x)",
}
TIER_PRIORITY_MAP = {
    "Tier A+ (1.35x)": 1,
    "Tier A (1.15x)": 2,
    "Tier B (0.85x)": 3,
    "Tier C (0.70x)": 4,
}


# ==============================================================================
# 2. EXACT ANGEL ONE INTRADAY CHARGES & SLIPPAGE
# ==============================================================================
def calculate_angel_one_charges(buy_val: float, sell_val: float, exchange: str = "NSE", orders: int = 2) -> float:
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


# ==============================================================================
# 3. UNIVERSE BUILDER, YFINANCE 5M RVOL SCANNER & SMARTAPI TOP-9 FETCHER
# ==============================================================================
def load_token_map() -> pd.DataFrame:
    urls = [
        "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json",
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    ]
    for url in urls:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    return pd.DataFrame(json.loads(response.read().decode()))
            except Exception as e:
                print(f"ScripMaster download attempt {attempt + 1} failed: {e}")
                time.sleep(3)
    raise RuntimeError("Failed to download OpenAPIScripMaster.json")


def build_nse_fno_universe(scrip_df: pd.DataFrame) -> list:
    nfo_futs = scrip_df[
        (scrip_df["exch_seg"] == "NFO") & (scrip_df["instrumenttype"] == "FUTSTK")
    ]
    fno_names = sorted(set(nfo_futs["name"].dropna().astype(str).str.strip()))

    nse_eq = scrip_df[
        (scrip_df["exch_seg"] == "NSE") & (scrip_df["symbol"].astype(str).str.endswith("-EQ"))
    ]
    eq_token_map = dict(zip(nse_eq["symbol"].astype(str), nse_eq["token"].astype(str)))

    universe = []
    for sym in fno_names:
        if sym in ("IDEA",):
            continue
        tok = eq_token_map.get(f"{sym}-EQ")
        if tok:
            universe.append({"symbol": sym, "exchange": "NSE", "token": tok})

    print(f"Auto-discovered {len(universe)} active NSE F&O cash equities (Full A-Z).")
    return universe


def compute_today_rvol(df: pd.DataFrame, today_str: str) -> tuple:
    """
    Returns (today_rvol, open_price) using strictly prior 20-day average 09:15-09:25 volume.
    """
    or_slice = df.between_time("09:15", "09:25")
    if or_slice.empty:
        return np.nan, np.nan

    or_vol_daily = or_slice.groupby(or_slice.index.date)["Volume"].sum()
    or_vol_daily.index = [str(d) for d in or_vol_daily.index]
    if today_str not in or_vol_daily.index:
        return np.nan, np.nan

    prior_vols = or_vol_daily.loc[:today_str].iloc[:-1].tail(20)
    if len(prior_vols) < 5:
        return np.nan, np.nan

    avg20 = float(prior_vols.mean())
    if avg20 <= 0:
        return np.nan, np.nan

    today_vol = float(or_vol_daily.loc[today_str])
    today_or_bars = or_slice[or_slice.index.strftime("%Y-%m-%d") == today_str]
    if len(today_or_bars) < 3:
        return np.nan, np.nan

    open_price = float(today_or_bars["Close"].iloc[-1])
    return round(today_vol / avg20, 2), open_price


def scan_rvol_yfinance(universe: list, today_str: str) -> tuple:
    """
    Fast batch 5m RVOL scan across the full A-Z NSE F&O universe via yfinance (~25s).
    """
    start_dt = (datetime.now(IST) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    qualified = []
    scanned_cnt = 0

    for i in range(0, len(universe), YF_BATCH_SIZE):
        batch = universe[i : i + YF_BATCH_SIZE]
        yf_tickers = [f"{item['symbol']}.NS" for item in batch]

        try:
            raw = yf.download(
                yf_tickers,
                start=start_dt,
                interval="5m",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[Warn] yfinance batch {i}-{i + len(batch)} failed: {e}")
            continue

        if raw is None or raw.empty:
            continue

        for item, y_t in zip(batch, yf_tickers):
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if y_t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[y_t].copy()
                else:
                    sub = raw.copy()

                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if sub.empty:
                    continue

                if sub.index.tz is None:
                    sub.index = sub.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
                else:
                    sub.index = sub.index.tz_convert("Asia/Kolkata")

                scanned_cnt += 1
                rvol_val, open_px = compute_today_rvol(sub, today_str)
                if not np.isnan(rvol_val) and open_px >= MIN_SHARE_PRICE and rvol_val >= MIN_RVOL_0930:
                    qualified.append({
                        "symbol": item["symbol"],
                        "exchange": item["exchange"],
                        "token": item["token"],
                        "rvol": rvol_val,
                    })
            except Exception:
                continue

        print(f"  Batch {min(i + YF_BATCH_SIZE, len(universe))}/{len(universe)} scanned | {len(qualified)} qualified")
        time.sleep(0.4)

    qualified.sort(key=lambda x: x["rvol"], reverse=True)
    return qualified, scanned_cnt


def get_angel_session() -> SmartConnect:
    api_key = (os.environ.get("ANGEL_API_KEY") or os.environ.get("ANGEL_HIST_API_KEY") or "").strip()
    client_code = os.environ["ANGEL_CLIENT_CODE"].strip()
    pin = os.environ["ANGEL_PIN"].strip()
    totp_secret = os.environ["ANGEL_TOTP_SECRET"].strip().replace(" ", "")

    if not api_key:
        raise RuntimeError("Missing ANGEL_API_KEY / ANGEL_HIST_API_KEY secret.")

    for attempt in range(4):
        try:
            smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            data = smart_api.generateSession(client_code, pin, totp)
            if data and data.get("status"):
                print(f"SmartAPI login successful for {client_code}.")
                return smart_api
            print(f"Login attempt {attempt + 1} failed: {data}. Retrying in 5s...")
        except Exception as e:
            print(f"Login exception on attempt {attempt + 1}: {e}")
        time.sleep(5)
    raise RuntimeError("Angel One Login Failed after 4 attempts.")


def fetch_5m_candles_single(smart_api: SmartConnect, exchange: str, token: str, days_back: int = LOOKBACK_DAYS) -> pd.DataFrame:
    now_ist = datetime.now(IST)
    from_dt = now_ist - timedelta(days=days_back)
    params = {
        "exchange": exchange,
        "symboltoken": str(token),
        "interval": "FIVE_MINUTE",
        "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
        "todate": now_ist.strftime("%Y-%m-%d 15:30"),
    }
    for attempt in range(4):
        try:
            time.sleep(SLEEP_SEC + attempt * 2.0)
            res = smart_api.getCandleData(params)
            if res and res.get("status") and res.get("data"):
                df = pd.DataFrame(
                    res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.drop_duplicates(subset=["Timestamp"], keep="last")
                df = df.sort_values("Timestamp")
                df = df.set_index("Timestamp")
                return df
            elif res and res.get("errorCode") in ("AB1004", "AB2001", "AB1021"):
                time.sleep(4.0 + attempt * 2.0)
            else:
                time.sleep(1.5)
        except Exception as e:
            print(f"[Warn] SmartAPI fetch try {attempt + 1} for token {token}: {e}")
            time.sleep(4.0)
    return pd.DataFrame()


# ==============================================================================
# 4. ORVWB v6 (ENGINE A ONLY) TODAY SIMULATOR
# ==============================================================================
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


def simulate_today_orvwb(df: pd.DataFrame, symbol: str, exchange: str, today_str: str, rvol_val: float) -> list:
    df = compute_indicators(df)
    day_df = df[df.index.strftime("%Y-%m-%d") == today_str]
    if day_df.empty:
        return []

    or_bars = day_df.between_time("09:15", "09:25")
    if len(or_bars) < 3:
        return []

    or_high = float(or_bars["High"].max())
    or_low = float(or_bars["Low"].min())
    or_mid = (or_high + or_low) * 0.5
    or_width = or_high - or_low

    active_bars = day_df.between_time("09:30", "15:15")
    if len(active_bars) < 1:
        return []

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
        c, h, l, o, v = (
            float(row["Close"]), float(row["High"]), float(row["Low"]),
            float(row["Open"]), float(row["Volume"]),
        )
        atr, vwap, ema20, avg_vol = (
            float(row["ATR"]), float(row["VWAP"]), float(row["EMA20"]), float(row["AVG_VOL"])
        )
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
                if not tp1_hit and h >= tp1_price:
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
                if not tp1_hit and l <= tp1_price:
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

        if pos == 0 and entry_time is None and t_str <= "11:10":
            if c < MIN_SHARE_PRICE:
                continue

            bar_range = max(h - l, 0.05)
            body_pct = abs(c - o) / bar_range
            or_valid = 0 < or_width <= (3.4 * atr)
            has_vol = v >= (1.10 * avg_vol)
            strong_body = body_pct >= 0.45
            bull_top = ((h - c) / bar_range) <= 0.35
            bear_bot = ((c - l) / bar_range) <= 0.35

            long_sig = (
                or_valid and has_vol and strong_body and bull_top
                and (c > or_high) and (c > vwap) and (c > ema20)
            )
            short_sig = (
                or_valid and has_vol and strong_body and bear_bot
                and (c < or_low) and (c < vwap) and (c < ema20)
            )

            if long_sig:
                pos = 1
                entry_time = t_str
                entry_price = c
                raw_sl = min(or_mid - 0.75 * atr, l - 0.30 * atr)
                active_stop = min(raw_sl, entry_price - 0.75 * atr)
                init_risk = max(entry_price - active_stop, 0.05)
                tp1_price = entry_price + 1.0 * init_risk
                tp2_price = entry_price + 2.5 * init_risk
            elif short_sig:
                pos = -1
                entry_time = t_str
                entry_price = c
                raw_sl = max(or_mid + 0.75 * atr, h + 0.30 * atr)
                active_stop = max(raw_sl, entry_price + 0.75 * atr)
                init_risk = max(active_stop - entry_price, 0.05)
                tp1_price = entry_price - 1.0 * init_risk
                tp2_price = entry_price - 2.5 * init_risk

    if entry_time is not None and pos != 0 and exit_reason == "":
        move = (last_close - entry_price) if pos == 1 else (entry_price - last_close)
        realized_r += rem_weight * (move / init_risk)
        exit_price = last_close
        exit_reason = "OPEN (TP1 Hit, Trailing)" if tp1_hit else "OPEN (Floating)"

    if entry_time is None or exit_reason == "":
        return []

    side_str = "LONG" if (tp2_price > entry_price) else "SHORT"
    per_share_gross = realized_r * init_risk
    base_pnl = compute_real_trade_pnl(
        entry_price, per_share_gross, side_str, exchange, exit_reason, CAPITAL_PER_TRADE
    )
    pnl_50k = compute_real_trade_pnl(
        entry_price, per_share_gross, side_str, exchange, exit_reason, LIVE_START_CAPITAL_1X
    )
    return [{
        "Date": today_str,
        "Ticker": f"{symbol}.{exchange}",
        "Engine": "Engine A (Top9 RVOL)",
        "Side": side_str,
        "Entry_Time": entry_time,
        "RVOL_0930": rvol_val,
        "Entry_Price": round(entry_price, 2),
        "Stop_Loss": round(active_stop, 2),
        "Target_2.5R": round(tp2_price, 2),
        "Exit_Price": round(exit_price, 2),
        "Result_R": round(realized_r, 2),
        "Base_1L_PnL": base_pnl,
        "Net_PnL_INR": base_pnl,
        "Net_PnL_50k_1x": pnl_50k,
        "Net_PnL_Max1_50k": pnl_50k,
        "Status": exit_reason,
    }]


# ==============================================================================
# 5. LEDGER MERGE (PRESERVES ALL PAST ROWS) & CONVICTION SIZING FOR TODAY
# ==============================================================================
def update_ledger_with_today(today_trades: list, today_str: str) -> tuple:
    if os.path.exists(LEDGER_FILE):
        old_df = pd.read_csv(LEDGER_FILE)
    else:
        old_df = pd.DataFrame()

    for col in ("Base_1L_PnL", "Net_PnL_50k_1x", "Net_PnL_Max1_50k", "Conviction", "RVOL_0930"):
        if not old_df.empty and col not in old_df.columns:
            old_df[col] = np.nan

    if not old_df.empty:
        kept_old = old_df[old_df["Date"].astype(str) != today_str].copy()
    else:
        kept_old = pd.DataFrame()

    if not today_trades:
        if not kept_old.empty:
            kept_old.to_csv(LEDGER_FILE, index=False)
        return pd.DataFrame(), kept_old

    today_df = pd.DataFrame(today_trades)
    today_df = today_df.sort_values(by=["Entry_Time", "Ticker"]).reset_index(drop=True)

    scores = []
    for idx, row in today_df.iterrows():
        ticker = row["Ticker"]
        is_prime = int(str(row["Entry_Time"]) <= "10:15")

        if not kept_old.empty:
            past_t = kept_old[kept_old["Ticker"] == ticker]
            prior_3_sum = float(past_t["Base_1L_PnL"].tail(3).sum()) if not past_t.empty else 0.0
            exp_mean_r = float(past_t["Result_R"].mean()) if len(past_t) >= 3 else 0.35
        else:
            prior_3_sum = 0.0
            exp_mean_r = 0.35

        sub = today_df[(today_df["Entry_Time"] <= row["Entry_Time"]) & (today_df.index != idx)]
        same_s = int((sub["Side"] == row["Side"]).sum())
        opp_s = int((sub["Side"] != row["Side"]).sum())
        breadth_adj = 1 if (same_s >= 1 and opp_s == 0) else (-1 if opp_s >= 1 else 0)

        sc = is_prime + int(exp_mean_r >= 0.30) + int(prior_3_sum <= 0.0) + breadth_adj
        scores.append(int(np.clip(sc, 0, 3)))

    today_df["Conviction"] = [CONVICTION_LABEL_MAP[s] for s in scores]

    init_risk = (today_df["Target_2.5R"] - today_df["Entry_Price"]).abs() / 2.5
    per_share_gross = today_df["Result_R"] * init_risk

    net_adap, net_50k = [], []
    for idx, r in today_df.iterrows():
        exch = "BSE" if ".BSE" in str(r["Ticker"]) else "NSE"
        cap = CONVICTION_CAP_MAP[scores[idx]]
        net_adap.append(
            compute_real_trade_pnl(
                float(r["Entry_Price"]), float(per_share_gross.loc[idx]),
                str(r["Side"]), exch, str(r["Status"]), cap,
            )
        )
        net_50k.append(
            compute_real_trade_pnl(
                float(r["Entry_Price"]), float(per_share_gross.loc[idx]),
                str(r["Side"]), exch, str(r["Status"]), LIVE_START_CAPITAL_1X,
            )
        )

    today_df["Net_PnL_INR"] = net_adap
    today_df["Net_PnL_50k_1x"] = net_50k

    today_df["_Tier_P"] = today_df["Conviction"].map(TIER_PRIORITY_MAP).fillna(9)
    today_df["_RVOL_D"] = -pd.to_numeric(today_df["RVOL_0930"], errors="coerce").fillna(0.0)
    today_df = today_df.sort_values(by=["Entry_Time", "_Tier_P", "_RVOL_D", "Ticker"]).reset_index(drop=True)

    today_df["Net_PnL_Max1_50k"] = np.where(
        np.arange(len(today_df)) < MAX_DAILY_TRADES_CAP, today_df["Net_PnL_50k_1x"], 0.0
    ).round(2)
    today_df = today_df.drop(columns=["_Tier_P", "_RVOL_D"])

    combined_df = pd.concat([kept_old, today_df], ignore_index=True)
    combined_df.to_csv(LEDGER_FILE, index=False)
    return today_df, combined_df


# ==============================================================================
# 6. TELEGRAM REPORTING & MAIN RUNNER
# ==============================================================================
def _calc_window_metrics(sub_df: pd.DataFrame, pnl_col: str) -> tuple:
    if sub_df.empty:
        return 0, 0.0, 0.0, 0.0, 0.0
    n = len(sub_df)
    wr = (sub_df[pnl_col] > 0).mean() * 100
    gw = sub_df[sub_df[pnl_col] > 0][pnl_col].sum()
    gl = abs(sub_df[sub_df[pnl_col] <= 0][pnl_col].sum())
    pf = (gw / gl) if gl > 0 else 99.0
    tot = float(sub_df[pnl_col].sum())
    daily_eq = sub_df.groupby("Date")[pnl_col].sum().cumsum()
    max_dd = float((daily_eq - daily_eq.cummax()).min())
    return n, wr, pf, tot, max_dd


def send_telegram_text(msg: str):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        print("Telegram credentials missing; printing message to console:\n" + msg)
        return
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"Telegram alert skipped: {e}")


def send_telegram_summary(
    today_str: str,
    scanned_cnt: int,
    qualified_cnt: int,
    top9_list: list,
    today_df: pd.DataFrame,
    combined_df: pd.DataFrame,
):
    lines = [f"📊 *ORVWB v6 Hybrid Auto-F&O → Top {TOP_N_RVOL} RVOL ({today_str})*"]
    lines.append(f"_(Scanned {scanned_cnt} NSE F&O | {qualified_cnt} ≥ {MIN_RVOL_0930}x RVOL | SmartAPI Top-{TOP_N_RVOL} Fills)_")

    if top9_list:
        top_str = ", ".join(f"{x['symbol']} ({x['rvol']:.2f}x)" for x in top9_list)
        lines.append(f"• *Today's Top {len(top9_list)} RVOL Basket:* {top_str}")
    else:
        lines.append("• *Today's Basket:* No stocks qualified (market holiday or no intraday bars yet).")

    if today_df.empty:
        lines.append(f"• *Today's Action:* No ORVWB setups triggered among today's Top {TOP_N_RVOL} RVOL stocks.")
    else:
        for slot_idx, (_, r) in enumerate(today_df.iterrows(), start=1):
            tier_str = r.get("Conviction", "Tier A")
            pnl_50k_val = r.get("Net_PnL_50k_1x", 0.0)
            rv_val = r.get("RVOL_0930", np.nan)
            rv_tag = f"RVOL {rv_val:.2f}x" if pd.notna(rv_val) else "RVOL N/A"
            slot_tag = f"Slot #{slot_idx}/{MAX_DAILY_TRADES_CAP}" if slot_idx <= MAX_DAILY_TRADES_CAP else "Cap Skip"
            lines.append(
                f"• `{r['Ticker']}` [{tier_str} | {rv_tag} | {slot_tag}] ({r['Side']} @ {r['Entry_Time']} | "
                f"Entry: ₹{r['Entry_Price']} | SL: ₹{r['Stop_Loss']}): "
                f"{r['Result_R']:+.2f}R | ₹{r['Net_PnL_INR']:+.2f} (50k: ₹{pnl_50k_val:+.2f}) [{r['Status']}]"
            )
        lines.append(
            f"• *Today's Net (₹1L Adap):* ₹{today_df['Net_PnL_INR'].sum():+,.2f}\n"
            f"• *Today's Net (₹50k All):* ₹{today_df['Net_PnL_50k_1x'].sum():+,.2f}\n"
            f"• *Today's Net (Max {MAX_DAILY_TRADES_CAP}/d @ ₹50k 1x):* ₹{today_df['Net_PnL_Max1_50k'].sum():+,.2f}"
        )

    if not combined_df.empty:
        paper_df = combined_df[combined_df["Date"].astype(str) >= PAPER_TRADE_START_DATE]
        if not paper_df.empty:
            p_days = paper_df["Date"].nunique()
            p_n, p_wr, p_pf, p_1l, p_dd = _calc_window_metrics(paper_df, "Net_PnL_INR")
            p_50k = paper_df["Net_PnL_50k_1x"].sum()

            p_seq = paper_df.groupby("Date").cumcount() + 1
            p_cap_df = paper_df[p_seq <= MAX_DAILY_TRADES_CAP]
            pc_n, pc_wr, pc_pf, pc_tot, pc_dd = _calc_window_metrics(p_cap_df, "Net_PnL_Max1_50k")

            lines.append(
                f"\n🔒 *Live Paper Window (Since {PAPER_TRADE_START_DATE} | {p_days}d):*\n"
                f"• *All Signals ({p_n}):* WR {p_wr:.1f}% | PF {p_pf:.2f} | ₹1L: ₹{p_1l:+,.2f} | ₹50k: ₹{p_50k:+,.2f}\n"
                f"• *Max {MAX_DAILY_TRADES_CAP}/d @ ₹50k 1x ({pc_n}):* WR {pc_wr:.1f}% | PF {pc_pf:.2f} | Net: ₹{pc_tot:+,.2f} | DD: ₹{pc_dd:,.2f}"
            )

    send_telegram_text("\n".join(lines))


def main():
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    print(f"=== ORVWB v6 Hybrid Auto-F&O -> Top {TOP_N_RVOL} RVOL Bot ({today_str}) ===")

    scrip_df = load_token_map()
    universe = build_nse_fno_universe(scrip_df)

    # Stage 1: Full A-Z RVOL scan via yfinance (~25s, zero SmartAPI rate limits)
    qualified, scanned_cnt = scan_rvol_yfinance(universe, today_str)
    top9 = qualified[:TOP_N_RVOL]

    print(f"\nTop {len(top9)} 09:30 RVOL Leaders for {today_str}:")
    for rank, q in enumerate(top9, start=1):
        print(f"  #{rank} {q['symbol']}.NSE — RVOL {q['rvol']:.2f}x")

    # Stage 2: Exact broker 5m simulation on Top 9 only via SmartAPI (~12s)
    today_trades = []
    if top9:
        smart_api = get_angel_session()
        for q in top9:
            candles = fetch_5m_candles_single(smart_api, q["exchange"], q["token"], days_back=LOOKBACK_DAYS)
            if candles.empty:
                print(f"  [Warn] No SmartAPI candles returned for {q['symbol']}.NSE; skipping.")
                continue
            t_list = simulate_today_orvwb(candles, q["symbol"], q["exchange"], today_str, q["rvol"])
            today_trades.extend(t_list)

    today_df, combined_df = update_ledger_with_today(today_trades, today_str)
    send_telegram_summary(today_str, scanned_cnt, len(qualified), top9, today_df, combined_df)
    print(f"Done. Today's setups: {len(today_trades)} | Ledger rows: {len(combined_df)}")


if __name__ == "__main__":
    main()
