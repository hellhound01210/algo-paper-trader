import os
import time
import urllib.request
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyotp
from SmartApi import SmartConnect

SCREENER_FILE = "ORVWB_Screener_Ranked.csv"

CANDIDATE_UNIVERSE = [
    "RELIANCE", "MCX", "PAYTM", "BSE", "ADANIENT", "HDFCAMC", "LT",
    "ETERNAL", "TRENT", "HAL", "BEL", "DIXON", "POLYCAB", "PERSISTENT",
    "COFORGE", "OFSS", "CDSL", "ANGELONE", "MOTILALOFS", "CHOLAFIN",
    "BAJFINANCE", "BAJAJFINSV", "SBIN", "ICICIBANK", "AXISBANK", "INDUSINDBK",
    "TMPV", "M&M", "MARUTI", "BAJAJ-AUTO", "TVSMOTOR", "EICHERMOT",
    "TATASTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "DLF", "GODREJPROP",
    "SUNPHARMA", "DIVISLAB", "APOLLOHOSP", "TITAN", "BHARTIARTL", "INFY", "TCS"
]

BSE_TOKEN_OVERRIDES = {
    "PAYTM": "543396",
    "MCX": "534091",
}

PSU_GRINDERS = ["SBIN", "BANKBARODA", "PNB", "CANBK", "NTPC", "POWERGRID", "COALINDIA"]
CAPITAL_PER_TRADE = 100000.0
COMMISSION_PCT = 0.0003


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


def resolve_token(scrip_df: pd.DataFrame, symbol: str, exchange: str) -> str:
    if exchange == "NSE":
        sub = scrip_df[(scrip_df["exch_seg"] == "NSE") & (scrip_df["symbol"] == f"{symbol}-EQ")]
    else:
        if symbol in BSE_TOKEN_OVERRIDES:
            return BSE_TOKEN_OVERRIDES[symbol]
        sub = scrip_df[
            (scrip_df["exch_seg"] == "BSE") &
            ((scrip_df["name"] == symbol) | (scrip_df["symbol"] == symbol))
        ]
    if sub.empty:
        return ""
    return str(sub.iloc[0]["token"])


def fetch_5m_candles(smart_api, exchange: str, token: str, days_back: int = 90) -> pd.DataFrame:
    now = datetime.now()
    frames = []
    chunks = [(90, 45), (45, 0)] if days_back > 60 else [(days_back, 0)]

    for start_d, end_d in chunks:
        from_dt = now - timedelta(days=start_d)
        to_dt = now - timedelta(days=end_d)
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": to_dt.strftime("%Y-%m-%d 15:30"),
        }
        for attempt in range(3):
            try:
                time.sleep(0.85)
                res = smart_api.getCandleData(params)
                if res and res.get("status") and res.get("data"):
                    chunk_df = pd.DataFrame(
                        res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                    )
                    frames.append(chunk_df)
                    break
                elif res and res.get("errorCode") == "AB1004":
                    print(f"Rate limit hit on {exchange} {token}, cooling down 3s (attempt {attempt + 1})...")
                    time.sleep(3.0)
                else:
                    break
            except Exception as e:
                print(f"Chunk fetch retry {attempt + 1} ({exchange} {token}): {e}")
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


def simulate_stock(df: pd.DataFrame, symbol: str, exchange: str) -> dict:
    if df.empty:
        return {}
    df = compute_indicators(df)
    use_eng_a = not any(psu in symbol.upper() for psu in PSU_GRINDERS)
    r_list, pnl_list = [], []

    for _, day_df in df.groupby("Date"):
        or_bars = day_df.between_time("09:15", "09:25")
        if len(or_bars) < 3:
            continue

        or_high, or_low = float(or_bars["High"].max()), float(or_bars["Low"].min())
        or_mid = (or_high + or_low) * 0.5
        or_width = or_high - or_low
        entry_end = "11:10" if use_eng_a else "11:25"
        active_bars = day_df.between_time("09:30", "15:15")
        if len(active_bars) < 5:
            continue

        pos = 0
        entry_price = active_stop = tp1_price = tp2_price = init_risk = 0.0
        tp1_hit = False
        entry_time = None
        realized_r = 0.0
        rem_weight = 1.0
        closed_trade = False

        for ts, row in active_bars.iterrows():
            t_str = ts.strftime("%H:%M")
            c, h, l, o, v = float(row["Close"]), float(row["High"]), float(row["Low"]), float(row["Open"]), float(row["Volume"])
            atr, vwap, ema20, avg_vol = float(row["ATR"]), float(row["VWAP"]), float(row["EMA20"]), float(row["AVG_VOL"])

            if np.isnan(atr) or np.isnan(avg_vol) or atr <= 0:
                continue

            if pos != 0:
                if pos == 1:
                    if l <= active_stop:
                        realized_r += rem_weight * ((active_stop - entry_price) / init_risk)
                        closed_trade = True
                        break
                    if use_eng_a and not tp1_hit and h >= tp1_price:
                        realized_r += 0.35 * 1.0
                        rem_weight = 0.65
                        active_stop = entry_price + (0.15 * init_risk)
                        tp1_hit = True
                    if h >= tp2_price:
                        realized_r += rem_weight * 2.5
                        closed_trade = True
                        break
                elif pos == -1:
                    if h >= active_stop:
                        realized_r += rem_weight * ((entry_price - active_stop) / init_risk)
                        closed_trade = True
                        break
                    if use_eng_a and not tp1_hit and l <= tp1_price:
                        realized_r += 0.35 * 1.0
                        rem_weight = 0.65
                        active_stop = entry_price - (0.15 * init_risk)
                        tp1_hit = True
                    if l <= tp2_price:
                        realized_r += rem_weight * 2.5
                        closed_trade = True
                        break

                if t_str >= "15:10":
                    move = (c - entry_price) if pos == 1 else (entry_price - c)
                    realized_r += rem_weight * (move / init_risk)
                    closed_trade = True
                    break

            if pos == 0 and entry_time is None and t_str <= entry_end:
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

        if entry_time is not None and closed_trade:
            qty = int(CAPITAL_PER_TRADE // entry_price)
            net_pnl = (realized_r * init_risk * qty) - (CAPITAL_PER_TRADE * COMMISSION_PCT * 2)
            r_list.append(realized_r)
            pnl_list.append(net_pnl)

    if not pnl_list:
        return {}

    wins = sum(1 for p in pnl_list if p > 0)
    gross_win = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p <= 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0

    return {
        "Ticker": f"{symbol}.{exchange}",
        "Trades_90d": len(pnl_list),
        "Win_Rate_Pct": round((wins / len(pnl_list)) * 100, 1),
        "Profit_Factor": pf,
        "Mean_R": round(float(np.mean(r_list)), 2),
        "Total_Net_PnL_INR": round(float(np.sum(pnl_list)), 2),
    }


if __name__ == "__main__":
    smart_api = get_angel_session()
    scrip_df = load_token_map()
    results = []

    for sym in CANDIDATE_UNIVERSE:
        for exch in ["NSE", "BSE"]:
            if sym == "BSE" and exch == "BSE":
                continue
            token = resolve_token(scrip_df, sym, exch)
            if not token:
                continue
            candles = fetch_5m_candles(smart_api, exch, token, days_back=90)
            stats = simulate_stock(candles, sym, exch)
            if stats:
                results.append(stats)
                print(f"{sym}.{exch}: {stats['Trades_90d']} trades | Win: {stats['Win_Rate_Pct']}% | PnL: INR {stats['Total_Net_PnL_INR']}")

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df.sort_values(by="Total_Net_PnL_INR", ascending=False, inplace=True)
        res_df.to_csv(SCREENER_FILE, index=False)
        print(f"Saved {len(res_df)} ranked NSE & BSE stocks (90-day window) to {SCREENER_FILE}")
