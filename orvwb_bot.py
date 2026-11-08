import os
import time
import urllib.request
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyotp
from SmartApi import SmartConnect

LEDGER_FILE = "ORVWB_v6_Elite_Ledger.csv"

# Exact Angel One tokens for BSE (Paytm & MCX) + NSE lookup for the rest
WATCHLIST = [
    {"symbol": "PAYTM", "exchange": "BSE", "token": "543396"},
    {"symbol": "PAYTM", "exchange": "NSE", "token": None},
    {"symbol": "MCX", "exchange": "BSE", "token": "534091"},
    {"symbol": "MCX", "exchange": "NSE", "token": None},
    {"symbol": "BSE", "exchange": "NSE", "token": None},
    {"symbol": "HDFCAMC", "exchange": "NSE", "token": None},
    {"symbol": "LT", "exchange": "NSE", "token": None},
    {"symbol": "ADANIENT", "exchange": "NSE", "token": None},
    {"symbol": "RELIANCE", "exchange": "NSE", "token": None},
]


PSU_GRINDERS = ["SBIN", "BANKBARODA", "PNB", "CANBK", "NTPC", "POWERGRID", "COALINDIA"]
CAPITAL_PER_TRADE = 100000.0
COMMISSION_PCT = 0.0003


def get_angel_session():
    api_key = os.environ["ANGEL_API_KEY"]
    client_code = os.environ["ANGEL_CLIENT_CODE"]
    pin = os.environ["ANGEL_PIN"]
    totp_secret = os.environ["ANGEL_TOTP_SECRET"]

    smart_api = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    data = smart_api.generateSession(client_code, pin, totp)
    if not data.get("status"):
        raise RuntimeError(f"Angel One Login Failed: {data}")
    return smart_api


def load_token_map():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    with urllib.request.urlopen(url) as response:
        scrip_list = json.loads(response.read().decode())
    return pd.DataFrame(scrip_list)


def resolve_token(scrip_df: pd.DataFrame, item: dict) -> str:
    if item.get("token"):
        return str(item["token"])
    sym = item["symbol"]
    sub = scrip_df[(scrip_df["exch_seg"] == "NSE") & (scrip_df["symbol"] == f"{sym}-EQ")]
    if sub.empty:
        raise ValueError(f"Token not found for {sym} on NSE")
    return str(sub.iloc[0]["token"])


def fetch_5m_candles(smart_api, exchange: str, token: str, days_back: int = 30) -> pd.DataFrame:
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    params = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": "FIVE_MINUTE",
        "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
        "todate": to_date.strftime("%Y-%m-%d 15:30"),
    }
    res = smart_api.getCandleData(params)
    time.sleep(0.5)
    if not res.get("status") or not res.get("data"):
        print(f"Warning: No candle data returned for token {token} on {exchange}: {res}")
        return pd.DataFrame()

    df = pd.DataFrame(res["data"], columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
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

        pos = 0
        entry_price = active_stop = tp1_price = tp2_price = init_risk = 0.0
        tp1_hit = False
        entry_time = None
        realized_r = 0.0
        rem_weight = 1.0
        exit_reason = ""
        exit_price = 0.0

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

        if entry_time is not None:
            qty = int(CAPITAL_PER_TRADE // entry_price)
            gross_pnl = realized_r * init_risk * qty
            fees = CAPITAL_PER_TRADE * COMMISSION_PCT * 2
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
                "Net_PnL_INR": round(gross_pnl - fees, 2),
                "Status": exit_reason,
            })
    return trades


if __name__ == "__main__":
    smart_api = get_angel_session()
    scrip_df = load_token_map()
    all_new_trades = []

    for item in WATCHLIST:
        sym, exch = item["symbol"], item["exchange"]
        token = resolve_token(scrip_df, item)
        print(f"Fetching {sym} on {exch} (Token: {token})...")
        candles = fetch_5m_candles(smart_api, exch, token, days_back=30)
        sim_trades = simulate_orvwb_v6(candles, sym, exch)
        print(f" -> Found {len(sim_trades)} trades for {sym}.{exch}")
        all_new_trades.extend(sim_trades)

    new_df = pd.DataFrame(all_new_trades)
    if not new_df.empty:
        new_df.sort_values(by=["Date", "Entry_Time"], inplace=True)
        new_df.to_csv(LEDGER_FILE, index=False)
        print(f"Saved fresh Angel One ledger! Total trades: {len(new_df)}. Net PnL: INR {new_df['Net_PnL_INR'].sum():,.2f}")
    else:
        print("No trades generated across watchlist.")
