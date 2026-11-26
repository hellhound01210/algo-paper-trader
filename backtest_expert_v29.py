import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ==========================================
# CONFIGURATION & DEFAULT UNIVERSES
# ==========================================
CORE_6_STOCKS = ["MCX", "BSE", "RELIANCE", "BHARTIARTL", "SBIN", "BAJFINANCE"]
OOS_EXTENDED = [
    "SHRIRAMFIN",
    "POLICYBZR",
    "JINDALSTEL",
    "CDSL",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "TCS",
    "ITC",
    "TRENT",
    "TATASTEEL",
    "AXISBANK",
]

IST_TZ = "Asia/Kolkata"
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"



# ==========================================
# PINE SCRIPT v6 INDICATOR PARITY (WILDER RMA)
# ==========================================
def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    if len(vals) < length:
        return pd.Series(out, index=series.index)
    first_valid = np.where(~np.isnan(vals))[0]
    if len(first_valid) == 0:
        return pd.Series(out, index=series.index)
    start = first_valid[0]
    if start + length > len(vals):
        return pd.Series(out, index=series.index)
    out[start + length - 1] = np.nanmean(vals[start : start + length])
    alpha = 1.0 / length
    for i in range(start + length, len(vals)):
        if np.isnan(vals[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # Daily IST-reset VWAP (hlc3)
    hlc3 = (high + low + close) / 3.0
    day_key = df.index.date
    cum_pv = (hlc3 * volume).groupby(day_key).cumsum()
    cum_vol = volume.groupby(day_key).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol

    # 50 EMA & 20-bar Average Volume
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["avg_vol20"] = volume.rolling(20).mean()

    # True Range & Wilder ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = wilder_rma(tr, 14)

    # Wilder RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, 14)
    avg_loss = wilder_rma(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi14"] = df["rsi14"].fillna(50.0)

    # Wilder DMI / ADX(14, 14)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )
    atr_safe = df["atr14"].replace(0, np.nan)
    df["di_plus"] = 100.0 * wilder_rma(plus_dm, 14) / atr_safe
    df["di_minus"] = 100.0 * wilder_rma(minus_dm, 14) / atr_safe
    dx = (
        100.0
        * (df["di_plus"] - df["di_minus"]).abs()
        / (df["di_plus"] + df["di_minus"]).replace(0, np.nan)
    )
    df["adx14"] = wilder_rma(dx.fillna(0.0), 14)

    # Rolling 5-bar high/low for breakouts (prior 5 bars)
    df["hh5_prev"] = high.rolling(5).max().shift(1)
    df["ll5_prev"] = low.rolling(5).min().shift(1)

    return df


# ==========================================
# DATA LOADERS (ANGEL ONE SMARTAPI & YFINANCE)
# ==========================================
class AngelDataClient:
    def __init__(self):
        from SmartApi import SmartConnect
        import pyotp

        api_key = os.getenv("ANGEL_HIST_API_KEY") or os.getenv("ANGEL_API_KEY")
        client_code = os.getenv("ANGEL_CLIENT_CODE") or os.getenv("ANGEL_CLIENT_ID")
        pin = os.getenv("ANGEL_PIN") or os.getenv("ANGEL_PASSWORD")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET")

        missing = [
            name
            for name, val in [
                ("ANGEL_HIST_API_KEY", api_key),
                ("ANGEL_CLIENT_CODE", client_code),
                ("ANGEL_PIN/ANGEL_PASSWORD", pin),
                ("ANGEL_TOTP_SECRET", totp_secret),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(f"Missing Angel One secrets: {', '.join(missing)}")

        self.smart = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        session = self.smart.generateSession(client_code, pin, totp)
        if not session or not session.get("status"):
            raise RuntimeError(f"Angel One login failed: {session}")

        print("[AngelAPI] Session authenticated. Downloading NSE scrip master...")
        resp = requests.get(SCRIP_MASTER_URL, timeout=30)
        resp.raise_for_status()
        master = resp.json()
        self.token_map = {
            item["symbol"]: item["token"]
            for item in master
            if item.get("exch_seg") == "NSE" and item.get("symbol", "").endswith("-EQ")
        }
        print(f"[AngelAPI] Loaded {len(self.token_map)} NSE-EQ tokens.")

    def fetch_15m(self, ticker: str, days: int) -> pd.DataFrame:
        symbol_eq = f"{ticker.upper()}-EQ"
        token = self.token_map.get(symbol_eq)
        if not token:
            print(f"[WARN] Token not found for {symbol_eq}")
            return pd.DataFrame()

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
        chunk_days = 120  # Safe chunk size under SmartAPI 15m 200-day cap
        all_rows = []

        cur_start = start_dt
        while cur_start < end_dt:
            cur_end = min(cur_start + timedelta(days=chunk_days), end_dt)
            params = {
                "exchange": "NSE",
                "symboltoken": token,
                "interval": "FIFTEEN_MINUTE",
                "fromdate": cur_start.strftime("%Y-%m-%d 09:15"),
                "todate": cur_end.strftime("%Y-%m-%d 15:30"),
            }

            success = False
            for attempt in range(4):
                try:
                    time.sleep(1.5)  # Conservative pacing under 120 req/min limit
                    res = self.smart.getCandleData(params)
                    if res and res.get("status") and res.get("data"):
                        all_rows.extend(res["data"])
                        success = True
                        break
                    msg = str(res)
                    if "rate" in msg.lower() or "access denied" in msg.lower():
                        wait = 5 * (2**attempt)
                        print(f"[AngelAPI] Rate limit on {ticker}, backing off {wait}s...")
                        time.sleep(wait)
                    else:
                        success = True  # Empty window (holidays/listing date)
                        break
                except Exception as exc:
                    wait = 5 * (2**attempt)
                    print(f"[AngelAPI] Chunk error on {ticker}: {exc} (retry in {wait}s)")
                    time.sleep(wait)

            if not success:
                print(f"[WARN] Skipped chunk {cur_start.date()} -> {cur_end.date()} for {ticker}")

            cur_start = cur_end + timedelta(days=1)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_rows, columns=["DateTime", "Open", "High", "Low", "Close", "Volume"]
        )
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        if df["DateTime"].dt.tz is None:
            df["DateTime"] = df["DateTime"].dt.tz_localize(IST_TZ)
        else:
            df["DateTime"] = df["DateTime"].dt.tz_convert(IST_TZ)
        df = df.drop_duplicates(subset=["DateTime"]).sort_values("DateTime")
        df = df.set_index("DateTime")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()


def fetch_yfinance_15m(ticker: str, days: int) -> pd.DataFrame:
    import yfinance as yf

    eff_days = min(days, 59)  # yfinance caps 15m intraday history at 60 days
    yf_sym = f"{ticker.upper()}.NS"
    df = yf.download(
        yf_sym,
        period=f"{eff_days}d",
        interval="15m",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST_TZ)
    else:
        df.index = df.index.tz_convert(IST_TZ)
    return df.sort_index()


# ==========================================
# TRI-CORE ROUTER & v2.9 BACKTEST ENGINE
# ==========================================
def resolve_core(ticker: str, core_mode: str) -> str:
    if core_mode in ("c24", "c25", "c21"):
        return core_mode.upper()
    tkr = ticker.upper()
    if any(k in tkr for k in ("RELIANCE", "HDFC", "INFY", "TCS", "ITC")):
        return "C-2.4"
    if any(k in tkr for k in ("BAJFIN", "BAJAJ", "CHOLA", "JIOFIN")):
        return "C-2.1"
    return "C-2.5"


def get_core_params(core: str) -> dict:
    use24 = core == "C-2.4"
    use21 = core == "C-2.1"
    return {
        "core": core,
        "use24": use24,
        "use21": use21,
        "use25": not use24 and not use21,
        "max_trades_day": 2 if use24 else 3,
        "cooldown_bars": 0 if use21 else 3,
        "adx_thresh": 18,
        "vol_mult": 1.05,
        "vol_pull_mult": 0.85 if use24 else 0.90,
        "max_vwap_dist": 2.30 if use24 else (2.50 if use21 else 2.40),
        "risk_per_trade": 0.90 if use24 else (1.00 if use21 else 0.95),
        "max_leverage": 1.00 if use24 else (2.50 if use21 else 1.35),
        "sl_mult": 1.50,
        "sizing_floor": 0.55 if use24 else (0.00 if use21 else 0.50),
        "tp1_mult": 1.00 if use24 else (1.10 if use21 else 1.05),
        "tp2_mult": 2.30 if use24 else 2.60,
        "tp1_qty_pct": 0.60 if use24 else 0.50,
        "be_lock_mult": 0.25 if use24 else (0.15 if use21 else 0.20),
        "min_shares": 3 if use24 else 2,
        "solid_body_min": 0.35 if use21 else 0.38,
    }


def in_session(ts: pd.Timestamp, start_hhmm: int, end_hhmm: int) -> bool:
    hhmm = ts.hour * 100 + ts.minute
    return start_hhmm <= hhmm <= end_hhmm


def run_backtest_single(
    ticker: str,
    df: pd.DataFrame,
    core_mode: str = "auto",
    initial_capital: float = 100000.0,
    commission_pct: float = 0.0003,
    slippage_tick: float = 0.05,
) -> tuple[dict, list[dict]]:
    core = resolve_core(ticker, core_mode)
    p = get_core_params(core)
    df = compute_indicators(df)

    equity = initial_capital
    peak_equity = initial_capital
    max_dd_pct = 0.0

    # Active position state
    pos_dir = 0  # +1 long, -1 short, 0 flat
    pos_qty = 0
    init_qty = 0
    entry_fill = 0.0
    entry_ref = 0.0
    one_r = 0.0
    active_sl = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    tp1_filled = False
    entry_time = None

    # Pending next-bar market order (mirrors TradingView fill on next bar open)
    pending_order = None

    trades_today = 0
    last_exit_bar = -999
    prev_day = None
    executions = []

    n = len(df)
    idx = df.index
    O = df["Open"].to_numpy()
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    V = df["Volume"].to_numpy()
    VWAP = df["vwap"].to_numpy()
    EMA = df["ema50"].to_numpy()
    ATR = df["atr14"].to_numpy()
    RSI = df["rsi14"].to_numpy()
    AVGVOL = df["avg_vol20"].to_numpy()
    DIP = df["di_plus"].to_numpy()
    DIM = df["di_minus"].to_numpy()
    ADX = df["adx14"].to_numpy()
    HH5 = df["hh5_prev"].to_numpy()
    LL5 = df["ll5_prev"].to_numpy()

    def close_leg(qty_close: int, raw_exit_price: float, reason: str, ts: pd.Timestamp):
        nonlocal equity, peak_equity, max_dd_pct, pos_qty, pos_dir, last_exit_bar
        fill_exit = (
            raw_exit_price - slippage_tick
            if pos_dir == 1
            else raw_exit_price + slippage_tick
        )
        gross_pnl = (fill_exit - entry_fill) * qty_close * pos_dir
        comm = (entry_fill * qty_close + fill_exit * qty_close) * commission_pct
        net_pnl = gross_pnl - comm
        equity += net_pnl
        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        executions.append(
            {
                "ticker": ticker,
                "core": core,
                "entry_time": entry_time,
                "exit_time": ts,
                "side": "LONG" if pos_dir == 1 else "SHORT",
                "qty": qty_close,
                "entry_price": round(entry_fill, 2),
                "exit_price": round(fill_exit, 2),
                "net_pnl": round(net_pnl, 2),
                "reason": reason,
                "year": ts.year,
            }
        )
        pos_qty -= qty_close
        if pos_qty <= 0:
            pos_qty = 0
            pos_dir = 0
            last_exit_bar = i

    for i in range(55, n):
        ts = idx[i]
        cur_day = ts.date()
        if prev_day is None or cur_day != prev_day:
            trades_today = 0
            prev_day = cur_day

        # 1. Execute pending entry on current bar's OPEN
        if pending_order is not None and pos_dir == 0:
            side = pending_order["side"]
            pos_dir = side
            pos_qty = pending_order["qty"]
            init_qty = pos_qty
            entry_fill = O[i] + slippage_tick if side == 1 else O[i] - slippage_tick
            entry_ref = pending_order["entry_ref"]
            one_r = pending_order["one_r"]
            active_sl = pending_order["sl"]
            tp1_price = pending_order["tp1"]
            tp2_price = pending_order["tp2"]
            tp1_filled = False
            entry_time = ts
            if not p["use21"]:
                trades_today += 1
            pending_order = None

        # 2. Manage open position on current bar's OHLC
        if pos_dir != 0:
            in_sqoff = in_session(ts, 1515, 1530)

            if pos_dir == 1:
                # Pessimistic ordering: check SL before TP if both touched on same bar
                if L[i] <= active_sl:
                    close_leg(pos_qty, min(O[i], active_sl), "SL", ts)
                else:
                    if not tp1_filled and H[i] >= tp1_price:
                        tp1_qty = max(1, int(round(init_qty * p["tp1_qty_pct"])))
                        tp1_qty = min(tp1_qty, pos_qty)
                        close_leg(tp1_qty, max(O[i], tp1_price), "TP1", ts)
                        tp1_filled = True
                        active_sl = entry_ref + (one_r * p["be_lock_mult"])
                    if pos_dir == 1 and H[i] >= tp2_price:
                        close_leg(pos_qty, max(O[i], tp2_price), "TP2", ts)
                    elif pos_dir == 1 and in_sqoff:
                        close_leg(pos_qty, C[i], "15:15_SQOFF", ts)

            elif pos_dir == -1:
                if H[i] >= active_sl:
                    close_leg(pos_qty, max(O[i], active_sl), "SL", ts)
                else:
                    if not tp1_filled and L[i] <= tp1_price:
                        tp1_qty = max(1, int(round(init_qty * p["tp1_qty_pct"])))
                        tp1_qty = min(tp1_qty, pos_qty)
                        close_leg(tp1_qty, min(O[i], tp1_price), "TP1", ts)
                        tp1_filled = True
                        active_sl = entry_ref - (one_r * p["be_lock_mult"])
                    if pos_dir == -1 and L[i] <= tp2_price:
                        close_leg(pos_qty, min(O[i], tp2_price), "TP2", ts)
                    elif pos_dir == -1 and in_sqoff:
                        close_leg(pos_qty, C[i], "15:15_SQOFF", ts)

        # 3. Evaluate new entry signal on bar close (only if flat and next bar exists)
        if pos_dir != 0 or i >= n - 1:
            continue
        if np.isnan(ATR[i]) or ATR[i] <= 0 or np.isnan(VWAP[i]) or np.isnan(AVGVOL[i]):
            continue

        # Session window check
        if p["use24"]:
            in_win = in_session(ts, 930, 1200) or in_session(ts, 1300, 1430)
        elif p["use21"]:
            in_win = in_session(ts, 930, 1445)
        else:
            in_win = in_session(ts, 930, 1430)

        cooldown_ok = True if p["use21"] else ((i - last_exit_bar) >= p["cooldown_bars"])
        cb_ok = (trades_today < p["max_trades_day"]) and cooldown_ok
        if not (in_win and cb_ok):
            continue

        # Regime & slope filters
        if p["use25"]:
            regime_trending = (ADX[i] >= p["adx_thresh"]) or (
                ADX[i] >= 15 and ADX[i] > ADX[i - 1] and ADX[i - 1] > ADX[i - 2]
            )
            ema_up = (EMA[i] > EMA[i - 3]) and ((C[i] - EMA[i]) >= ATR[i] * 0.18)
            ema_dn = (EMA[i] < EMA[i - 3]) and ((EMA[i] - C[i]) >= ATR[i] * 0.18)
        else:
            regime_trending = (ADX[i] >= p["adx_thresh"]) or (
                ADX[i] >= 15 and ADX[i] > ADX[i - 1]
            )
            ema_up = (EMA[i] >= EMA[i - 2]) and (C[i] > EMA[i] if p["use24"] else True)
            ema_dn = (EMA[i] <= EMA[i - 2]) and (C[i] < EMA[i] if p["use24"] else True)

        if not regime_trending:
            continue

        candle_range = H[i] - L[i]
        solid_body = (
            (abs(C[i] - O[i]) / candle_range) >= p["solid_body_min"]
            if candle_range > 0
            else False
        )
        not_ext = abs(C[i] - VWAP[i]) <= (ATR[i] * p["max_vwap_dist"])
        if not (solid_body and not_ext):
            continue

        vol_break_ok = V[i] >= AVGVOL[i] * p["vol_mult"]
        vol_pull_ok = V[i] >= AVGVOL[i] * p["vol_pull_mult"]

        # Bullish signal
        bull_regime = (C[i] > VWAP[i]) and (C[i] > EMA[i]) and ema_up and (DIP[i] > DIM[i])
        bull_bo = (C[i - 1] <= HH5[i - 1]) and (C[i] > HH5[i]) and vol_break_ok
        if p["use21"]:
            bull_pb = (
                (L[i] <= VWAP[i] * 1.002 or L[i] <= EMA[i] * 1.002)
                and C[i] > VWAP[i]
                and C[i] > O[i]
            )
            rsi_bull = 51.0 <= RSI[i] <= 72.0
        else:
            bull_pb = (
                (
                    L[i] <= VWAP[i] * 1.0018
                    or L[i - 1] <= VWAP[i - 1] * 1.0018
                    or L[i] <= EMA[i] * 1.0018
                )
                and (C[i - 1] < O[i - 1] or L[i] < L[i - 1])
                and C[i] > H[i - 1]
                and C[i] > O[i]
                and vol_pull_ok
            )
            rsi_bull = 52.0 <= RSI[i] <= 71.0
        long_cond = bull_regime and (bull_bo or bull_pb) and rsi_bull

        # Bearish signal
        bear_regime = (C[i] < VWAP[i]) and (C[i] < EMA[i]) and ema_dn and (DIM[i] > DIP[i])
        bear_bd = (C[i - 1] >= LL5[i - 1]) and (C[i] < LL5[i]) and vol_break_ok
        if p["use21"]:
            bear_pb = (
                (H[i] >= VWAP[i] * 0.998 or H[i] >= EMA[i] * 0.998)
                and C[i] < VWAP[i]
                and C[i] < O[i]
            )
            rsi_bear = 28.0 <= RSI[i] <= 49.0
        else:
            bear_pb = (
                (
                    H[i] >= VWAP[i] * 0.9982
                    or H[i - 1] >= VWAP[i - 1] * 0.9982
                    or H[i] >= EMA[i] * 0.9982
                )
                and (C[i - 1] > O[i - 1] or H[i] > H[i - 1])
                and C[i] < L[i - 1]
                and C[i] < O[i]
                and vol_pull_ok
            )
            rsi_bear = 29.0 <= RSI[i] <= 48.0
        short_cond = bear_regime and (bear_bd or bear_pb) and rsi_bear

        if not (long_cond or short_cond):
            continue

        # Position sizing & Core v2.1 cash gate
        true_stop = ATR[i] * p["sl_mult"]
        sizing_dist = (
            true_stop
            if p["use21"]
            else max(true_stop, C[i] * (p["sizing_floor"] / 100.0))
        )
        risk_cash = equity * (p["risk_per_trade"] / 100.0)
        raw_qty = risk_cash / sizing_dist if sizing_dist > 0 else 0
        max_qty = (equity * p["max_leverage"]) / C[i]
        qty = max(p["min_shares"], int(np.floor(min(raw_qty, max_qty))))

        if p["use21"]:
            trades_today += 1
            if (qty * C[i]) > equity:
                continue  # Exact v2.1 unleveraged cash gate

        side = 1 if long_cond else -1
        pending_order = {
            "side": side,
            "qty": qty,
            "entry_ref": C[i],
            "one_r": true_stop,
            "sl": C[i] - true_stop if side == 1 else C[i] + true_stop,
            "tp1": C[i] + true_stop * p["tp1_mult"]
            if side == 1
            else C[i] - true_stop * p["tp1_mult"],
            "tp2": C[i] + true_stop * p["tp2_mult"]
            if side == 1
            else C[i] - true_stop * p["tp2_mult"],
        }

    # Compile summary stats
    tot_exec = len(executions)
    pnls = [e["net_pnl"] for e in executions]
    wins = sum(1 for x in pnls if x > 0)
    gross_profit = sum(x for x in pnls if x > 0)
    gross_loss = abs(sum(x for x in pnls if x < 0))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    net_profit = equity - initial_capital
    net_pct = (net_profit / initial_capital) * 100.0
    win_rate = (wins / tot_exec) * 100.0 if tot_exec > 0 else 0.0

    summary = {
        "Ticker": ticker,
        "Group": "IN-SAMPLE (Core 6)" if ticker in CORE_6_STOCKS else "OUT-OF-SAMPLE",
        "Core": core,
        "Bars": len(df),
        "From": str(df.index[0].date()),
        "To": str(df.index[-1].date()),
        "Executions": tot_exec,
        "WinRate%": round(win_rate, 2),
        "PF": round(pf, 2),
        "NetPnL": round(net_profit, 2),
        "Return%": round(net_pct, 2),
        "MaxDD%": round(max_dd_pct, 2),
    }
    return summary, executions


# ==========================================
# CLI ENTRYPOINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Backtest Expert Confluence v2.9 across NSE stocks")
    parser.add_argument("--source", choices=["angel", "yfinance"], default="angel")
    parser.add_argument("--days", type=int, default=365, help="Lookback calendar days (yfinance caps at 59d)")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(CORE_6_STOCKS + OOS_EXTENDED[:6]),
        help="Comma-separated NSE tickers (without -EQ or .NS)",
    )
    parser.add_argument("--core", choices=["auto", "c25", "c24", "c21"], default="auto")
    parser.add_argument("--capital", type=float, default=100000.0)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"=== Expert Confluence v2.9 Backtest | Source: {args.source} | Days: {args.days} | Core: {args.core} ===")

    angel_client = AngelDataClient() if args.source == "angel" else None
    summaries = []
    all_trades = []

    for tkr in tickers:
        print(f"Fetching 15m data for {tkr}...")
        df = (
            angel_client.fetch_15m(tkr, args.days)
            if args.source == "angel"
            else fetch_yfinance_15m(tkr, args.days)
        )
        if df.empty or len(df) < 100:
            print(f"[SKIP] Insufficient data for {tkr} ({len(df)} bars)")
            continue
        summary, trades = run_backtest_single(
            tkr, df, core_mode=args.core, initial_capital=args.capital
        )
        summaries.append(summary)
        all_trades.extend(trades)

    if not summaries:
        raise SystemExit("No stocks produced valid backtest results.")

    res_dir = Path("results")
    res_dir.mkdir(exist_ok=True)
    sum_df = pd.DataFrame(summaries).sort_values("Return%", ascending=False)
    trades_df = pd.DataFrame(all_trades)

    sum_df.to_csv(res_dir / "expert_v29_summary.csv", index=False)
    if not trades_df.empty:
        trades_df.to_csv(res_dir / "expert_v29_trades.csv", index=False)

    print("\n================ PER-STOCK PERFORMANCE ================")
    print(sum_df.to_string(index=False))

    print("\n================ IN-SAMPLE vs OUT-OF-SAMPLE ================")
    grp = sum_df.groupby("Group").agg(
        Stocks=("Ticker", "count"),
        Winners=("NetPnL", lambda s: int((s > 0).sum())),
        TotalExec=("Executions", "sum"),
        AvgWR=("WinRate%", "mean"),
        AvgPF=("PF", "mean"),
        TotalPnL=("NetPnL", "sum"),
        AvgMaxDD=("MaxDD%", "mean"),
    )
    print(grp.round(2).to_string())

    if not trades_df.empty:
        print("\n================ YEARLY BREAKDOWN ================")
        yr = trades_df.groupby("year").agg(
            Executions=("net_pnl", "count"),
            WinRate=("net_pnl", lambda s: round((s > 0).mean() * 100.0, 2)),
            NetPnL=("net_pnl", "sum"),
        )
        print(yr.round(2).to_string())


if __name__ == "__main__":
    main()
