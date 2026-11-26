import argparse
import json
import os
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

STATE_DIR = Path("state")
STATE_FILE = STATE_DIR / "mom_v3_state.json"
EQUITY_CSV = STATE_DIR / "mom_v3_equity.csv"
TRADES_CSV = STATE_DIR / "mom_v3_trades.csv"

SLOTS = 5
HOLD_RANK = 12
REBALANCE_DAYS = 21
TRANCHE_OFFSETS = {"T1": 0, "T2": 10}
TARGET_VOL = 0.18
SECTOR_CAP = 0.30
COST_PER_SIDE = 0.0022
CASH_DAILY_RATE = 0.06 / 252.0
CATASTROPHE_SL = 0.15

UNIVERSE_SECTORS: Dict[str, str] = {
    "HDFCBANK.NS": "FIN", "ICICIBANK.NS": "FIN", "SBIN.NS": "FIN", "AXISBANK.NS": "FIN",
    "KOTAKBANK.NS": "FIN", "BAJFINANCE.NS": "FIN", "BAJAJFINSV.NS": "FIN", "SHRIRAMFIN.NS": "FIN",
    "CHOLAFIN.NS": "FIN", "INDUSINDBK.NS": "FIN", "SBILIFE.NS": "FIN", "HDFCLIFE.NS": "FIN",
    "PFC.NS": "FIN", "RECLTD.NS": "FIN", "CDSL.NS": "FIN", "BANKBARODA.NS": "FIN",
    "PNB.NS": "FIN", "CANBK.NS": "FIN", "MUTHOOTFIN.NS": "FIN", "ICICIGI.NS": "FIN",
    "FEDERALBNK.NS": "FIN", "AUBANK.NS": "FIN",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
    "TECHM.NS": "IT", "PERSISTENT.NS": "IT", "COFORGE.NS": "IT", "TATAELXSI.NS": "IT",
    "RELIANCE.NS": "ENERGY", "ONGC.NS": "ENERGY", "NTPC.NS": "ENERGY", "POWERGRID.NS": "ENERGY",
    "COALINDIA.NS": "ENERGY", "BPCL.NS": "ENERGY", "TATAPOWER.NS": "ENERGY",
    "IOC.NS": "ENERGY", "GAIL.NS": "ENERGY",
    "M&M.NS": "AUTO", "MARUTI.NS": "AUTO", "BAJAJ-AUTO.NS": "AUTO", "EICHERMOT.NS": "AUTO",
    "HEROMOTOCO.NS": "AUTO", "TVSMOTOR.NS": "AUTO",
    "TATASTEEL.NS": "METALS", "JSWSTEEL.NS": "METALS", "HINDALCO.NS": "METALS",
    "JINDALSTEL.NS": "METALS", "VEDL.NS": "METALS", "ULTRACEMCO.NS": "METALS",
    "GRASIM.NS": "METALS", "PIDILITIND.NS": "METALS", "PIIND.NS": "METALS",
    "SUNPHARMA.NS": "PHARMA", "DRREDDY.NS": "PHARMA", "CIPLA.NS": "PHARMA",
    "DIVISLAB.NS": "PHARMA", "APOLLOHOSP.NS": "PHARMA", "LUPIN.NS": "PHARMA",
    "TORNTPHARM.NS": "PHARMA", "AUROPHARMA.NS": "PHARMA",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "TITAN.NS": "FMCG", "TRENT.NS": "FMCG", "TATACONSUM.NS": "FMCG", "COLPAL.NS": "FMCG",
    "HAVELLS.NS": "FMCG", "DIXON.NS": "FMCG",
    "LT.NS": "INFRA", "BHARTIARTL.NS": "INFRA", "BEL.NS": "INFRA", "HAL.NS": "INFRA",
    "SIEMENS.NS": "INFRA", "ABB.NS": "INFRA", "CUMMINSIND.NS": "INFRA", "INDIGO.NS": "INFRA",
    "POLYCAB.NS": "INFRA", "BHEL.NS": "INFRA", "CGPOWER.NS": "INFRA",
    "DLF.NS": "REALTY", "GODREJPROP.NS": "REALTY", "OBEROIRLTY.NS": "REALTY",
}


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set]")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=15) as resp:
            if resp.status != 200:
                print(f"[Telegram warning: HTTP {resp.status}]")
    except Exception as exc:
        print(f"[Telegram error: {exc}]")


def load_or_init_state(capital: float, reset: bool) -> dict:
    STATE_DIR.mkdir(exist_ok=True)
    if STATE_FILE.exists() and not reset:
        return json.loads(STATE_FILE.read_text())
    half = capital * 0.5
    state = {
        "initial_capital": capital,
        "trading_day_count": 0,
        "last_processed_date": None,
        "tranches": {
            name: {"cash": half, "positions": {}, "prev_equity": half, "daily_rets": [], "total_costs": 0.0}
            for name in TRANCHE_OFFSETS
        },
    }
    if reset:
        for f in [EQUITY_CSV, TRADES_CSV]:
            if f.exists():
                f.unlink()
    return state


def download_data() -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    tickers = list(UNIVERSE_SECTORS.keys())
    raw = yf.download(tickers, period="2y", interval="1d", auto_adjust=True, progress=False, threads=False)
    fields = {}
    for col in ["Open", "Low", "Close"]:
        df = raw[col].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[[col]].copy()
        fields[col] = df.dropna(axis=1, thresh=260).ffill()
    common = sorted(set(fields["Close"].columns) & set(fields["Open"].columns) & set(fields["Low"].columns))
    for col in fields:
        fields[col] = fields[col][common]
    bench = yf.download("^NSEI", period="2y", interval="1d", auto_adjust=True, progress=False, threads=False)
    bench_close = bench["Close"].squeeze().dropna().reindex(fields["Close"].index).ffill()
    return fields, bench_close


def compute_signals(fields: Dict[str, pd.DataFrame], bench_close: pd.Series) -> dict:
    close = fields["Close"]
    rets = close.pct_change()
    vol_20 = rets.rolling(20).std() * np.sqrt(252)
    vol_63 = rets.rolling(63).std() * np.sqrt(252)
    vol_126 = rets.rolling(126).std() * np.sqrt(252)
    vol_252 = rets.rolling(252).std() * np.sqrt(252)

    raw_mom = (
        0.35 * ((close.shift(10) / close.shift(63) - 1.0) / vol_63.replace(0, np.nan))
        + 0.40 * ((close.shift(15) / close.shift(126) - 1.0) / vol_126.replace(0, np.nan))
        + 0.25 * ((close.shift(21) / close.shift(252) - 1.0) / vol_252.replace(0, np.nan))
    )

    # R^2 of log-price over last 90 bars (computed only for last 2 dates for speed)
    log_p = np.log(close.replace(0, np.nan)).to_numpy()
    x = np.arange(90, dtype=float)
    xd = x - x.mean()
    xss = np.sum(xd ** 2)
    r2_rows = {}
    for idx in [-2, -1]:
        end = len(log_p) + idx + 1
        y = log_p[end - 90 : end]
        yd = y - np.nanmean(y, axis=0)
        cov = np.sum(xd[:, None] * yd, axis=0)
        yss = np.sum(yd ** 2, axis=0)
        denom = np.sqrt(xss * yss)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 1e-12, cov / denom, 0.0)
        r2_rows[idx] = pd.Series(np.maximum(corr, 0.0) ** 2, index=close.columns)

    sma_100 = close.rolling(100).mean()
    valid = sma_100.notna() & close.notna()
    breadth = (((close > sma_100) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    nifty_on = bench_close > bench_close.rolling(200).mean()

    return {
        "vol_20": vol_20,
        "sma_100": sma_100,
        "breadth": breadth,
        "nifty_on": nifty_on,
        "mom_score": {idx: raw_mom.iloc[idx] * np.sqrt(r2_rows[idx]) for idx in [-2, -1]},
    }


def get_regime_scale(nifty_on: bool, breadth: float) -> float:
    if not nifty_on or breadth < 0.35:
        return 0.0
    return 1.0 if breadth >= 0.50 else 0.50


def get_vol_scale(daily_rets: List[float]) -> float:
    if len(daily_rets) < 30:
        return 1.0
    v10 = float(np.std(daily_rets[-10:], ddof=1) * np.sqrt(252))
    v30 = float(np.std(daily_rets[-30:], ddof=1) * np.sqrt(252))
    rvol = max(v10, v30)
    return min(1.0, TARGET_VOL / rvol) if rvol > 1e-6 else 1.0


def plan_rebalance(tranche: dict, scores: pd.Series, prev_c: pd.Series, sma100: pd.Series, vol20: pd.Series, regime_scale: float) -> Tuple[List[str], List[str], List[str]]:
    scores = scores.dropna()
    scores = scores[scores > 0].sort_values(ascending=False)
    ranks = {t: r + 1 for r, t in enumerate(scores.index)}

    exits = []
    for t in list(tranche["positions"].keys()):
        pc, psma = float(prev_c.get(t, np.nan)), float(sma100.get(t, np.nan))
        if t not in ranks or ranks[t] > HOLD_RANK or (pd.notna(psma) and pc < psma):
            exits.append(t)

    kept = [t for t in tranche["positions"] if t not in exits]
    target = list(kept)
    if regime_scale > 0.0:
        sec_counts: Dict[str, int] = {}
        for t in target:
            sec = UNIVERSE_SECTORS.get(t, "OTHER")
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
        max_sec = max(1, int(np.floor(SLOTS * SECTOR_CAP)))
        for t in scores.index:
            if len(target) >= SLOTS:
                break
            if t in target:
                continue
            pc, psma = float(prev_c.get(t, np.nan)), float(sma100.get(t, np.nan))
            if pd.notna(psma) and pc < psma:
                continue
            sec = UNIVERSE_SECTORS.get(t, "OTHER")
            if sec_counts.get(sec, 0) >= max_sec:
                continue
            v20 = vol20.get(t, np.nan)
            if pd.isna(v20) or v20 <= 0.05:
                continue
            target.append(t)
            sec_counts[sec] = sec_counts.get(sec, 0) + 1

    entries = [t for t in target if t not in kept]
    return exits, entries, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=500000.0)
    parser.add_argument("--reset", type=str, default="false")
    args = parser.parse_args()

    state = load_or_init_state(args.capital, args.reset.lower() == "true")
    fields, bench_close = download_data()
    sig = compute_signals(fields, bench_close)

    latest_date = fields["Close"].index[-1].strftime("%Y-%m-%d")
    if state["last_processed_date"] == latest_date:
        print(f"Already processed {latest_date}. Exiting cleanly.")
        return

    day_num = state["trading_day_count"]
    open_r, low_r, close_r = fields["Open"].iloc[-1], fields["Low"].iloc[-1], fields["Close"].iloc[-1]
    prev_c_r = fields["Close"].iloc[-2]

    sig_nifty_on = bool(sig["nifty_on"].iloc[-2])
    sig_breadth = float(sig["breadth"].iloc[-2])
    sig_regime = get_regime_scale(sig_nifty_on, sig_breadth)

    eod_nifty_on = bool(sig["nifty_on"].iloc[-1])
    eod_breadth = float(sig["breadth"].iloc[-1])
    eod_regime = get_regime_scale(eod_nifty_on, eod_breadth)

    actions_log: List[str] = []
    closed_trades: List[dict] = []

    for name, offset in TRANCHE_OFFSETS.items():
        tr = state["tranches"][name]
        tr["cash"] *= (1.0 + CASH_DAILY_RATE)
        vol_scale = get_vol_scale(tr["daily_rets"])

        open_eq = tr["cash"] + sum(
            p["shares"] * float(open_r.get(t, p["entry_price"]))
            for t, p in tr["positions"].items()
        )

        # 1. Scheduled Rebalance (executed at today's Open using yesterday's Close signals)
        if day_num % REBALANCE_DAYS == offset:
            exits, entries, target = plan_rebalance(
                tr, sig["mom_score"][-2], prev_c_r, sig["sma_100"].iloc[-2], sig["vol_20"].iloc[-2], sig_regime
            )
            for t in exits:
                pos = tr["positions"].pop(t)
                fill = float(open_r.get(t, prev_c_r[t]))
                proceeds = pos["shares"] * fill
                cost = proceeds * COST_PER_SIDE
                tr["cash"] += proceeds - cost
                tr["total_costs"] += cost
                pnl = (fill - pos["entry_price"]) * pos["shares"] - cost - (pos["entry_price"] * pos["shares"] * COST_PER_SIDE)
                closed_trades.append({
                    "date": latest_date, "tranche": name, "ticker": t, "shares": pos["shares"],
                    "entry": round(pos["entry_price"], 2), "exit": round(fill, 2), "pnl": round(pnl, 2), "reason": "REBAL_EXIT",
                })
                actions_log.append(f"[{name}] SELL {t.replace('.NS','')} x{pos['shares']} @ {fill:.2f} (PnL {pnl:+,.0f})")

            if entries:
                inv_vols = {t: 1.0 / float(sig["vol_20"].iloc[-2][t]) for t in target}
                tot_inv = sum(inv_vols.values())
                deploy_cap = open_eq * vol_scale * sig_regime
                for t in entries:
                    fill = float(open_r.get(t, np.nan))
                    if pd.isna(fill) or fill <= 0:
                        continue
                    alloc = min(deploy_cap * (inv_vols[t] / tot_inv), tr["cash"] / (1.0 + COST_PER_SIDE))
                    shares = int(alloc // fill)
                    if shares <= 0:
                        continue
                    notional = shares * fill
                    cost = notional * COST_PER_SIDE
                    tr["cash"] -= notional + cost
                    tr["total_costs"] += cost
                    stop_px = fill * (1.0 - CATASTROPHE_SL)
                    tr["positions"][t] = {
                        "shares": shares, "entry_price": fill, "entry_date": latest_date, "stop": stop_px
                    }
                    actions_log.append(f"[{name}] BUY {t.replace('.NS','')} x{shares} @ {fill:.2f} (SL {stop_px:.2f})")

        # 2. Daily Catastrophe Stop Check (-15%)
        for t in list(tr["positions"].keys()):
            pos = tr["positions"][t]
            d_low, d_open = float(low_r.get(t, np.nan)), float(open_r.get(t, np.nan))
            if pd.notna(d_low) and d_low <= pos["stop"]:
                fill = min(d_open, pos["stop"]) if pd.notna(d_open) else pos["stop"]
                tr["positions"].pop(t)
                proceeds = pos["shares"] * fill
                cost = proceeds * COST_PER_SIDE
                tr["cash"] += proceeds - cost
                tr["total_costs"] += cost
                pnl = (fill - pos["entry_price"]) * pos["shares"] - cost - (pos["entry_price"] * pos["shares"] * COST_PER_SIDE)
                closed_trades.append({
                    "date": latest_date, "tranche": name, "ticker": t, "shares": pos["shares"],
                    "entry": round(pos["entry_price"], 2), "exit": round(fill, 2), "pnl": round(pnl, 2), "reason": "CATASTROPHE_SL",
                })
                actions_log.append(f"[{name}] STOP {t.replace('.NS','')} x{pos['shares']} @ {fill:.2f} (PnL {pnl:+,.0f})")

        # 3. EOD Mark-to-Market
        pos_val = sum(p["shares"] * float(close_r.get(t, p["entry_price"])) for t, p in tr["positions"].items())
        eod_eq = tr["cash"] + pos_val
        ret = (eod_eq / tr["prev_equity"]) - 1.0 if tr["prev_equity"] > 0 else 0.0
        tr["daily_rets"] = (tr["daily_rets"] + [ret])[-60:]
        tr["prev_equity"] = eod_eq

    # Next-day rebalance preview
    next_day = day_num + 1
    preview_lines: List[str] = []
    for name, offset in TRANCHE_OFFSETS.items():
        if next_day % REBALANCE_DAYS == offset:
            exits, entries, _ = plan_rebalance(
                state["tranches"][name], sig["mom_score"][-1], close_r, sig["sma_100"].iloc[-1], sig["vol_20"].iloc[-1], eod_regime
            )
            e_str = ", ".join(t.replace(".NS", "") for t in exits) or "None"
            b_str = ", ".join(t.replace(".NS", "") for t in entries) or "None"
            preview_lines.append(f"<b>{name} Rebalance Tomorrow:</b> SELL [{e_str}] | BUY [{b_str}]")

    state["trading_day_count"] = next_day
    state["last_processed_date"] = latest_date
    STATE_FILE.write_text(json.dumps(state, indent=2))

    tot_eq = sum(tr["prev_equity"] for tr in state["tranches"].values())
    tot_cash = sum(tr["cash"] for tr in state["tranches"].values())
    tot_exp = (tot_eq - tot_cash) / tot_eq * 100 if tot_eq > 0 else 0.0
    tot_ret = (tot_eq / state["initial_capital"] - 1.0) * 100

    eq_row = pd.DataFrame([{
        "date": latest_date, "day": day_num, "equity": round(tot_eq, 2), "cash": round(tot_cash, 2),
        "exposure_pct": round(tot_exp, 2), "breadth_pct": round(eod_breadth * 100, 1),
        "nifty_on": eod_nifty_on, "regime_scale": eod_regime,
    }])
    eq_row.to_csv(EQUITY_CSV, mode="a", header=not EQUITY_CSV.exists(), index=False)

    if closed_trades:
        pd.DataFrame(closed_trades).to_csv(TRADES_CSV, mode="a", header=not TRADES_CSV.exists(), index=False)

    # Build Telegram & Console Report
    lines = [
        f"<b>Mom v3 Paper Bot | {latest_date} (Day {day_num})</b>",
        f"Equity: <b>INR {tot_eq:,.0f}</b> ({tot_ret:+.2f}%) | Exp: {tot_exp:.1f}%",
        f"Regime: Nifty&gt;200DMA={'ON' if eod_nifty_on else 'OFF'} | Breadth: {eod_breadth*100:.1f}% | Scale: {eod_regime*100:.0f}%",
    ]
    if actions_log:
        lines.append("\n<b>Executed Today:</b>\n" + "\n".join(actions_log))
    if preview_lines:
        lines.append("\n<b>Next Session Alert:</b>\n" + "\n".join(preview_lines))

    lines.append("\n<b>Active Positions:</b>")
    for name in TRANCHE_OFFSETS:
        pos_map = state["tranches"][name]["positions"]
        if not pos_map:
            lines.append(f"  {name}: 100% Liquid Cash")
            continue
        p_strs = []
        for t, p in pos_map.items():
            ltp = float(close_r.get(t, p["entry_price"]))
            chg = (ltp / p["entry_price"] - 1.0) * 100
            p_strs.append(f"{t.replace('.NS','')} ({chg:+.1f}%)")
        lines.append(f"  {name}: " + ", ".join(p_strs))

    report = "\n".join(lines)
    print(report.replace("<b>", "").replace("</b>", "").replace("&gt;", ">"))
    send_telegram(report)


if __name__ == "__main__":
    main()
