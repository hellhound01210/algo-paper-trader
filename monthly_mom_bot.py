import os
import time
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION (VALIDATED 5-YR STRESS WINNER: Mom_2Slot_12SL)
# ==============================================================================
IST = timezone(timedelta(hours=5, minutes=30))
FETCH_DAYS = 400            # ~270+ trading days (plenty for 200 DMA & ROC120)
YF_BATCH_SIZE = 45

START_CAPITAL = 50000.0
MAX_SLOTS = 2               # 2 concurrent CNC slots (~50% equity each, 1x cash)
HOLD_RANK_BUFFER = 6        # Keep existing holding if still inside Top 6
STOP_LOSS_PCT = 0.12        # -12% hard catastrophe stop
MIN_SHARE_PRICE = 100.0

SLIPPAGE_PER_SIDE = 0.0005  # 0.05% per side = 0.10% round-trip slippage
DP_CHARGE_PER_SELL = 23.60  # Angel One CDSL DP charge: ₹20 + 18% GST per sell

STATE_FILE = "Mom_2Slot_12SL_State.json"
LEDGER_FILE = "Mom_2Slot_12SL_Ledger.csv"
EQUITY_FILE = "Mom_2Slot_12SL_Equity.csv"


# ==============================================================================
# 2. EXACT 2026 ANGEL ONE EQUITY DELIVERY (CNC) CHARGES & SLIPPAGE
# ==============================================================================
def calc_cnc_net_pnl(entry_price: float, exit_price: float, qty: int) -> float:
    if qty < 1:
        return 0.0
    buy_val = entry_price * qty
    sell_val = exit_price * qty
    turnover = buy_val + sell_val
    gross = sell_val - buy_val

    b_buy = max(5.0, min(20.0, buy_val * 0.001))
    b_sell = max(5.0, min(20.0, sell_val * 0.001))
    brokerage = b_buy + b_sell

    stt = 0.001 * turnover
    exch_txn = 0.0000307 * turnover
    sebi = 0.000001 * turnover
    stamp = 0.00015 * buy_val
    gst = 0.18 * (brokerage + exch_txn + sebi)
    dp_fee = DP_CHARGE_PER_SELL
    slippage = turnover * SLIPPAGE_PER_SIDE

    total_cost = brokerage + stt + exch_txn + sebi + stamp + gst + dp_fee + slippage
    return round(gross - total_cost, 2)


# ==============================================================================
# 3. AUTO NSE F&O UNIVERSE & YFINANCE DAILY FETCHER
# ==============================================================================
def load_nse_fno_symbols() -> list:
    urls = [
        "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json",
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    ]
    scrip_df = None
    for url in urls:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    scrip_df = pd.DataFrame(json.loads(response.read().decode()))
                break
            except Exception as e:
                print(f"ScripMaster fetch failed ({url}, try {attempt + 1}): {e}")
                time.sleep(2)
        if scrip_df is not None:
            break
    if scrip_df is None:
        raise RuntimeError("Failed to download OpenAPIScripMaster.json")

    nfo_futs = scrip_df[
        (scrip_df["exch_seg"] == "NFO") & (scrip_df["instrumenttype"] == "FUTSTK")
    ]
    fno_names = sorted(set(nfo_futs["name"].dropna().astype(str).str.strip()))

    nse_eq = scrip_df[
        (scrip_df["exch_seg"] == "NSE") & (scrip_df["symbol"].astype(str).str.endswith("-EQ"))
    ]
    valid_eq = set(nse_eq["symbol"].astype(str))

    symbols = [s for s in fno_names if s != "IDEA" and f"{s}-EQ" in valid_eq]
    print(f"Auto-discovered {len(symbols)} active NSE F&O cash equities (Full A-Z).")
    return symbols


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["ROC60"] = (df["Close"] / df["Close"].shift(60) - 1.0) * 100.0
    df["ROC120"] = (df["Close"] / df["Close"].shift(120) - 1.0) * 100.0

    daily_ret = df["Close"].pct_change()
    df["Vol20_Ann"] = daily_ret.rolling(20).std() * np.sqrt(252) * 100.0
    df["Mom_Score"] = (0.5 * df["ROC60"] + 0.5 * df["ROC120"]) / df["Vol20_Ann"].clip(lower=10.0)
    return df.set_index("Date")


def fetch_nifty_daily() -> pd.DataFrame:
    start_dt = (datetime.now(IST) - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")
    raw = yf.download("^NSEI", start=start_dt, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Failed to fetch Nifty 50 (^NSEI) daily data.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()
    date_col = raw.columns[0]
    raw["Date"] = pd.to_datetime(raw[date_col]).dt.strftime("%Y-%m-%d")
    raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw = raw.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
    return add_daily_indicators(raw)


def fetch_batch_daily_yfinance(symbols: list) -> dict:
    start_dt = (datetime.now(IST) - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")
    stock_dfs = {}

    for i in range(0, len(symbols), YF_BATCH_SIZE):
        batch = symbols[i : i + YF_BATCH_SIZE]
        yf_tickers = [f"{s}.NS" for s in batch]
        try:
            raw = yf.download(
                yf_tickers,
                start=start_dt,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[Warn] Batch {i}-{i + len(batch)} download error: {e}")
            continue

        if raw is None or raw.empty:
            continue

        for sym, y_t in zip(batch, yf_tickers):
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if y_t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[y_t].copy()
                else:
                    sub = raw.copy()
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(sub) < 205:
                    continue
                sub = sub.reset_index()
                date_col = sub.columns[0]
                sub["Date"] = pd.to_datetime(sub[date_col]).dt.strftime("%Y-%m-%d")
                sub = sub[["Date", "Open", "High", "Low", "Close", "Volume"]]
                sub = sub.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
                stock_dfs[sym] = add_daily_indicators(sub)
            except Exception:
                continue

        print(f"  Batch {min(i + YF_BATCH_SIZE, len(symbols))}/{len(symbols)} done | Valid: {len(stock_dfs)}")
        time.sleep(0.4)

    return stock_dfs


# ==============================================================================
# 4. STATE, LEDGER & EQUITY PERSISTENCE
# ==============================================================================
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "start_capital": START_CAPITAL,
        "cash": START_CAPITAL,
        "open_positions": {},
        "last_run_date": "",
        "last_rebalance_month": "",
    }


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_closed_trades(new_trades: list) -> pd.DataFrame:
    if os.path.exists(LEDGER_FILE):
        ledger_df = pd.read_csv(LEDGER_FILE)
    else:
        ledger_df = pd.DataFrame(columns=[
            "Entry_Date", "Exit_Date", "Ticker", "Qty",
            "Entry_Price", "Exit_Price", "Gross_Ret_Pct",
            "Net_PnL_INR", "Days_Held", "Exit_Reason",
        ])

    if new_trades:
        add_df = pd.DataFrame(new_trades)
        ledger_df = pd.concat([ledger_df, add_df], ignore_index=True)
        ledger_df.to_csv(LEDGER_FILE, index=False)
    elif not os.path.exists(LEDGER_FILE):
        ledger_df.to_csv(LEDGER_FILE, index=False)

    return ledger_df


def update_equity_log(row_dict: dict):
    if os.path.exists(EQUITY_FILE):
        eq_df = pd.read_csv(EQUITY_FILE)
        eq_df = eq_df[eq_df["Date"].astype(str) != str(row_dict["Date"])].copy()
        eq_df = pd.concat([eq_df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        eq_df = pd.DataFrame([row_dict])
    eq_df = eq_df.sort_values("Date").reset_index(drop=True)
    eq_df.to_csv(EQUITY_FILE, index=False)


# ==============================================================================
# 5. RANKING & PORTFOLIO EXECUTION ENGINE
# ==============================================================================
def compute_rankings(stock_dfs: dict, as_of_date: str) -> list:
    ranked = []
    for sym, df in stock_dfs.items():
        if as_of_date not in df.index:
            continue
        r = df.loc[as_of_date]
        if any(pd.isna(r[k]) for k in ("SMA50", "SMA200", "ROC60", "ROC120", "Mom_Score", "Vol20_Ann")):
            continue
        c = float(r["Close"])
        if (
            c >= MIN_SHARE_PRICE
            and c > float(r["SMA200"])
            and c > float(r["SMA50"])
            and float(r["ROC60"]) > 0.0
            and float(r["ROC120"]) > 0.0
        ):
            ranked.append({
                "symbol": sym,
                "close": round(c, 2),
                "mom_score": round(float(r["Mom_Score"]), 3),
                "roc60": round(float(r["ROC60"]), 1),
                "roc120": round(float(r["ROC120"]), 1),
                "vol20": round(float(r["Vol20_Ann"]), 1),
            })
    ranked.sort(key=lambda x: x["mom_score"], reverse=True)
    return ranked


def check_daily_stops(state: dict, stock_dfs: dict, nifty_dates: list, latest_date: str) -> tuple:
    """
    Checks every trading day strictly after last_run_date up to latest_date
    so -12% hard stops are never missed even if the bot isn't run every single day.
    """
    last_run = state.get("last_run_date", "")
    if not last_run:
        dates_to_check = [latest_date]
    else:
        dates_to_check = [d for d in nifty_dates if last_run < d <= latest_date]

    closed_trades = []
    actions = []

    for d in dates_to_check:
        to_close = []
        for sym, pos in state["open_positions"].items():
            if d <= pos["entry_date"]:
                continue
            if sym not in stock_dfs or d not in stock_dfs[sym].index:
                continue
            row = stock_dfs[sym].loc[d]
            o, l = float(row["Open"]), float(row["Low"])
            pos["days_held"] = int(pos.get("days_held", 0)) + 1
            if l <= float(pos["stop_loss"]):
                fill_px = round(min(o, float(pos["stop_loss"])), 2)
                to_close.append((sym, fill_px, d, f"Hard Stop (-{int(STOP_LOSS_PCT * 100)}%)"))

        for sym, exit_p, exit_d, reason in to_close:
            pos = state["open_positions"].pop(sym)
            net_pnl = calc_cnc_net_pnl(pos["entry_price"], exit_p, pos["qty"])
            state["cash"] = round(state["cash"] + (pos["qty"] * pos["entry_price"]) + net_pnl, 2)
            gross_ret = round((exit_p / pos["entry_price"] - 1.0) * 100.0, 2)
            closed_trades.append({
                "Entry_Date": pos["entry_date"],
                "Exit_Date": exit_d,
                "Ticker": f"{sym}.NSE",
                "Qty": pos["qty"],
                "Entry_Price": round(pos["entry_price"], 2),
                "Exit_Price": exit_p,
                "Gross_Ret_Pct": gross_ret,
                "Net_PnL_INR": net_pnl,
                "Days_Held": pos["days_held"],
                "Exit_Reason": reason,
            })
            actions.append(
                f"🛑 *STOP HIT ({exit_d}):* Sold `{sym}.NSE` ({pos['qty']} qty @ ₹{exit_p:.2f}) | "
                f"Net: ₹{net_pnl:+,.2f} ({gross_ret:+.1f}%)"
            )

    return closed_trades, actions


def run_monthly_rebalance(
    state: dict, stock_dfs: dict, ranked: list, nifty_bull: bool, latest_date: str
) -> tuple:
    closed_trades = []
    actions = []
    top_buffer_syms = {x["symbol"] for x in ranked[:HOLD_RANK_BUFFER]}

    # 1. Determine Sells (either Nifty < 200 DMA Cash Lock or Rank Drop > Top 6)
    sells = []
    if not nifty_bull:
        for sym in list(state["open_positions"].keys()):
            sells.append((sym, "Nifty < SMA200 Cash Lock"))
    else:
        for sym in list(state["open_positions"].keys()):
            if sym not in top_buffer_syms:
                sells.append((sym, f"Rank Drop (>Top {HOLD_RANK_BUFFER})"))

    for sym, reason in sells:
        pos = state["open_positions"].pop(sym)
        if sym in stock_dfs and latest_date in stock_dfs[sym].index:
            exit_p = round(float(stock_dfs[sym].loc[latest_date, "Close"]), 2)
        else:
            exit_p = round(float(pos["entry_price"]), 2)
        net_pnl = calc_cnc_net_pnl(pos["entry_price"], exit_p, pos["qty"])
        state["cash"] = round(state["cash"] + (pos["qty"] * pos["entry_price"]) + net_pnl, 2)
        gross_ret = round((exit_p / pos["entry_price"] - 1.0) * 100.0, 2)
        closed_trades.append({
            "Entry_Date": pos["entry_date"],
            "Exit_Date": latest_date,
            "Ticker": f"{sym}.NSE",
            "Qty": pos["qty"],
            "Entry_Price": round(pos["entry_price"], 2),
            "Exit_Price": exit_p,
            "Gross_Ret_Pct": gross_ret,
            "Net_PnL_INR": net_pnl,
            "Days_Held": pos.get("days_held", 0),
            "Exit_Reason": reason,
        })
        actions.append(
            f"🔴 *REBAL SELL:* `{sym}.NSE` ({pos['qty']} qty @ ₹{exit_p:.2f}) | "
            f"Net: ₹{net_pnl:+,.2f} ({gross_ret:+.1f}%) [{reason}]"
        )

    # 2. Fill Open Slots with Top-Ranked Leaders (only if Nifty > 200 DMA)
    if nifty_bull:
        slots_needed = MAX_SLOTS - len(state["open_positions"])
        if slots_needed > 0:
            for cand in ranked:
                if len(state["open_positions"]) >= MAX_SLOTS:
                    break
                sym = cand["symbol"]
                if sym in state["open_positions"]:
                    continue
                entry_p = cand["close"]
                if entry_p <= 0:
                    continue

                mtm_open = sum(
                    p["qty"] * (
                        float(stock_dfs[t].loc[latest_date, "Close"])
                        if (t in stock_dfs and latest_date in stock_dfs[t].index)
                        else p["entry_price"]
                    )
                    for t, p in state["open_positions"].items()
                )
                total_eq = state["cash"] + mtm_open
                slot_budget = min(state["cash"], total_eq / MAX_SLOTS)
                qty = int(slot_budget // entry_p)
                if qty < 1:
                    continue

                sl_px = round(entry_p * (1.0 - STOP_LOSS_PCT), 2)
                state["cash"] = round(state["cash"] - qty * entry_p, 2)
                state["open_positions"][sym] = {
                    "entry_date": latest_date,
                    "entry_price": entry_p,
                    "qty": qty,
                    "stop_loss": sl_px,
                    "days_held": 0,
                }
                actions.append(
                    f"🟢 *REBAL BUY:* `{sym}.NSE` ({qty} qty @ ₹{entry_p:.2f} | "
                    f"Value: ₹{qty * entry_p:,.0f} | -12% SL: ₹{sl_px:.2f})"
                )

    state["last_rebalance_month"] = latest_date[:7]
    return closed_trades, actions


# ==============================================================================
# 6. TELEGRAM REPORTING & MAIN RUNNER
# ==============================================================================
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


def build_telegram_report(
    latest_date: str,
    universe_cnt: int,
    nifty_close: float,
    nifty_sma200: float,
    nifty_bull: bool,
    rebalance_triggered: bool,
    actions: list,
    state: dict,
    stock_dfs: dict,
    ranked: list,
    ledger_df: pd.DataFrame,
    total_eq: float,
) -> str:
    nifty_dist = ((nifty_close / nifty_sma200) - 1.0) * 100.0
    regime_badge = "🟢 BULL (Active)" if nifty_bull else "🔴 BEAR (100% Cash Lock)"
    run_tag = "🔄 MONTHLY REBALANCE" if rebalance_triggered else "👁️ DAILY STOP & RANK CHECK"
    rank_lookup = {x["symbol"]: idx + 1 for idx, x in enumerate(ranked)}

    lines = [
        f"📈 *Mom_2Slot_12SL Paper Bot ({latest_date})*",
        f"_{run_tag} | Scanned {universe_cnt} NSE F&O Stocks_",
        f"• *Nifty 50 Regime:* {regime_badge} (`{nifty_close:,.1f}` vs 200DMA `{nifty_sma200:,.1f}`, `{nifty_dist:+.1f}%`)",
    ]

    if actions:
        lines.append("\n⚡ *Today's Executions:*")
        lines.extend(actions)
    else:
        lines.append("• *Today's Action:* No stops hit and no rebalance changes.")

    lines.append("\n💼 *Open Positions:*")
    if not state["open_positions"]:
        lines.append("• 100% Cash (0 open slots)")
    else:
        for sym, pos in state["open_positions"].items():
            cur_px = (
                float(stock_dfs[sym].loc[latest_date, "Close"])
                if (sym in stock_dfs and latest_date in stock_dfs[sym].index)
                else pos["entry_price"]
            )
            unr_pct = ((cur_px / pos["entry_price"]) - 1.0) * 100.0
            unr_inr = (cur_px - pos["entry_price"]) * pos["qty"]
            rk = rank_lookup.get(sym, None)
            rk_str = f"Rank #{rk}" if rk else "Unranked"
            lines.append(
                f"• `{sym}.NSE` ({pos['qty']} qty | {rk_str}) | Entry ₹{pos['entry_price']:.2f} → "
                f"LTP ₹{cur_px:.2f} (`{unr_pct:+.1f}%`, ₹{unr_inr:+,.0f}) | SL ₹{pos['stop_loss']:.2f}"
            )

    lines.append(f"\n🏆 *Top {HOLD_RANK_BUFFER} Momentum Leaders (Buffer Zone):*")
    if not ranked:
        lines.append("• No stocks passed trend filters today.")
    else:
        for idx, x in enumerate(ranked[:HOLD_RANK_BUFFER], start=1):
            held = " ⭐" if x["symbol"] in state["open_positions"] else ""
            lines.append(
                f"{idx}. `{x['symbol']}`{held} — ₹{x['close']} | Score `{x['mom_score']:.2f}` "
                f"(60d `{x['roc60']:+.0f}%`, 120d `{x['roc120']:+.0f}%`)"
            )

    tot_ret = ((total_eq / START_CAPITAL) - 1.0) * 100.0
    lines.append(
        f"\n💰 *Portfolio Summary (Start ₹{START_CAPITAL:,.0f}):*\n"
        f"• *Total Equity:* ₹{total_eq:,.2f} (`{tot_ret:+.2f}%`) | *Cash:* ₹{state['cash']:,.2f}"
    )

    if not ledger_df.empty:
        n = len(ledger_df)
        wr = (ledger_df["Net_PnL_INR"] > 0).mean() * 100.0
        gw = ledger_df[ledger_df["Net_PnL_INR"] > 0]["Net_PnL_INR"].sum()
        gl = abs(ledger_df[ledger_df["Net_PnL_INR"] <= 0]["Net_PnL_INR"].sum())
        pf = (gw / gl) if gl > 0 else 99.0
        realized = float(ledger_df["Net_PnL_INR"].sum())
        lines.append(
            f"• *Closed Trades ({n}):* WR `{wr:.1f}%` | PF `{pf:.2f}` | Realized: ₹{realized:+,.2f}"
        )

    return "\n".join(lines)


def main():
    force_rebal = os.environ.get("FORCE_REBALANCE", "false").strip().lower() == "true"

    nifty_df = fetch_nifty_daily()
    nifty_dates = sorted(nifty_df.index.tolist())
    latest_date = nifty_dates[-1]
    latest_month = latest_date[:7]

    n_row = nifty_df.loc[latest_date]
    nifty_close = float(n_row["Close"])
    nifty_sma200 = float(n_row["SMA200"])
    nifty_bull = (not np.isnan(nifty_sma200)) and (nifty_close > nifty_sma200)

    print(f"=== Mom_2Slot_12SL Paper Bot | Market Date: {latest_date} ===")
    symbols = load_nse_fno_symbols()
    stock_dfs = fetch_batch_daily_yfinance(symbols)
    ranked = compute_rankings(stock_dfs, latest_date)

    state = load_state()
    already_ran_today = state.get("last_run_date", "") == latest_date

    all_new_trades = []
    all_actions = []
    rebalance_triggered = False

    if not already_ran_today:
        stop_trades, stop_actions = check_daily_stops(state, stock_dfs, nifty_dates, latest_date)
        all_new_trades.extend(stop_trades)
        all_actions.extend(stop_actions)

    need_rebalance = force_rebal or (state.get("last_rebalance_month", "") != latest_month)
    if need_rebalance and (not already_ran_today or force_rebal):
        rebalance_triggered = True
        reb_trades, reb_actions = run_monthly_rebalance(state, stock_dfs, ranked, nifty_bull, latest_date)
        all_new_trades.extend(reb_trades)
        all_actions.extend(reb_actions)

    state["last_run_date"] = latest_date

    open_mtm = sum(
        pos["qty"] * (
            float(stock_dfs[sym].loc[latest_date, "Close"])
            if (sym in stock_dfs and latest_date in stock_dfs[sym].index)
            else pos["entry_price"]
        )
        for sym, pos in state["open_positions"].items()
    )
    total_eq = round(state["cash"] + open_mtm, 2)

    save_state(state)
    ledger_df = append_closed_trades(all_new_trades)
    update_equity_log({
        "Date": latest_date,
        "Cash": round(state["cash"], 2),
        "Open_MTM": round(open_mtm, 2),
        "Total_Equity": total_eq,
        "Nifty_Close": round(nifty_close, 2),
        "Nifty_SMA200": round(nifty_sma200, 2),
        "Regime": "BULL" if nifty_bull else "CASH_LOCK",
        "Open_Positions": ",".join(sorted(state["open_positions"].keys())) if state["open_positions"] else "CASH",
    })

    report = build_telegram_report(
        latest_date, len(stock_dfs), nifty_close, nifty_sma200, nifty_bull,
        rebalance_triggered, all_actions, state, stock_dfs, ranked, ledger_df, total_eq
    )
    print("\n" + report)
    send_telegram_text(report)


if __name__ == "__main__":
    main()
