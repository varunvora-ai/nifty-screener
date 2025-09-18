import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os

# ===== CONFIG =====
CONFIG = {
    "telegram_bot_token": "8118559525:AAEtXukIxcb_0BC3gnkCZPXu9KTOwQPVCPA",
    "telegram_chat_id": "846980447",   # Replace with your numeric chat_id if this fails
    "lbb_length": 20,
    "lbb_mult": 2.0,
    "rsi_length": 14,
    "vol_length": 20,
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_h1_length": 50,  # approximated with daily EMA
    "sma_h1_length": 50,
    "use_pinbar_filter": False,
    "use_volume_filter": False,
    "volume_multiplier": 1.5,
    "cache_file": "last_signals.json",  # dedupe storage
}
# ==================

# --- Fetch Nifty500 symbols from NSE CSV ---
from nifty500_symbols import NIFTY500_SYMBOLS

def fetch_nifty500_symbols():
    return NIFTY500_SYMBOLS
    symbols = df["Symbol"].tolist()
    symbols = [s.strip().upper() + ".NS" for s in symbols]
    return sorted(set(symbols))

# --- Indicators ---
def sma(series, length): return series.rolling(length).mean()
def ema(series, length): return series.ewm(span=length, adjust=False).mean()
def rsi(series, length):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up/ma_down
    return 100 - (100/(1+rs))
def compute_lbb(close, length, mult):
    logc = np.log(close)
    basis = logc.rolling(length).mean()
    dev = mult*logc.rolling(length).std()
    return np.exp(basis-dev), np.exp(basis), np.exp(basis+dev)
def pinbar(o,h,l,c):
    body = abs(c-o); upw = h-max(c,o); loww = min(c,o)-l
    bull = (loww > body*2) and (upw < body)
    bear = (upw > body*2) and (loww < body)
    return bull,bear

# --- Telegram ---
def send_telegram(text):
    url = f"https://api.telegram.org/bot{CONFIG['telegram_bot_token']}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CONFIG["telegram_chat_id"], "text": text}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)

# --- Commentary for signals ---
def signal_commentary(signal, rsi):
    if "Buy" in signal:
        if rsi < 25: return "📉 Very oversold, possible rebound"
        elif rsi < 35: return "⚠️ Oversold but risky"
        else: return "🔎 Weak buy signal, momentum uncertain"
    elif "Sell" in signal:
        if rsi > 75: return "📈 Very overbought, reversal likely"
        elif rsi > 65: return "⚠️ Overbought, cautious short"
        else: return "🔎 Weak sell signal, momentum uncertain"
    return ""

# --- Analyzer for one ticker ---
def analyze_ticker(df, symbol):
    try:
        df["RSI"] = rsi(df["Close"], CONFIG["rsi_length"])
        df["EMA_fast"] = ema(df["Close"], CONFIG["ema_fast"])
        df["EMA_slow"] = ema(df["Close"], CONFIG["ema_slow"])
        df["EMA_h1"] = ema(df["Close"], CONFIG["ema_h1_length"])
        df["SMA_h1"] = sma(df["Close"], CONFIG["sma_h1_length"])
        df["Vol_SMA"] = sma(df["Volume"], CONFIG["vol_length"])
        df["LBB_low"], df["LBB_basis"], df["LBB_up"] = compute_lbb(df["Close"], CONFIG["lbb_length"], CONFIG["lbb_mult"])

        last, prev = df.iloc[-1], df.iloc[-2]
        vol_ok = (not CONFIG["use_volume_filter"]) or (last["Volume"] > last["Vol_SMA"]*CONFIG["volume_multiplier"])
        pin_bull,pin_bear = pinbar(last["Open"], last["High"], last["Low"], last["Close"])
        base_long = (last["Close"] < last["LBB_low"]) and (not CONFIG["use_pinbar_filter"] or pin_bull) and vol_ok
        base_short = (last["Close"] > last["LBB_up"]) and (not CONFIG["use_pinbar_filter"] or pin_bear) and vol_ok

        # Trend filters
        ema_cross_long, ema_cross_short = last["EMA_fast"]>last["EMA_slow"], last["EMA_fast"]<last["EMA_slow"]
        ema_trend_long, ema_trend_short = (last["Close"]>last["EMA_h1"])&(last["EMA_h1"]>prev["EMA_h1"]), (last["Close"]<last["EMA_h1"])&(last["EMA_h1"]<prev["EMA_h1"])
        sma_trend_long, sma_trend_short = (last["Close"]>last["SMA_h1"])&(last["SMA_h1"]>prev["SMA_h1"]), (last["Close"]<last["SMA_h1"])&(last["SMA_h1"]<prev["SMA_h1"])

        base_long = base_long and ema_cross_long and ema_trend_long and sma_trend_long
        base_short = base_short and ema_cross_short and ema_trend_short and sma_trend_short

        sig="-"
        if base_long and last["RSI"]<30: sig="🟢 Strong Buy"
        elif base_long and last["RSI"]<35: sig="🟠 Medium Buy"
        elif base_long and last["RSI"]<40: sig="🟡 Weak Buy"
        elif base_short and last["RSI"]>70: sig="🔴 Strong Sell"
        elif base_short and last["RSI"]>65: sig="🟣 Medium Sell"
        elif base_short and last["RSI"]>60: sig="🔵 Weak Sell"

        return {"symbol":symbol,"close":last["Close"],"rsi":last["RSI"],"signal":sig}
    except Exception:
        return None

# --- Load/Save Dedupe Cache ---
def load_cache():
    if os.path.exists(CONFIG["cache_file"]):
        with open(CONFIG["cache_file"], "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CONFIG["cache_file"], "w") as f:
        json.dump(cache, f)

# --- Main Screener ---
def run_screener():
    symbols = fetch_nifty500_symbols()
    print(f"Fetched {len(symbols)} Nifty500 symbols from NSE")

    data = yf.download(symbols, period="6mo", interval="1d", group_by="ticker", progress=False, threads=True)

    last_cache = load_cache()
    today_cache = {}

    results=[]
    alerts_sent=[]
    for sym in symbols:
        try:
            df = data[sym].dropna()
            if df.empty:
                continue
            res = analyze_ticker(df, sym)
            if res:
                results.append(res)
                today_cache[sym] = res["signal"]
                print(res)

                # Alert only if NEW signal (different from last run)
                if res["signal"] != "-" and last_cache.get(sym) != res["signal"]:
                    msg = f"{res['symbol']} {res['signal']} @ {res['close']:.2f}, RSI={res['rsi']:.1f}\n{signal_commentary(res['signal'], res['rsi'])}"
                    send_telegram(msg)
                    alerts_sent.append(res)
        except Exception as e:
            print(f"Error {sym}:", e)

    save_cache(today_cache)

    # --- Summary Message ---
    if alerts_sent:
        summary = pd.DataFrame(alerts_sent)
        counts = summary["signal"].value_counts().to_dict()
        summary_msg = "📊 Daily Screener Summary:\n"
        for sig, count in counts.items():
            summary_msg += f"{sig}: {count}\n"

        # Top 5 strongest signals
        buys = summary[summary.signal.str.contains("Buy")].sort_values("rsi").head(5)
        sells = summary[summary.signal.str.contains("Sell")].sort_values("rsi", ascending=False).head(5)

        if not buys.empty:
            summary_msg += "\n🔥 Top 5 Buys:\n"
            for _, row in buys.iterrows():
                summary_msg += f"{row['symbol']} {row['signal']} (RSI {row['rsi']:.1f}) → {signal_commentary(row['signal'], row['rsi'])}\n"

        if not sells.empty:
            summary_msg += "\n⚡ Top 5 Sells:\n"
            for _, row in sells.iterrows():
                summary_msg += f"{row['symbol']} {row['signal']} (RSI {row['rsi']:.1f}) → {signal_commentary(row['signal'], row['rsi'])}\n"

        send_telegram(summary_msg.strip())
    else:
        send_telegram("📊 Daily Screener Summary: No new signals today.")

    return pd.DataFrame(results)

if __name__=="__main__":
    df=run_screener()
    print("\n--- Screener Results ---")
    if not df.empty:
        print(df[df["signal"] != "-"])
    else:
        print("No signals found today.")
