import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="Pattern Scanner (MEXC)", layout="wide")

FUT = "https://contract.mexc.com/api/v1/contract"
# label: (futures interval, seconds)
TF = {"15m": ("Min15", 900), "1h": ("Min60", 3600),
      "4h": ("Hour4", 14400), "1d": ("Day1", 86400)}
PATTERNS = ["Hammer", "Inverted Hammer", "Bullish Engulfing", "Bearish Engulfing",
            "100 EMA", "20 Candle Breakout", "30 Candle Breakout",
            "RSI Divergence", "Volume Bubble", "SFP"]


# ------------------------------------------------------------------ data
def get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=15,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429:
                time.sleep(1.2 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(0.6)


@st.cache_data(ttl=60)
def load_universe():
    df = pd.DataFrame(get(f"{FUT}/ticker")["data"])
    df = df[df.symbol.str.endswith("_USDT")]
    return pd.DataFrame({
        "symbol": df.symbol, "price": df.lastPrice.astype(float),
        "chg24": df.riseFallRate.astype(float) * 100,
        "vol24": df.amount24.astype(float),
        "funding": df.fundingRate.astype(float) * 100}).reset_index(drop=True)


@st.cache_data(ttl=90)
def klines(symbol, tf, limit=300):
    iv, sec = TF[tf]
    end = int(time.time())
    d = get(f"{FUT}/kline/{symbol}",
            {"interval": iv, "start": end - sec * limit, "end": end})["data"]
    df = pd.DataFrame({"t": d["time"], "o": d["open"], "h": d["high"],
                       "l": d["low"], "c": d["close"], "v": d["vol"]}).astype(float)
    df["dt"] = pd.to_datetime(df.t, unit="s")
    return df.iloc[:-1].reset_index(drop=True)  # drop still-forming candle


# ------------------------------------------------------------------ indicators
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def pivots(s, n, kind):
    v, out = s.values, []
    for i in range(n, len(v) - n):
        w = v[i - n:i + n + 1]
        if (kind == "low" and v[i] == w.min()) or (kind == "high" and v[i] == w.max()):
            out.append(i)
    return out


# ------------------------------------------------------------------ detection
def detect(df, p):
    """Signals on the LAST candle of df. Returns list of (pattern, side)."""
    if len(df) < 130:
        return []
    o, h, l, c, v = df.o, df.h, df.l, df.c, df.v
    oo, hh, ll, cc = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    body, rng = abs(cc - oo), hh - ll
    lower, upper = min(oo, cc) - ll, hh - max(oo, cc)
    down = c.iloc[-2] < c.iloc[-2 - p["trend"]]
    out = []

    if rng > 0 and down and lower >= 2 * body and upper <= 0.15 * rng:
        out.append(("Hammer", "Bull"))
    if rng > 0 and down and upper >= 2 * body and lower <= 0.15 * rng:
        out.append(("Inverted Hammer", "Bull"))

    po, pc = o.iloc[-2], c.iloc[-2]
    if pc < po and cc > oo and oo <= pc and cc >= po and body > abs(pc - po):
        out.append(("Bullish Engulfing", "Bull"))
    if pc > po and cc < oo and oo >= pc and cc <= po and body > abs(pc - po):
        out.append(("Bearish Engulfing", "Bear"))

    ema = c.ewm(span=100, adjust=False).mean()
    if c.iloc[-2] < ema.iloc[-2] and cc > ema.iloc[-1]:
        out.append(("100 EMA", "Bull"))
    if c.iloc[-2] > ema.iloc[-2] and cc < ema.iloc[-1]:
        out.append(("100 EMA", "Bear"))

    for n in (20, 30):
        hi, lo = h.iloc[-n - 1:-1].max(), l.iloc[-n - 1:-1].min()
        if cc > hi and c.iloc[-2] <= h.iloc[-n - 2:-2].max():
            out.append((f"{n} Candle Breakout", "Bull"))
        if cc < lo and c.iloc[-2] >= l.iloc[-n - 2:-2].min():
            out.append((f"{n} Candle Breakout", "Bear"))

    r = rsi(c)
    n = len(df)
    pl, ph = pivots(l, 3, "low"), pivots(h, 3, "high")
    if len(pl) >= 2:
        a, b = pl[-2], pl[-1]
        if n - 1 - b <= 10 and 5 <= b - a <= 60 and l[b] < l[a] and r[b] > r[a] and r[a] < 40:
            out.append(("RSI Divergence", "Bull"))
    if len(ph) >= 2:
        a, b = ph[-2], ph[-1]
        if n - 1 - b <= 10 and 5 <= b - a <= 60 and h[b] > h[a] and r[b] < r[a] and r[a] > 60:
            out.append(("RSI Divergence", "Bear"))

    avg = v.iloc[-21:-1].mean()
    if avg > 0 and v.iloc[-1] >= p["vol_mult"] * avg:
        out.append(("Volume Bubble", "Bull" if cc >= oo else "Bear"))

    sh, sl = h.iloc[-p["sfp_n"] - 1:-1].max(), l.iloc[-p["sfp_n"] - 1:-1].min()
    if hh > sh and cc < sh:
        out.append(("SFP", "Bear"))
    if ll < sl and cc > sl:
        out.append(("SFP", "Bull"))
    return out


def scan_symbol(sym, tf, p, recent, wanted):
    try:
        df = klines(sym, tf)
        rows = []
        for k in range(recent):
            sub = df.iloc[:len(df) - k] if k else df
            for pat, side in detect(sub, p):
                if pat in wanted:
                    rows.append((pat, side, k))
        return sym, rows
    except Exception:
        return sym, None


# ------------------------------------------------------------------ sidebar
st.title("Pattern Scanner - MEXC Futures")
with st.sidebar:
    tf = st.selectbox("Timeframe", list(TF), index=2)
    top_n = st.slider("Top coins by 24h volume", 20, 200, 60, 10)
    min_vol = st.number_input("Min 24h volume (USDT)", value=1_000_000, step=500_000)
    wanted = st.multiselect("Patterns", PATTERNS, default=PATTERNS)
    recent = st.slider("Look back N closed candles", 1, 5, 1,
                       help="1 = only the latest closed candle")
    with st.expander("Pattern settings"):
        p = {"trend": st.slider("Downtrend check (candles) for hammers", 3, 10, 5),
             "vol_mult": st.slider("Volume bubble: x avg volume", 1.5, 6.0, 2.5, 0.5),
             "sfp_n": st.slider("SFP swing lookback", 10, 50, 20)}
    run = st.button("Run scan", type="primary", use_container_width=True)

uni = load_universe()
uni = uni[uni.vol24 >= min_vol].sort_values("vol24", ascending=False).head(top_n)

t1, t2, t3 = st.tabs(["Scanner", "Chart", "Exchange comparison"])

# ------------------------------------------------------------------ scanner
with t1:
    if run:
        res, bar = [], st.progress(0.0, "Scanning...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(scan_symbol, s, tf, p, recent, wanted) for s in uni.symbol]
            for i, f in enumerate(futs):
                res.append(f.result())
                bar.progress((i + 1) / len(futs), f"Scanned {i + 1}/{len(futs)}")
        bar.empty()
        st.session_state["res"] = (res, tf)

    if "res" not in st.session_state:
        st.info("Set filters on the left and press Run scan.")
    else:
        res, tf_used = st.session_state["res"]
        failed = [s for s, r in res if r is None]
        info = uni.set_index("symbol")
        rows = []
        for sym, sigs in res:
            if not sigs or sym not in info.index:
                continue
            bull = sum(1 for _, sd, _ in sigs if sd == "Bull")
            bear = sum(1 for _, sd, _ in sigs if sd == "Bear")
            tags = [f"{'+' if sd == 'Bull' else '-'} {pt}" + (f" ({k}c ago)" if k else "")
                    for pt, sd, k in sigs]
            rows.append({"Symbol": sym, "Price": info.loc[sym, "price"],
                         "24h %": round(info.loc[sym, "chg24"], 2),
                         "Funding %": round(info.loc[sym, "funding"], 4),
                         "Vol24 (M)": round(info.loc[sym, "vol24"] / 1e6, 1),
                         "Bias": "Bullish" if bull > bear else "Bearish" if bear > bull else "Mixed",
                         "Score": bull - bear, "Signals": ", ".join(tags)})
        out = pd.DataFrame(rows)
        st.caption(f"Futures | {tf_used} | {len(res)} coins scanned | {len(out)} with signals"
                   + (f" | {len(failed)} failed (rate limit?)" if failed else ""))
        if out.empty:
            st.warning("No signals found with current filters.")
        else:
            counts = {pt: sum(1 for _, r in res if r for x, _, _ in r if x == pt) for pt in wanted}
            cols = st.columns(5)
            for i, (pt, n) in enumerate(counts.items()):
                cols[i % 5].metric(pt, n)
            bias = st.multiselect("Bias filter", ["Bullish", "Bearish", "Mixed"],
                                  default=["Bullish", "Bearish", "Mixed"])
            need = st.multiselect("Must include pattern", wanted)
            view = out[out.Bias.isin(bias)]
            for pt in need:
                view = view[view.Signals.str.contains(pt, regex=False)]
            st.dataframe(view.sort_values("Score", ascending=False), hide_index=True,
                         use_container_width=True, height=560)
            st.download_button("Download CSV", view.to_csv(index=False), "scan.csv")

# ------------------------------------------------------------------ chart
with t3:
    st.subheader("Funding rate: MEXC vs Binance vs Bybit")
    st.caption("Big gaps between exchanges show where positioning is crowded. "
               "Binance/Bybit may block some hosting regions; if so that column stays empty.")
    if st.button("Load comparison"):
        cmp = uni[["symbol", "price", "funding"]].copy()
        cmp["key"] = cmp.symbol.str.replace("_", "")
        cmp = cmp.rename(columns={"funding": "MEXC %"})
        try:
            b = pd.DataFrame(get("https://fapi.binance.com/fapi/v1/premiumIndex"))
            b["Binance %"] = b.lastFundingRate.astype(float) * 100
            cmp = cmp.merge(b[["symbol", "Binance %"]].rename(columns={"symbol": "key"}), on="key", how="left")
        except Exception:
            cmp["Binance %"] = np.nan
        try:
            y = get("https://api.bybit.com/v5/market/tickers", {"category": "linear"})["result"]["list"]
            y = pd.DataFrame(y)
            y["Bybit %"] = pd.to_numeric(y.fundingRate, errors="coerce") * 100
            cmp = cmp.merge(y[["symbol", "Bybit %"]].rename(columns={"symbol": "key"}), on="key", how="left")
        except Exception:
            cmp["Bybit %"] = np.nan
        cmp["Max gap"] = cmp[["MEXC %", "Binance %", "Bybit %"]].max(axis=1) - \
            cmp[["MEXC %", "Binance %", "Bybit %"]].min(axis=1)
        st.dataframe(cmp.drop(columns="key").sort_values("Max gap", ascending=False)
                     .round(4), hide_index=True, use_container_width=True, height=560)

with t2:
    sym = st.selectbox("Symbol", list(uni.symbol) or ["BTC_USDT"])
    if sym:
        full = klines(sym, tf)
        df = full.tail(150)
        ema = full.c.ewm(span=100, adjust=False).mean().tail(150)
        rs = rsi(full.c).tail(150)
        avg = full.v.rolling(20).mean().tail(150)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.dt, open=df.o, high=df.h, low=df.l, close=df.c,
                                     name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.dt, y=ema, name="EMA 100"), row=1, col=1)
        for n, dash in ((20, "dot"), (30, "dash")):
            fig.add_trace(go.Scatter(x=df.dt, y=full.h.rolling(n).max().shift(1).tail(150),
                                     name=f"{n}c high", line=dict(dash=dash, width=1)), row=1, col=1)
        bub = df.v >= p["vol_mult"] * avg
        fig.add_trace(go.Bar(x=df.dt, y=df.v, name="Volume",
                             marker_color=np.where(bub, "orange", "gray")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.dt, y=rs, name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", row=3, col=1)
        fig.update_layout(height=760, xaxis_rangeslider_visible=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.write("Latest signals:", ", ".join(f"{s} ({sd})" for s, sd in detect(full, p)) or "none")
