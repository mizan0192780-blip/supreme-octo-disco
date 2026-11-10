‎
‎import os
‎import sqlite3
‎import requests
‎import numpy as np
‎import pandas as pd
‎import streamlit as st
‎import plotly.graph_objects as go
‎
‎from sklearn.ensemble import RandomForestClassifier
‎from sklearn.metrics import accuracy_score, confusion_matrix
‎
‎# =====================================
‎# CONFIG
‎# =====================================
‎
‎st.set_page_config(
‎    page_title="Pocket Broker AI",
‎    page_icon="📈",
‎    layout="wide"
‎)
‎
‎DB = "pocket_ai.db"
‎FEATURES = [
‎    "return1", "return3", "return5",
‎    "body", "range", "upper_wick", "lower_wick",
‎    "ema_gap", "rsi", "macd", "macd_diff",
‎    "atr", "volatility"
‎]
‎
‎# =====================================
‎# DATABASE
‎# =====================================
‎
‎def db():
‎    return sqlite3.connect(DB)
‎
‎def init_db():
‎    with db() as con:
‎        con.execute("""
‎        CREATE TABLE IF NOT EXISTS signals (
‎            id INTEGER PRIMARY KEY AUTOINCREMENT,
‎            source TEXT,
‎            symbol TEXT,
‎            candle_time TEXT,
‎            direction TEXT,
‎            probability REAL,
‎            entry REAL,
‎            outcome TEXT DEFAULT 'PENDING'
‎        )
‎        """)
‎
‎init_db()
‎
‎# =====================================
‎# TWELVE DATA API
‎# =====================================
‎
‎def get_api_key():
‎    try:
‎        return st.secrets["TWELVE_DATA_API_KEY"]
‎    except Exception:
‎        return os.getenv("TWELVE_DATA_API_KEY", "")
‎
‎def fetch_live(symbol):
‎    key = get_api_key()
‎
‎    if not key:
‎        raise ValueError("Twelve Data API key is missing.")
‎
‎    response = requests.get(
‎        "https://api.twelvedata.com/time_series",
‎        params={
‎            "symbol": symbol,
‎            "interval": "1min",
‎            "outputsize": 1000,
‎            "timezone": "UTC",
‎            "order": "ASC",
‎            "apikey": key
‎        },
‎        timeout=20
‎    )
‎
‎    response.raise_for_status()
‎    payload = response.json()
‎
‎    if payload.get("status") == "error":
‎        raise ValueError(payload.get("message", "API error"))
‎
‎    if "values" not in payload:
‎        raise ValueError("No OHLC data returned.")
‎
‎    df = pd.DataFrame(payload["values"])
‎
‎    df = df.rename(columns={
‎        "datetime": "timestamp"
‎    })
‎
‎    return df
‎
‎# =====================================
‎# CSV / DATA CLEANING
‎# =====================================
‎
‎def clean_data(df):
‎
‎    df = df.copy()
‎    df.columns = df.columns.str.lower().str.strip()
‎
‎    required = ["open", "high", "low", "close"]
‎
‎    for col in required:
‎        if col not in df.columns:
‎            raise ValueError("Missing column: " + col)
‎
‎        df[col] = pd.to_numeric(
‎            df[col], errors="coerce"
‎        )
‎
‎    if "timestamp" in df.columns:
‎        df["timestamp"] = pd.to_datetime(
‎            df["timestamp"],
‎            errors="coerce",
‎            utc=True
‎        )
‎
‎        df = df.dropna(subset=["timestamp"])
‎        df = df.sort_values("timestamp")
‎
‎    else:
‎        df["timestamp"] = pd.date_range(
‎            end=pd.Timestamp.now(tz="UTC"),
‎            periods=len(df),
‎            freq="min"
‎        )
‎
‎    df = df.dropna(subset=required)
‎
‎    if df.empty:
‎        raise ValueError("No valid OHLC records.")
‎
‎    if (
‎        (df["open"] <= 0).any()
‎        or (df["close"] <= 0).any()
‎    ):
‎        raise ValueError("Prices must be positive.")
‎
‎    if (
‎        (df["high"] < df["low"]).any()
‎        or (df["high"] < df[["open", "close"]].max(axis=1)).any()
‎        or (df["low"] > df[["open", "close"]].min(axis=1)).any()
‎    ):
‎        raise ValueError("Invalid OHLC candle.")
‎
‎    df = df.drop_duplicates(
‎        subset=["timestamp"], keep="last"
‎    )
‎
‎    return df.reset_index(drop=True)
‎
‎# =====================================
‎# TECHNICAL INDICATORS
‎# =====================================
‎
‎def indicators(df):
‎
‎    df = df.copy()
‎
‎    close = df["close"]
‎    open_ = df["open"]
‎
‎    df["return1"] = close.pct_change(1)
‎    df["return3"] = close.pct_change(3)
‎    df["return5"] = close.pct_change(5)
‎
‎    df["body"] = (close - open_) / close
‎
‎    df["range"] = (
‎        df["high"] - df["low"]
‎    ) / close
‎
‎    df["upper_wick"] = (
‎        df["high"] -
‎        df[["open", "close"]].max(axis=1)
‎    ) / close
‎
‎    df["lower_wick"] = (
‎        df[["open", "close"]].min(axis=1) -
‎        df["low"]
‎    ) / close
‎
‎    ema9 = close.ewm(
‎        span=9, adjust=False
‎    ).mean()
‎
‎    ema21 = close.ewm(
‎        span=21, adjust=False
‎    ).mean()
‎
‎    df["ema_gap"] = (ema9 - ema21) / close
‎
‎    delta = close.diff()
‎
‎    gain = delta.clip(lower=0)
‎    loss = -delta.clip(upper=0)
‎
‎    avg_gain = gain.ewm(
‎        alpha=1/14, adjust=False
‎    ).mean()
‎
‎    avg_loss = loss.ewm(
‎        alpha=1/14, adjust=False
‎    ).mean()
‎
‎    rs = avg_gain / avg_loss.replace(0, np.nan)
‎
‎    df["rsi"] = 100 - 100 / (1 + rs)
‎
‎    fast = close.ewm(span=12, adjust=False).mean()
‎    slow = close.ewm(span=26, adjust=False).mean()
‎
‎    macd = fast - slow
‎    macd_signal = macd.ewm(
‎        span=9, adjust=False
‎    ).mean()
‎
‎    df["macd"] = macd / close
‎    df["macd_diff"] = (macd - macd_signal) / close
‎
‎    previous = close.shift(1)
‎
‎    tr = pd.concat([
‎        df["high"] - df["low"],
‎        (df["high"] - previous).abs(),
‎        (df["low"] - previous).abs()
‎    ], axis=1).max(axis=1)
‎
‎    df["atr"] = tr.rolling(14).mean() / close
‎
‎    df["volatility"] = (
‎        df["return1"].rolling(10).std()
‎    )
‎
‎    # Next candle close direction
‎    next_close = close.shift(-1)
‎
‎    df["target"] = np.where(
‎        next_close.notna(),
‎        (next_close > close).astype(float),
‎        np.nan
‎    )
‎
‎    return df
‎
‎# =====================================
‎# MODEL AND WALK-FORWARD BACKTEST
‎# =====================================
‎
‎def build_model():
‎
‎    return RandomForestClassifier(
‎        n_estimators=250,
‎        max_depth=8,
‎        min_samples_leaf=5,
‎        class_weight="balanced",
‎        random_state=42,
‎        n_jobs=-1
‎    )
‎
‎def backtest(df):
‎
‎    data = df.dropna(
‎        subset=FEATURES + ["target"]
‎    ).reset_index(drop=True)
‎
‎    if len(data) < 350:
‎        raise ValueError(
‎            "Need at least 350 valid labeled candles."
‎        )
‎
‎    split = int(len(data) * 0.80)
‎
‎    X = data[FEATURES]
‎    y = data["target"].astype(int)
‎
‎    predictions = []
‎    probabilities = []
‎
‎    # Expanding-window walk-forward testing
‎    for i in range(split, len(data)):
‎
‎        model = build_model()
‎
‎        model.fit(
‎            X.iloc[:i],
‎            y.iloc[:i]
‎        )
‎
‎        pred = model.predict(
‎            X.iloc[[i]]
‎        )[0]
‎
‎        proba = model.predict_proba(
‎            X.iloc[[i]]
‎        )[0]
‎
‎        up_index = list(model.classes_).index(1)
‎
‎        predictions.append(pred)
‎        probabilities.append(proba[up_index])
‎
‎    actual = y.iloc[split:].to_numpy()
‎
‎    accuracy = accuracy_score(
‎        actual, predictions
‎    )
‎
‎    return {
‎        "accuracy": accuracy,
‎        "actual": actual,
‎        "predictions": np.array(predictions),
‎        "probabilities": probabilities,
‎        "count": len(actual)
‎    }
‎
‎# =====================================
‎# FINAL PREDICTION
‎# =====================================
‎
‎def predict_next(df):
‎
‎    labeled = df.dropna(
‎        subset=FEATURES + ["target"]
‎    )
‎
‎    latest = df.dropna(
‎        subset=FEATURES
‎    ).tail(1)
‎
‎    if len(labeled) < 300 or latest.empty:
‎        raise ValueError("Not enough training data.")
‎
‎    y = labeled["target"].astype(int)
‎
‎    if y.nunique() != 2:
‎        raise ValueError("Need both UP and DOWN examples.")
‎
‎    model = build_model()
‎
‎    model.fit(
‎        labeled[FEATURES],
‎        y
‎    )
‎
‎    probabilities = model.predict_proba(
‎        latest[FEATURES]
‎    )[0]
‎
‎    up_index = list(model.classes_).index(1)
‎
‎    up = float(probabilities[up_index])
‎    down = 1 - up
‎
‎    if up >= 0.60:
‎        direction = "UP"
‎    elif down >= 0.60:
‎        direction = "DOWN"
‎    else:
‎        direction = "WAIT"
‎
‎    return direction, up, down
‎
‎# =====================================
‎# SIGNAL STORAGE
‎# =====================================
‎
‎
‎def settle_signals(df, source, symbol):
‎
‎    with db() as con:
‎
‎        pending = con.execute("""
‎            SELECT id, candle_time, direction, entry
‎            FROM signals
‎            WHERE source=? AND symbol=?
‎            AND outcome='PENDING'
‎        """, (source, symbol)).fetchall()
‎
‎        for signal_id, candle_time, direction, entry in pending:
‎
‎            signal_time = pd.to_datetime(
‎                candle_time, utc=True
‎            )
‎
‎            next_rows = df[
‎                (df["timestamp"] > signal_time) &
‎                (df["timestamp"] <= signal_time +
‎                 pd.Timedelta(minutes=2))
‎            ]
‎
‎            if next_rows.empty:
‎                continue
‎
‎            next_close = float(next_rows.iloc[0]["close"])
‎
‎            if direction == "UP":
‎                outcome = (
‎                    "WIN" if next_close > entry
‎                    else "LOSS" if next_close < entry
‎                    else "DRAW"
‎                )
‎
‎            elif direction == "DOWN":
‎                outcome = (
‎                    "WIN" if next_close < entry
‎                    else "LOSS" if next_close > entry
‎                    else "DRAW"
‎                )
‎
‎            else:
‎                outcome = "SKIPPED"
‎
‎            con.execute("""
‎                UPDATE signals SET outcome=?
‎                WHERE id=?
‎            """, (outcome, signal_id))
‎=====================================
‎# DASHBOARD
‎# =====================================
‎
‎st.title("📈 POCKET BROKER AI")
‎st.caption("Live Market Analysis | AI | Backtesting")
‎
‎with st.sidebar:
‎
‎    st.header("Market Settings")
‎
‎    category = st.selectbox(
‎        "Market",
‎        [
‎            "Forex",
‎            "OTC Forex",
‎            "Crypto",
‎            "Gold",
‎            "Stocks",
‎            "Indices",
‎            "Commodities"
‎        ]
‎    )
‎
‎    symbol = st.text_input(
‎        "Market Symbol",
‎        value={
‎            "Forex": "EUR/USD",
‎            "OTC Forex": "EUR/USD OTC",
‎            "Crypto": "BTC/USD",
‎            "Gold": "XAU/USD",
‎            "Stocks": "AAPL",
‎            "Indices": "IXIC",
‎            "Commodities": "XTI/USD"
‎        }[category]
‎    )
‎
‎    source = st.radio(
‎        "Data Source",
‎        ["Live API", "CSV Upload"]
‎    )
‎
‎    st.caption("Timeframe: 1 Minute")
‎
‎    st.divider()
‎
‎    st.warning(
‎        "OTC live data requires an authorized "
‎        "Pocket Broker feed or CSV export."
‎    )
‎
‎uploaded = None
‎
‎if source == "CSV Upload":
‎    uploaded = st.file_uploader(
‎        "Upload OHLC CSV",
‎        type=["csv"]
‎    )
‎
‎refresh = st.button(
‎    "🔄 Fetch / Analyze Market",
‎    type="primary",
‎    use_container_width=True
‎)
‎
‎if refresh:
‎
‎    try:
‎
‎        if category == "OTC Forex" and source == "Live API":
‎            raise ValueError(
‎                "Generic market API cannot reproduce "
‎                "Pocket Broker OTC prices. Use authorized CSV."
‎            )
‎
‎        if source == "Live API":
‎            raw = fetch_live(symbol)
‎            source_name = "Twelve Data"
‎
‎        else:
‎            if uploaded is None:
‎                raise ValueError("Upload CSV first.")
‎
‎            raw = pd.read_csv(uploaded)
‎            source_name = "CSV"
‎
‎        clean = clean_data(raw)
‎
‎        # For this model, data should be one-minute bars.
‎        if len(clean) < 350:
‎            raise ValueError(
‎                "At least 350 valid one-minute candles required."
‎            )
‎
‎        df = indicators(clean)
‎
‎        # Backtest
‎        test = backtest(df)
‎
‎        # Prediction
‎        direction, up, down = predict_next(df)
‎
‎        latest = df.iloc[-1]
‎
‎        # Reconcile older signals using new candles
‎        settle_signals(df, source_name, symbol)
‎
‎        # Save current prediction
‎        save_signal(
‎            source_name,
‎            symbol,
‎            latest.timestamp,
‎            direction,
‎            up if direction == "UP" else down,
‎            latest.close
‎        )
‎
‎        st.session_state["market_df"] = df
‎        st.session_state["market_symbol"] = symbol
‎        st.session_state["source_name"] = source_name
‎        st.session_state["direction"] = direction
‎        st.session_state["up"] = up
‎        st.session_state["down"] = down
‎        st.session_state["test"] = test
‎
‎    except Exception as e:
‎        st.error(str(e))
‎
‎# =====================================
‎# DISPLAY RESULTS
‎# =====================================
‎
‎if "market_df" in st.session_state:
‎
‎    df = st.session_state["market_df"]
‎
‎    direction = st.session_state["direction"]
‎    up = st.session_state["up"]
‎    down = st.session_state["down"]
‎    test = st.session_state["test"]
‎
‎    st.subheader(
‎        st.session_state["market_symbol"]
‎    )
‎
‎    c1, c2, c3 = st.columns(3)
‎
‎    c1.metric("UP Probability", f"{up*100:.1f}%")
‎    c2.metric("DOWN Probability", f"{down*100:.1f}%")
‎    c3.metric("Backtest Accuracy", f"{test['accuracy']*100:.1f}%")
‎
‎    if direction == "UP":
‎        st.success("AI SIGNAL: UP 🟢")
‎
‎    elif direction == "DOWN":
‎        st.error("AI SIGNAL: DOWN 🔴")
‎
‎    else:
‎        st.warning("AI SIGNAL: WAIT ⚪")
‎
‎    st.caption(
‎        "Backtest accuracy is historical, not a guarantee "
‎        "of future results. Probabilities are not calibrated."
‎    )
‎
‎    latest = df.iloc[-1]
‎
‎    a, b, c, d = st.columns(4)
‎
‎    a.metric("RSI", f"{latest.rsi:.2f}")
‎    b.metric("EMA Gap", f"{latest.ema_gap:.6f}")
‎    c.metric("MACD", f"{latest.macd:.6f}")
‎    d.metric("ATR", f"{latest.atr:.6f}")
‎
‎    # Chart
‎    st.subheader("Live / Historical Candle Chart")
‎
‎    fig = go.Figure()
‎
‎    fig.add_trace(go.Candlestick(
‎        x=df.timestamp,
‎        open=df.open,
‎        high=df.high,
‎        low=df.low,
‎        close=df.close,
‎        name="Candles"
‎    ))
‎
‎    fig.add_trace(go.Scatter(
‎        x=df.timestamp,
‎        y=df.close.ewm(span=9).mean(),
‎        name="EMA 9"
‎    ))
‎
‎    fig.add_trace(go.Scatter(
‎        x=df.timestamp,
‎        y=df.close.ewm(span=21).mean(),
‎        name="EMA 21"
‎    ))
‎
‎    fig.update_layout(
‎        template="plotly_dark",
‎        height=450,
‎        xaxis_rangeslider_visible=False
‎    )
‎
‎    st.plotly_chart(
‎        fig,
‎        use_container_width=True
‎    )
‎
‎    # Backtest report
‎    st.subheader("Backtest Report")
‎
‎    report = pd.DataFrame({
‎        "Actual": test["actual"],
‎        "Predicted": test["predictions"]
‎    })
‎
‎    report["Result"] = np.where(
‎        report["Actual"] == report["Predicted"],
‎        "Correct",
‎        "Incorrect"
‎    )
‎
‎    st.metric(
‎        "Tested Predictions",
‎        test["count"]
‎    )
‎
‎    st.dataframe(
‎        report.tail(50),
‎        use_container_width=True
‎    )
‎
‎    # Accuracy history
‎    st.subheader("Prediction History")
‎
‎    with db() as con:
‎
‎        history = pd.read_sql_query("""
‎            SELECT source, symbol, candle_time,
‎                   direction, probability, entry, outcome
‎            FROM signals
‎            ORDER BY id DESC
‎            LIMIT 200
‎        """, con)
‎
‎    if not history.empty:
‎
‎        st.dataframe(
‎            history,
‎            use_container_width=True
‎        )
‎
‎        settled = history[
‎            history["outcome"].isin(["WIN", "LOSS", "DRAW"])
‎        ]
‎
‎        if len(settled):
‎
‎            wins = (settled["outcome"] == "WIN").sum()
‎            losses = (settled["outcome"] == "LOSS").sum()
‎            draws = (settled["outcome"] == "DRAW").sum()
‎
‎            denominator = wins + losses
‎
‎            accuracy = (
‎                wins / denominator * 100
‎                if denominator else 0
‎            )
‎
‎            x, y, z = st.columns(3)
‎
‎            x.metric("Settled Wins", int(wins))
‎            y.metric("Settled Losses", int(losses))
‎            z.metric("Tracked Accuracy", f"{accuracy:.1f}%")
‎
‎    else:
‎        st.info("No historical predictions yet.")
‎
‎else:
‎
‎    st.info(
‎        "Choose a market, select a data source, "
‎        "and press Fetch / Analyze Market."
‎    )
‎
