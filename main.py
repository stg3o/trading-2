# ==============================
# P1: Imports, Setup, Configuration
# ==============================

from fastapi import FastAPI, Query, Request, Form, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import requests
import time
import hmac
import base64
import hashlib
import uuid
import pandas as pd
import json
import random
import datetime
import threading
import openai
from dotenv import load_dotenv
import logging
from sentiment import get_coin_sentiment
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import asyncio
import aiohttp
import tweepy
import textblob
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import websockets

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setup FastAPI app
app = FastAPI(title="Rich Quick")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
templates = Jinja2Templates(directory="templates")

KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY")
KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET")
KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://api.kucoin.com"

openai.api_key = OPENAI_API_KEY

bot_mode = {"enabled": False}
dry_run = True
trade_log = []
strategy_mode = "ema_rsi"

top_pairs = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT",
    "DOGE-USDT", "LINK-USDT", "XMR-USDT", "NEAR-USDT", "AVAX-USDT"
]

# Store active WebSocket connections
active_connections = []

# ==============================
# P2: KuCoin API Endpoints
# ==============================

@app.post("/strategy-toggle")
def toggle_strategy(strategy: str = Form(...)):
    global strategy_mode
    strategy_mode = strategy
    return RedirectResponse(url="/dashboard", status_code=303)

def generate_kucoin_signature(endpoint: str, method: str, body: str = ""):
    timestamp = str(int(time.time() * 1000))
    str_to_sign = timestamp + method + endpoint + body
    signature = base64.b64encode(
        hmac.new(KUCOIN_API_SECRET.encode(), str_to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    return {
        "KC-API-KEY": KUCOIN_API_KEY,
        "KC-API-SIGN": signature,
        "KC-API-TIMESTAMP": timestamp,
        "KC-API-PASSPHRASE": KUCOIN_API_PASSPHRASE,
        "KC-API-VERSION": "2",
        "Content-Type": "application/json"
    }

@app.get("/kucoin-balance")
async def kucoin_balance():
    try:
        endpoint = "/api/v1/accounts"
        url = BASE_URL + endpoint
        headers = generate_kucoin_signature(endpoint, "GET")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"KuCoin API request error: {str(e)}")
        return {"error": f"Failed to fetch balance: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in kucoin_balance: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}

class OrderRequest(BaseModel):
    symbol: str
    side: str
    type: str = "market"
    size: float
    price: float = None

def place_kucoin_order(data: OrderRequest):
    endpoint = "/api/v1/orders"
    url = BASE_URL + endpoint
    order_data = {
        "clientOid": str(uuid.uuid4()),
        "symbol": data.symbol,
        "side": data.side,
        "type": data.type,
        "size": str(data.size)
    }
    if data.type == "limit":
        order_data["price"] = str(data.price)
        order_data["timeInForce"] = "GTC"
    body = json.dumps(order_data)
    headers = generate_kucoin_signature(endpoint, "POST", body)
    response = requests.post(url, headers=headers, data=body)
    return response.json()

@app.post("/kucoin-place-order")
async def kucoin_place_order(order: OrderRequest):
    return place_kucoin_order(order)

# ==============================
# P3: Technical Indicators & Signal Generation
# ==============================

def get_kucoin_klines(symbol="BTC-USDT", interval="15min", limit=100):
    endpoint = f"/api/v1/market/candles?type={interval}&symbol={symbol}&limit={limit}"
    url = BASE_URL + endpoint
    response = requests.get(url)
    data = response.json()
    if data["code"] != "200000":
        return None
    df = pd.DataFrame(data["data"], columns=[
        "time", "open", "close", "high", "low", "volume", "turnover"
    ])
    df = df.sort_values("time")
    df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].astype(float)
    return df

def calculate_rsi(df, period=14):
    try:
        # Calculate price changes
        delta = df["close"].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses with Wilder's smoothing
        avg_gain = gain.ewm(com=period-1, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=1, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Add RSI trend
        df["RSI_Trend"] = np.where(df["RSI"] > 50, 1, -1)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        raise

def calculate_atr(df, period=14):
    df["H-L"] = df["high"] - df["low"]
    df["H-PC"] = abs(df["high"] - df["close"].shift())
    df["L-PC"] = abs(df["low"] - df["close"].shift())
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=period).mean()
    return df

def calculate_ema(df, short_period=12, long_period=26):
    try:
        # Calculate EMAs
        df["EMA_Short"] = df["close"].ewm(span=short_period, min_periods=1, adjust=False).mean()
        df["EMA_Long"] = df["close"].ewm(span=long_period, min_periods=1, adjust=False).mean()
        
        # Calculate MACD
        df["MACD"] = df["EMA_Short"] - df["EMA_Long"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, min_periods=1, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Add trend direction
        df["EMA_Trend"] = np.where(df["MACD"] > df["MACD_Signal"], 1, -1)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        raise

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    try:
        # Calculate MACD components
        df["EMA_Fast"] = df["close"].ewm(span=fast_period, min_periods=1, adjust=False).mean()
        df["EMA_Slow"] = df["close"].ewm(span=slow_period, min_periods=1, adjust=False).mean()
        df["MACD"] = df["EMA_Fast"] - df["EMA_Slow"]
        df["Signal"] = df["MACD"].ewm(span=signal_period, min_periods=1, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["Signal"]
        
        # Add MACD trend and momentum
        df["MACD_Trend"] = np.where(df["MACD"] > df["Signal"], 1, -1)
        df["MACD_Momentum"] = df["MACD_Hist"].diff()
        
        # Calculate volume trend
        df["Volume_MA"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["Volume_Trend"] = np.where(df["volume"] > df["Volume_MA"], 1, -1)
        
        logger.debug(f"MACD calculation successful. Sample data:\n{df[['MACD', 'Signal', 'MACD_Hist', 'MACD_Trend']].tail()}")
        return df
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        raise

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    try:
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (std * std_dev)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        return df
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        raise

def calculate_volume_profile(df):
    """Calculate Volume Profile indicators"""
    try:
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
        df['Price_Volume'] = df['close'] * df['volume']
        return df
    except Exception as e:
        logger.error(f"Error calculating Volume Profile: {str(e)}")
        raise

def calculate_advanced_macd(df, fast=12, slow=26, signal=9):
    """Enhanced MACD with additional signals"""
    try:
        # Standard MACD calculation
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Additional MACD indicators
        df['MACD_Hist_Change'] = df['MACD_Hist'].diff()
        df['MACD_Divergence'] = (df['MACD'].diff() > 0) & (df['close'].diff() < 0)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating Advanced MACD: {str(e)}")
        raise

def generate_signal(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df = calculate_rsi(df)

    if strategy_mode == "ema_only":
        last = df.iloc[-1]
        return "buy" if last["close"] > last["EMA20"] else "sell"

    elif strategy_mode == "rsi_only":
        last = df.iloc[-1]
        rsi = last["RSI"]
        return "buy" if rsi < 30 else "sell" if rsi > 70 else "hold"

    elif strategy_mode == "keltner":
        df = calculate_atr(df, period=14)
        df["Upper"] = df["EMA20"] + 2 * df["ATR"]
        df["Lower"] = df["EMA20"] - 2 * df["ATR"]
        last = df.iloc[-1]

        if last["close"] > last["Upper"]:
            return "buy"
        elif last["close"] < last["Lower"]:
            return "sell"
        else:
            return "hold"

    else:  # Default: EMA + RSI
        last = df.iloc[-1]
        close = last["close"]
        ema = last["EMA20"]
        rsi = last["RSI"]

        if close > ema and rsi < 30:
            return "buy"
        elif close < ema and rsi > 70:
            return "sell"
        else:
            return "hold"

# ==============================
# P4: AI Signal Advice
# ==============================

def get_ai_signal_advice(symbol: str, price: float, ema: float, rsi: float) -> str:
    """
    Generate a trade suggestion using OpenAI based on price, EMA20, and RSI.
    Returns a concise recommendation (e.g., "Buy because...").
    """
    prompt = f"""
    You are a crypto trading expert. Given this data for {symbol}:
    - Current Price: {price}
    - EMA20: {ema}
    - RSI: {rsi}

    Suggest a trade action (buy, sell, or hold) and explain it in 1 sentence.
    """
    try:
        logging.info(f"Sending prompt to AI: {prompt}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"Error in AI advice: {e}")
        return f"⚠️ AI advice unavailable: {e}"

@app.get("/ai-signal")
def ai_signal(symbol: str = Query("BTC-USDT"), interval: str = Query("15min")):
    """
    Endpoint to get both technical and AI-generated trading signals.
    Returns signal + price, EMA20, RSI, and AI commentary.
    """
    df = get_kucoin_klines(symbol, interval)
    if df is None or df.empty:
        return {"signal": "error", "message": "Could not load market data."}

    signal = generate_signal(df)
    latest_price = df["close"].iloc[-1]
    ema = df["EMA20"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    ai_advice = get_ai_signal_advice(symbol, latest_price, ema, rsi)

    return {
        "symbol": symbol,
        "signal": signal,
        "latest_price": latest_price,
        "ema": ema,
        "rsi": rsi,
        "ai_advice": ai_advice
    }


# ==============================
# P5: Routes for Dashboard and Wallet Views
# ==============================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, symbol: str = "BTC-USDT"):
    """
    Renders the dashboard page with market indicators, bot status, and AI signal.
    """
    df = get_kucoin_klines(symbol)
    signal = generate_signal(df) if df is not None else "error"
    price = float(df["close"].iloc[-1]) if df is not None else 0.0
    ema = float(df["EMA20"].iloc[-1]) if df is not None else 0.0
    rsi = float(df["RSI"].iloc[-1]) if df is not None else 50.0

    rsi_series = df["RSI"].tolist()[-30:] if df is not None and "RSI" in df else []
    volume_series = df["volume"].tolist()[-30:] if df is not None and "volume" in df else []

    # Get USDT balance from API
    balance_response = requests.get("http://127.0.0.1:8000/kucoin-balance").json()
    usdt_balance = next(
        (item for item in balance_response.get("data", []) if item["currency"] == "USDT"),
        {"available": "0.00"}
    )

    # Fetch AI advice
    ai_advice = get_ai_signal_advice(symbol, price, ema, rsi) or "AI advice unavailable"

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "signal": signal,
        "bot_status": bot_mode["enabled"],
        "price": round(price, 2),
        "ema": round(ema, 2),
        "rsi": round(rsi, 2),
        "dry_run": dry_run,
        "usdt_balance": usdt_balance.get("available", "0.00"),
        "symbol": symbol,
        "trade_log": trade_log[-10:],
        "strategy_mode": strategy_mode,
        "ai_advice": str(ai_advice),
        "top_pairs": top_pairs,
        "rsi_series": rsi_series,
        "volume_series": volume_series
    })


@app.get("/wallet", response_class=HTMLResponse)
def wallet(request: Request):
    """
    Renders the wallet page showing token balances and current market value.
    """
    try:
        response = requests.get("http://127.0.0.1:8000/kucoin-balance")
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            logger.error(f"Error fetching balances: {data['error']}")
            return templates.TemplateResponse("wallet.html", {
                "request": request,
                "error": data["error"],
                "balances": [],
                "total_value": 0,
                "prices": {}
            })

        balances = data.get("data", [])

        # Filter out tokens with 0 balance
        non_zero = [b for b in balances if float(b.get("balance", 0)) > 0]

        total_value = 0.0
        prices = {}

        for b in non_zero:
            symbol = f"{b['currency']}-USDT"
            try:
                res = requests.get(f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol}")
                res.raise_for_status()
                price_data = res.json()

                if price_data.get("code") == "200000" and price_data.get("data", {}).get("price"):
                    price = float(price_data["data"]["price"])
                    value = float(b["balance"]) * price
                    total_value += value
                    prices[b["currency"]] = {
                        "price": round(price, 4),
                        "value": round(value, 2)
                    }
                else:
                    logger.warning(f"Could not get price for {symbol}: {price_data}")
                    prices[b["currency"]] = {"price": "-", "value": "-"}
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {str(e)}")
                prices[b["currency"]] = {"price": "-", "value": "-"}

        return templates.TemplateResponse("wallet.html", {
            "request": request,
            "balances": non_zero,
            "total_value": round(total_value, 2),
            "prices": prices
        })

    except Exception as e:
        logger.error(f"Error in wallet route: {str(e)}")
        return templates.TemplateResponse("wallet.html", {
            "request": request,
            "error": f"Failed to load wallet: {str(e)}",
            "balances": [],
            "total_value": 0,
            "prices": {}
        })

# ==============================
# P6: Sentiment Analysis
# ==============================

@app.get("/sentiment")
def get_sentiment(symbol: str = "BTC"):
    result = get_coin_sentiment(symbol)

    if result["score"] is None:
        return JSONResponse(content={"error": result["summary"]}, status_code=500)

    return {
        "symbol": symbol,
        "sentiment_score": result["score"],
        "sentiment_label": result["label"],
        "summary": result["summary"]
    }

# ==============================
# P7: Bot Loop + Backtesting
# ==============================

def bot_loop():
    while True:
        time.sleep(60)
        if not bot_mode["enabled"]:
            continue
        df = get_kucoin_klines()
        if df is None or df.empty:
            continue
        signal = generate_signal(df)
        log_entry = {"signal": signal, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        if signal in ["buy", "sell"]:
            order = OrderRequest(symbol="BTC-USDT", side=signal, size=0.001)
            if dry_run:
                log_entry["action"] = f"DRY RUN: Would place {signal} order"
            else:
                res = place_kucoin_order(order)
                log_entry["action"] = f"LIVE ORDER: {res}"
        else:
            log_entry["action"] = "No trade taken"
        trade_log.append(log_entry)


# 🔁 Shared backtest logic
def get_price_data(symbol, interval="1day", limit=500):
    try:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}&limit={limit}"
        logger.debug(f"Fetching data from: {url}")
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        
        if data["code"] != "200000":
            logger.error(f"KuCoin API error: {data}")
            return None
            
        if not data.get("data"):
            logger.error("No data received from KuCoin")
            return None
            
        processed_data = []
        now = datetime.datetime.now()
        
        for candle in data["data"]:  # Remove reversed() to get correct order
            try:
                # Convert timestamp from milliseconds to seconds if needed
                timestamp = int(candle[0])
                if timestamp > 1e12:  # If timestamp is in milliseconds
                    timestamp = timestamp / 1000
                candle_time = datetime.datetime.fromtimestamp(timestamp)
                
                # Skip future dates
                if candle_time > now:
                    continue
                
                processed_data.append({
                    "timestamp": candle_time,
                    "open": float(candle[1]),
                    "close": float(candle[2]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "volume": float(candle[5])
                })
            except (IndexError, ValueError) as e:
                logger.error(f"Error processing candle data: {e}")
                logger.error(f"Problematic candle data: {candle}")
                continue
        
        # Sort by timestamp
        processed_data.sort(key=lambda x: x["timestamp"])
        
        logger.debug(f"Processed {len(processed_data)} candles")
        if processed_data:
            logger.debug(f"Sample timestamp range: {processed_data[0]['timestamp']} to {processed_data[-1]['timestamp']}")
        return processed_data
    except Exception as e:
        logger.error(f"Error fetching price data: {str(e)}")
        return None

def calculate_keltner(df, ema_period=20, atr_multiplier=2.5):
    try:
        # Calculate Typical Price with more weight on close price
        df["TP"] = (df["high"] + df["low"] + 2 * df["close"]) / 4
        
        # Calculate EMA of Typical Price
        df["EMA"] = df["TP"].ewm(span=ema_period, min_periods=1, adjust=False).mean()
        
        # Calculate True Range
        df["H-L"] = df["high"] - df["low"]
        df["H-PC"] = abs(df["high"] - df["close"].shift(1))
        df["L-PC"] = abs(df["low"] - df["close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        
        # Calculate ATR with Wilder's smoothing
        df["ATR"] = df["TR"].ewm(span=ema_period, min_periods=1, adjust=False).mean()
        
        # Calculate Keltner Channels
        df["KC_Middle"] = df["EMA"]
        df["KC_Upper"] = df["KC_Middle"] + (df["ATR"] * atr_multiplier)
        df["KC_Lower"] = df["KC_Middle"] - (df["ATR"] * atr_multiplier)
        
        # Add trend direction
        df["Trend"] = np.where(df["close"] > df["KC_Middle"], 1, 
                              np.where(df["close"] < df["KC_Middle"], -1, 0))
        
        logger.debug(f"Keltner calculation successful. Sample data:\n{df[['close', 'KC_Upper', 'KC_Lower', 'Trend']].tail()}")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_keltner: {str(e)}")
        raise

def run_backtest(symbol, strategy, start=None, end=None, interval="1day"):
    logger.debug(f"Starting backtest with params - Symbol: {symbol}, Strategy: {strategy}, Start: {start}, End: {end}, Interval: {interval}")
    
    try:
        # Get price data
        price_data = get_price_data(symbol, interval=interval, limit=500)
        
        if not price_data:
            logger.error("No price data received")
            return {"error": "Failed to fetch price data", "total_return": 0, "win_rate": 0, "max_drawdown": 0, "trades": []}

        # Create DataFrame and prepare data
        df = pd.DataFrame(price_data)
        df = prepare_data(df, start, end)
        
        if len(df) < 30:  # Need more data for reliable signals
            return {"error": "Not enough data for analysis", "total_return": 0, "win_rate": 0, "max_drawdown": 0, "trades": []}

        # Calculate strategy indicators
        df = calculate_strategy_indicators(df, strategy)
        
        # Generate signals
        trades = generate_signals(df, strategy)
        
        # Calculate metrics
        result = calculate_metrics(trades)
        
        logger.debug(f"Backtest completed with {len(trades)} trades")
        return result
        
    except Exception as e:
        logger.error(f"Error in run_backtest: {str(e)}", exc_info=True)
        return {"error": str(e), "total_return": 0, "win_rate": 0, "max_drawdown": 0, "trades": []}

def generate_signals(df, strategy):
    trades = []
    entry_price = None
    entry_time = None
    signals_generated = 0
    
    for i in range(1, len(df)):
        signal = None
        
        if strategy == "macd":
            # MACD Strategy with multiple confirmations
            current_close = df["close"].iloc[i]
            current_volume = df["volume"].iloc[i]
            
            # Buy conditions:
            # 1. MACD crosses above Signal line
            # 2. MACD histogram is increasing (momentum)
            # 3. Volume is above average
            # 4. Price is trending up
            if (df["MACD"].iloc[i] > df["Signal"].iloc[i] and 
                df["MACD"].iloc[i-1] <= df["Signal"].iloc[i-1] and 
                df["MACD_Momentum"].iloc[i] > 0 and 
                df["Volume_Trend"].iloc[i] == 1 and 
                current_close > df["close"].iloc[i-1]):
                
                signal = "BUY"
                
            # Sell conditions:
            # 1. MACD crosses below Signal line
            # 2. MACD histogram is decreasing
            # 3. Volume confirmation
            # 4. Price is trending down
            elif (df["MACD"].iloc[i] < df["Signal"].iloc[i] and 
                  df["MACD"].iloc[i-1] >= df["Signal"].iloc[i-1] and 
                  df["MACD_Momentum"].iloc[i] < 0 and 
                  df["Volume_Trend"].iloc[i] == 1 and 
                  current_close < df["close"].iloc[i-1]):
                
                signal = "SELL"
        
        elif strategy == "keltner":
            # Keltner Channel strategy with trend confirmation
            if (df["close"].iloc[i] > df["KC_Upper"].iloc[i] and 
                df["Trend"].iloc[i] == 1 and 
                df["volume"].iloc[i] > df["volume"].iloc[i-1]):
                signal = "BUY"
            elif (df["close"].iloc[i] < df["KC_Lower"].iloc[i] and 
                  df["Trend"].iloc[i] == -1 and 
                  df["volume"].iloc[i] > df["volume"].iloc[i-1]):
                signal = "SELL"
                
        elif strategy == "ema_rsi":
            # Combined EMA and RSI strategy with trend confirmation
            if (df["EMA_Trend"].iloc[i] == 1 and 
                df["RSI"].iloc[i] < 40 and 
                df["RSI"].iloc[i] > df["RSI"].iloc[i-1]):
                signal = "BUY"
            elif (df["EMA_Trend"].iloc[i] == -1 and 
                  df["RSI"].iloc[i] > 60 and 
                  df["RSI"].iloc[i] < df["RSI"].iloc[i-1]):
                signal = "SELL"
                
        elif strategy == "ema_only":
            # Enhanced EMA strategy with MACD confirmation
            if (df["EMA_Trend"].iloc[i] == 1 and 
                df["MACD_Hist"].iloc[i] > 0 and 
                df["MACD_Hist"].iloc[i-1] <= 0):
                signal = "BUY"
            elif (df["EMA_Trend"].iloc[i] == -1 and 
                  df["MACD_Hist"].iloc[i] < 0 and 
                  df["MACD_Hist"].iloc[i-1] >= 0):
                signal = "SELL"
                
        elif strategy == "rsi_only":
            # Enhanced RSI strategy with trend confirmation
            if (df["RSI"].iloc[i] < 30 and 
                df["RSI"].iloc[i] > df["RSI"].iloc[i-1] and 
                df["close"].iloc[i] > df["close"].iloc[i-1]):
                signal = "BUY"
            elif (df["RSI"].iloc[i] > 70 and 
                  df["RSI"].iloc[i] < df["RSI"].iloc[i-1] and 
                  df["close"].iloc[i] < df["close"].iloc[i-1]):
                signal = "SELL"
        
        if signal:
            signals_generated += 1
            
            if signal == "BUY" and entry_price is None:
                entry_price = df["close"].iloc[i]
                entry_time = df["timestamp"].iloc[i]
                logger.debug(f"Opening trade at {entry_time} with price {entry_price}")
            elif signal == "SELL" and entry_price is not None:
                exit_price = df["close"].iloc[i]
            pnl = (exit_price - entry_price) / entry_price * 100
                
                # Add risk management
            if pnl < -2:  # 2% stop loss
                logger.debug(f"Stop loss triggered at {pnl}%")
                
            trades.append({
                    "entry": entry_time.strftime("%Y-%m-%d %H:%M"),
                    "exit": df["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M"),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "signal_type": "MACD Cross" if strategy == "macd" else "Other"
                })
                
            logger.debug(f"Closing trade at {df['timestamp'].iloc[i]} with PnL {pnl}%")
            entry_price = None
            entry_time = None

    return trades

def calculate_strategy_indicators(df, strategy):
    if strategy == "macd":
        df = calculate_macd(df)
    elif strategy == "keltner":
        df = calculate_keltner(df)
    elif strategy in ["ema_rsi", "rsi_only"]:
        df = calculate_rsi(df)
    if strategy in ["ema_rsi", "ema_only"]:
        df = calculate_ema(df)
    return df

def calculate_metrics(trades):
    if not trades:
        return {
            "total_return": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "trades": [],
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0
        }
    
    # Calculate basic metrics
    total_return = round(sum(t["pnl"] for t in trades), 2)
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]
    win_rate = round(100 * len(winning_trades) / len(trades), 2)
    
    # Calculate average win and loss
    avg_win = round(sum(t["pnl"] for t in winning_trades) / len(winning_trades), 2) if winning_trades else 0
    avg_loss = round(sum(t["pnl"] for t in losing_trades) / len(losing_trades), 2) if losing_trades else 0
    
    # Calculate profit factor
    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss != 0 else float('inf')
    
    # Calculate max drawdown
    cumulative_returns = [0]
    for trade in trades:
        cumulative_returns.append(cumulative_returns[-1] + trade["pnl"])
    max_drawdown = round(min(0, min(cumulative_returns)), 2)

    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_trades": len(trades)
    }

# JSON API endpoint
@app.get("/api/backtest")
async def backtest(symbol: str = "BTC-USDT", strategy: str = "ema_rsi"):
    result = run_backtest(symbol, strategy)
    return JSONResponse(result)


@app.get("/strategies", response_class=HTMLResponse)
async def strategies(
    request: Request, 
    symbol: str = "BTC-USDT", 
    strategy: str = "ema_rsi", 
    start: str = None, 
    end: str = None, 
    interval: str = "1day"
):
    now = datetime.datetime.now()
    
    # Force dates to be in the past
    try:
        if end:
            end_date = pd.to_datetime(end)
            if end_date > now:
                end = now.strftime("%Y-%m-%d")
        else:
            end = now.strftime("%Y-%m-%d")
            
        if start:
            start_date = pd.to_datetime(start)
            if start_date > now:
                start = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            start = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            
        # Ensure start is not after end
        if pd.to_datetime(start) > pd.to_datetime(end):
            start = (pd.to_datetime(end) - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            
    except Exception as e:
        logger.error(f"Date parsing error: {e}")
        end = now.strftime("%Y-%m-%d")
        start = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    logger.debug(f"Using date range: {start} to {end}")
    
    try:
        result = run_backtest(symbol, strategy, start, end, interval)
        
        if not result.get("trades"):
            logger.warning("No trades generated in backtest")
            
        return templates.TemplateResponse(
            "strategies.html", 
            {
                "request": request,
                "symbol": symbol,
                "strategy": strategy,
                "start": start,  # Send back the corrected dates
                "end": end,
                "result": result,
                "interval": interval,
                "now": now,
                "timedelta": datetime.timedelta,
                "message": "No trades found in the selected date range" if not result.get("trades") else None
            }
        )
    except Exception as e:
        logger.error(f"Error in strategies endpoint: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "strategies.html",
            {
        "request": request,
        "symbol": symbol,
        "strategy": strategy,
        "start": start,
        "end": end,
                "interval": interval,
                "error": str(e),
                "now": now,
                "timedelta": datetime.timedelta
            }
        )

# Start bot thread
threading.Thread(target=bot_loop, daemon=True).start()

def run_advanced_backtest(symbol, strategy, start_date=None, end_date=None, initial_capital=1000):
    """Enhanced backtesting with detailed performance metrics"""
    try:
        # Get historical data
        df = get_price_data(symbol)
        if df is None:
            return {"error": "Failed to fetch price data"}
            
        # Apply date filters if provided
        if start_date and end_date:
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
        # Generate trading signals
        trades = generate_advanced_signals(df, strategy)
        
        if not trades:
            return {
                "error": "No trades generated",
                "message": "The strategy did not generate any signals in the given period"
            }
            
        # Calculate detailed metrics
        results = calculate_advanced_metrics(trades, df, initial_capital)
        
        # Add trade distribution analysis
        results['trade_distribution'] = analyze_trade_distribution(trades)
        
        # Add market conditions analysis
        results['market_conditions'] = analyze_market_conditions(df)
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        return {"error": str(e)}

def calculate_advanced_metrics(trades, df, initial_capital):
    """Calculate comprehensive trading metrics"""
    try:
        if not trades:
            return {}
            
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        # Performance metrics
        total_return = sum(t['pnl'] for t in trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Risk metrics
        max_drawdown = calculate_max_drawdown(trades)
        sharpe_ratio = calculate_sharpe_ratio(trades)
        sortino_ratio = calculate_sortino_ratio(trades)
        
        # Advanced metrics
        profit_factor = abs(sum(t['pnl'] for t in winning_trades)) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        recovery_factor = total_return / max_drawdown if max_drawdown != 0 else float('inf')
        
        return {
            'summary': {
                'total_return': round(total_return, 2),
                'total_trades': total_trades,
                'win_rate': round(win_rate * 100, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown': round(max_drawdown, 2),
                'recovery_factor': round(recovery_factor, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'sortino_ratio': round(sortino_ratio, 2)
            },
            'trade_metrics': {
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'largest_win': round(max(t['pnl'] for t in trades), 2),
                'largest_loss': round(min(t['pnl'] for t in trades), 2),
                'avg_hold_time': calculate_avg_hold_time(trades)
            },
            'equity_curve': calculate_equity_curve(trades, initial_capital)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {}

def calculate_position_size(account_balance, risk_per_trade=0.02):
    """Calculate position size based on account risk"""
    try:
        max_risk_amount = account_balance * risk_per_trade
        return max_risk_amount
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return 0

def calculate_stop_loss(entry_price, risk_amount, position_size):
    """Calculate stop loss price based on risk amount"""
    try:
        risk_per_unit = risk_amount / position_size
        stop_loss = entry_price - risk_per_unit
        return max(stop_loss, 0)  # Ensure stop loss is not negative
    except Exception as e:
        logger.error(f"Error calculating stop loss: {str(e)}")
        return entry_price * 0.98  # Default to 2% below entry

def adjust_position_for_volatility(base_position_size, symbol, lookback_period=14):
    """Adjust position size based on market volatility"""
    try:
        df = get_kucoin_klines(symbol, limit=lookback_period)
        if df is None:
            return base_position_size
            
        # Calculate ATR
        df = calculate_atr(df)
        current_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].mean()
        
        # Adjust position size inversely to volatility
        volatility_ratio = avg_atr / current_atr if current_atr > 0 else 1
        adjusted_size = base_position_size * volatility_ratio
        
        # Cap the adjustment
        max_adjustment = base_position_size * 1.5
        min_adjustment = base_position_size * 0.5
        
        return min(max(adjusted_size, min_adjustment), max_adjustment)
    except Exception as e:
        logger.error(f"Error adjusting position size: {str(e)}")
        return base_position_size

@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Receive any client messages
            data = await websocket.receive_text()
            # Process client messages if needed
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(websocket)

async def broadcast_market_data():
    """Broadcast market data to all connected clients"""
    while True:
        try:
            for symbol in top_pairs:
                data = await get_realtime_market_data(symbol)
                for connection in active_connections:
                    await connection.send_json(data)
            await asyncio.sleep(1)  # Update every second
        except Exception as e:
            logger.error(f"Broadcast error: {str(e)}")
            await asyncio.sleep(1)

async def get_realtime_market_data(symbol):
    """Fetch real-time market data from KuCoin"""
    try:
        endpoint = f"/api/v1/market/orderbook/level1?symbol={symbol}"
        url = BASE_URL + endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                if data["code"] == "200000":
                    return {
                        "symbol": symbol,
                        "price": data["data"]["price"],
                        "time": data["data"]["time"],
                        "sequence": data["data"]["sequence"],
                        "size": data["data"]["size"]
                    }
    except Exception as e:
        logger.error(f"Error fetching real-time data: {str(e)}")
        return None

@app.get("/api/initial-data")
async def get_initial_data():
    """Get initial data for dashboard"""
    try:
        # Get portfolio data
        portfolio = await get_portfolio_summary()
        
        # Get market data for top pairs
        market_data = {}
        signals = []
        
        for symbol in top_pairs:
            # Get candle data
            candles = get_kucoin_klines(symbol)
            if candles is not None:
                df = prepare_technical_indicators(candles)
                signal = generate_signal(df)
                signals.append({
                    "symbol": symbol,
                    "type": signal,
                    "indicator": get_signal_indicators(df)
                })
                
                market_data[symbol] = {
                    "price": float(df["close"].iloc[-1]),
                    "change_24h": calculate_24h_change(df),
                    "volume_24h": float(df["volume"].sum())
                }
        
        return {
            "portfolio": portfolio,
            "market_data": market_data,
            "signals": signals,
            "active_trades": get_active_trades()
        }
    except Exception as e:
        logger.error(f"Error getting initial data: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to load initial data"},
            status_code=500
        )

async def get_portfolio_summary():
    """Get detailed portfolio summary"""
    try:
        # Get account balances
        balances = await get_account_balances()
        
        # Calculate total value and 24h change
        total_value = 0
        total_value_24h_ago = 0
        
        for balance in balances:
            current_price = await get_current_price(f"{balance['currency']}-USDT")
            if current_price:
                value = float(balance['balance']) * current_price
                total_value += value
                
                # Get 24h ago price
                price_24h_ago = await get_historical_price(
                    f"{balance['currency']}-USDT",
                    (datetime.now() - timedelta(days=1)).timestamp()
                )
                if price_24h_ago:
                    total_value_24h_ago += float(balance['balance']) * price_24h_ago
        
        return {
            "total_value": total_value,
            "change_24h": ((total_value - total_value_24h_ago) / total_value_24h_ago * 100) 
                         if total_value_24h_ago > 0 else 0,
            "balances": balances
        }
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {str(e)}")
        return None

class MarketSentimentAnalyzer:
    def __init__(self):
        self.news_sources = [
            'https://api.cryptopanic.com/v1/posts/',
            'https://min-api.cryptocompare.com/data/v2/news/',
            # Add more news sources
        ]
        # Initialize Twitter API if credentials are available
        if os.getenv('TWITTER_API_KEY'):
            auth = tweepy.OAuthHandler(
                os.getenv('TWITTER_API_KEY'),
                os.getenv('TWITTER_API_SECRET')
            )
            self.twitter_api = tweepy.API(auth)
        else:
            self.twitter_api = None

    async def get_comprehensive_sentiment(self, symbol):
        """Get sentiment from multiple sources"""
        try:
            tasks = [
                self.get_news_sentiment(symbol),
                self.get_social_sentiment(symbol),
                self.get_market_metrics(symbol),
                self.get_fear_greed_index()
            ]
            results = await asyncio.gather(*tasks)
            
            # Combine and weight different sentiment sources
            news_sentiment, social_sentiment, market_metrics, fear_greed = results
            
            weighted_sentiment = {
                'overall_score': self._calculate_weighted_score(results),
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'market_metrics': market_metrics,
                'fear_greed_index': fear_greed,
                'timestamp': datetime.now().isoformat()
            }
            
            return weighted_sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

    async def get_news_sentiment(self, symbol):
        """Analyze sentiment from news articles"""
        async with aiohttp.ClientSession() as session:
            sentiments = []
            for source in self.news_sources:
                try:
                    async with session.get(f"{source}?coin={symbol}") as response:
                        news = await response.json()
                        for article in news['results']:
                            blob = textblob.TextBlob(article['title'] + " " + article['text'])
                            sentiments.append(blob.sentiment.polarity)
                except Exception as e:
                    logger.error(f"Error fetching news from {source}: {str(e)}")
                    continue
            
            return {
                'average': sum(sentiments) / len(sentiments) if sentiments else 0,
                'count': len(sentiments),
                'latest': sentiments[-3:] if sentiments else []
            }

    async def get_social_sentiment(self, symbol):
        """Analyze social media sentiment"""
        if not self.twitter_api:
            return None
            
        try:
            tweets = self.twitter_api.search_tweets(
                q=f"#{symbol} OR ${symbol}",
                lang="en",
                count=100
            )
            
            sentiments = []
            for tweet in tweets:
                blob = textblob.TextBlob(tweet.text)
                sentiments.append(blob.sentiment.polarity)
                
            return {
                'average': sum(sentiments) / len(sentiments) if sentiments else 0,
                'volume': len(sentiments),
                'trend': self._calculate_sentiment_trend(sentiments)
            }
        except Exception as e:
            logger.error(f"Error in social sentiment: {str(e)}")
            return None

    async def get_market_metrics(self, symbol):
        """Get market-based sentiment indicators"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE_URL}/api/v1/market/stats?symbol={symbol}-USDT") as response:
                    data = await response.json()
                    
                    return {
                        'volume_change': data['volumeChange'],
                        'price_change': data['priceChange'],
                        'bid_ask_spread': data['spread'],
                        'volatility': data['volatility']
                    }
        except Exception as e:
            logger.error(f"Error in market metrics: {str(e)}")
            return None

    def _calculate_weighted_score(self, results):
        """Calculate weighted sentiment score"""
        weights = {
            'news': 0.3,
            'social': 0.2,
            'market': 0.3,
            'fear_greed': 0.2
        }
        
        scores = {}
        if results[0]: scores['news'] = results[0]['average']
        if results[1]: scores['social'] = results[1]['average']
        if results[2]: scores['market'] = self._normalize_market_metrics(results[2])
        if results[3]: scores['fear_greed'] = results[3] / 100
        
        weighted_score = 0
        total_weight = 0
        
        for key, score in scores.items():
            weighted_score += score * weights[key]
            total_weight += weights[key]
            
        return weighted_score / total_weight if total_weight > 0 else 0

class PortfolioManager:
    def __init__(self, target_allocation=None):
        self.target_allocation = target_allocation or {
            'BTC': 0.4,
            'ETH': 0.3,
            'SOL': 0.15,
            'USDT': 0.15
        }
        self.rebalance_threshold = 0.05  # 5% deviation threshold

    async def get_current_allocation(self):
        """Get current portfolio allocation"""
        try:
            balances = await get_account_balances()
            total_value = 0
            allocation = {}
            
            # Calculate total portfolio value
            for balance in balances:
                if float(balance['balance']) > 0:
                    price = await get_current_price(f"{balance['currency']}-USDT")
                    value = float(balance['balance']) * price
                    total_value += value
                    allocation[balance['currency']] = {
                        'amount': float(balance['balance']),
                        'value': value
                    }
            
            # Calculate percentages
            for currency in allocation:
                allocation[currency]['percentage'] = allocation[currency]['value'] / total_value
                
            return allocation
            
        except Exception as e:
            logger.error(f"Error getting allocation: {str(e)}")
            return None

    async def check_rebalance_needed(self):
        """Check if portfolio needs rebalancing"""
        current = await self.get_current_allocation()
        if not current:
            return False
            
        for currency, target in self.target_allocation.items():
            if currency in current:
                deviation = abs(current[currency]['percentage'] - target)
                if deviation > self.rebalance_threshold:
                    return True
        return False

    async def generate_rebalance_orders(self):
        """Generate orders to rebalance portfolio"""
        try:
            current = await self.get_current_allocation()
            if not current:
                return []
                
            orders = []
            total_value = sum(coin['value'] for coin in current.values())
            
            for currency, target in self.target_allocation.items():
                target_value = total_value * target
                current_value = current.get(currency, {}).get('value', 0)
                
                if abs(current_value - target_value) > (target_value * self.rebalance_threshold):
                    if current_value < target_value:
                        # Need to buy
                        amount = (target_value - current_value) / await get_current_price(f"{currency}-USDT")
                        orders.append({
                            'symbol': f"{currency}-USDT",
                            'side': 'buy',
                            'amount': amount
                        })
                    else:
                        # Need to sell
                        amount = (current_value - target_value) / await get_current_price(f"{currency}-USDT")
                        orders.append({
                            'symbol': f"{currency}-USDT",
                            'side': 'sell',
                            'amount': amount
                        })
            
            return orders
            
        except Exception as e:
            logger.error(f"Error generating rebalance orders: {str(e)}")
            return []

    async def execute_rebalance(self, dry_run=True):
        """Execute portfolio rebalancing"""
        try:
            orders = await self.generate_rebalance_orders()
            results = []
            
            for order in orders:
                if dry_run:
                    results.append({
                        'order': order,
                        'status': 'simulated',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    # Execute actual order
                    result = await place_kucoin_order(OrderRequest(
                        symbol=order['symbol'],
                        side=order['side'],
                        size=order['amount']
                    ))
                    results.append({
                        'order': order,
                        'status': 'executed',
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing rebalance: {str(e)}")
            return []

class OrderManager:
    def __init__(self):
        self.active_orders = {}

    async def place_advanced_order(self, order_type, params):
        """Place advanced order types"""
        try:
            if order_type == "OCO":  # One-Cancels-Other
                return await self.place_oco_order(params)
            elif order_type == "trailing_stop":
                return await self.place_trailing_stop(params)
            elif order_type == "iceberg":
                return await self.place_iceberg_order(params)
            elif order_type == "twap":  # Time-Weighted Average Price
                return await self.start_twap_execution(params)
            else:
                raise ValueError(f"Unknown order type: {order_type}")
                
        except Exception as e:
            logger.error(f"Error placing advanced order: {str(e)}")
            return None

    async def place_oco_order(self, params):
        """Place OCO (One-Cancels-Other) order"""
        try:
            # Place limit order
            limit_order = await place_kucoin_order(OrderRequest(
                symbol=params['symbol'],
                side=params['side'],
                type='limit',
                size=params['size'],
                price=params['limit_price']
            ))
            
            # Place stop order
            stop_order = await place_kucoin_order(OrderRequest(
                symbol=params['symbol'],
                side=params['side'],
                type='stop',
                size=params['size'],
                price=params['stop_price'],
                stopPrice=params['stop_trigger']
            ))
            
            # Store OCO pair
            oco_id = str(uuid.uuid4())
            self.active_orders[oco_id] = {
                'type': 'OCO',
                'orders': [limit_order, stop_order],
                'params': params
            }
            
            return {
                'oco_id': oco_id,
                'limit_order': limit_order,
                'stop_order': stop_order
            }
            
        except Exception as e:
            logger.error(f"Error placing OCO order: {str(e)}")
            return None

    async def place_trailing_stop(self, params):
        """Place trailing stop order"""
        try:
            current_price = await get_current_price(params['symbol'])
            trail_amount = params['trail_amount']
            
            if params['trail_type'] == 'percentage':
                trail_value = current_price * (trail_amount / 100)
            else:
                trail_value = trail_amount
                
            stop_price = current_price - trail_value if params['side'] == 'sell' else current_price + trail_value
            
            order = await place_kucoin_order(OrderRequest(
                symbol=params['symbol'],
                side=params['side'],
                type='stop',
                size=params['size'],
                price=stop_price,
                stopPrice=stop_price
            ))
            
            # Start trailing price updates
            asyncio.create_task(self._update_trailing_stop(order['orderId'], params))
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing trailing stop: {str(e)}")
            return None

    async def _update_trailing_stop(self, order_id, params):
        """Update trailing stop price"""
        try:
            while True:
                current_price = await get_current_price(params['symbol'])
                order = await get_order(order_id)
                
                if order['status'] == 'done':
                    break
                    
                new_stop = self._calculate_new_stop_price(
                    current_price,
                    float(order['stopPrice']),
                    params
                )
                
                if new_stop != float(order['stopPrice']):
                    await modify_order(order_id, new_stop_price=new_stop)
                    
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")

    async def start_twap_execution(self, params):
        """Start TWAP order execution"""
        try:
            total_quantity = params['size']
            num_slices = params['num_slices']
            duration_seconds = params['duration_minutes'] * 60
            
            slice_quantity = total_quantity / num_slices
            interval = duration_seconds / num_slices
            
            orders = []
            for i in range(num_slices):
                # Calculate random delay within interval
                delay = random.uniform(0, interval)
                await asyncio.sleep(delay)
                
                order = await place_kucoin_order(OrderRequest(
                    symbol=params['symbol'],
                    side=params['side'],
                    type='market',
                    size=slice_quantity
                ))
                
                orders.append(order)
                
            return {
                'type': 'TWAP',
                'orders': orders,
                'params': params
            }
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {str(e)}")
            return None

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics_history = {}
        self.benchmark_symbol = "BTC-USDT"  # Default benchmark

    async def calculate_performance_metrics(self, timeframe="1d"):
        """Calculate comprehensive performance metrics"""
        try:
            portfolio = await get_portfolio_summary()
            trades = await get_historical_trades(timeframe)
            
            metrics = {
                'returns': self._calculate_returns(portfolio),
                'risk_metrics': self._calculate_risk_metrics(portfolio),
                'trade_metrics': self._analyze_trades(trades),
                'attribution': self._calculate_attribution(portfolio),
                'benchmark_comparison': await self._compare_to_benchmark(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.metrics_history[datetime.now().date()] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return None

    def _calculate_returns(self, portfolio):
        """Calculate various return metrics"""
        try:
            return {
                'total_return': self._calculate_total_return(portfolio),
                'daily_returns': self._calculate_daily_returns(portfolio),
                'monthly_returns': self._calculate_monthly_returns(portfolio),
                'annualized_return': self._calculate_annualized_return(portfolio),
                'risk_adjusted_return': self._calculate_risk_adjusted_return(portfolio)
            }
        except Exception as e:
            logger.error(f"Error calculating returns: {str(e)}")
            return None

    def _analyze_trades(self, trades):
        """Analyze trading performance"""
        try:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'average_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'average_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'largest_win': max([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'largest_loss': min([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'average_hold_time': self._calculate_average_hold_time(trades),
                'profit_factor': self._calculate_profit_factor(trades)
            }
        except Exception as e:
            logger.error(f"Error analyzing trades: {str(e)}")
            return None

    async def generate_performance_report(self, timeframe="1d"):
        """Generate comprehensive performance report"""
        try:
            metrics = await self.calculate_performance_metrics(timeframe)
            
            report = {
                'summary': {
                    'total_return': metrics['returns']['total_return'],
                    'sharpe_ratio': metrics['risk_metrics']['sharpe_ratio'],
                    'win_rate': metrics['trade_metrics']['win_rate'],
                    'profit_factor': metrics['trade_metrics']['profit_factor']
                },
                'detailed_metrics': metrics,
                'charts': await self._generate_performance_charts(),
                'recommendations': self._generate_performance_recommendations(metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return None

    async def _generate_performance_charts(self):
        """Generate performance visualization charts"""
        try:
            return {
                'equity_curve': await self._generate_equity_curve(),
                'drawdown_chart': await self._generate_drawdown_chart(),
                'monthly_returns': await self._generate_monthly_returns_chart(),
                'win_loss_distribution': await self._generate_win_loss_distribution()
            }
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            return None

class RiskManager:
    def __init__(self):
        self.max_position_size = 0.1  # 10% of portfolio
        self.max_daily_drawdown = 0.05  # 5% daily drawdown limit
        self.position_limits = {}
        self.risk_metrics = {}
        
    async def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        try:
            portfolio = await get_portfolio_summary()
            positions = await get_active_positions()
            
            # Calculate Value at Risk (VaR)
            var_95 = self._calculate_var(portfolio['balances'], 0.95)
            var_99 = self._calculate_var(portfolio['balances'], 0.99)
            
            # Calculate portfolio volatility
            volatility = self._calculate_portfolio_volatility(portfolio['balances'])
            
            # Calculate correlation matrix
            correlations = self._calculate_correlation_matrix(portfolio['balances'])
            
            # Calculate risk exposure by asset
            risk_exposure = self._calculate_risk_exposure(positions)
            
            self.risk_metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'volatility': volatility,
                'correlations': correlations,
                'risk_exposure': risk_exposure,
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio),
                'max_drawdown': self._calculate_max_drawdown(portfolio),
                'beta': self._calculate_portfolio_beta(portfolio),
                'timestamp': datetime.now().isoformat()
            }
            
            return self.risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return None

    def validate_trade(self, order_request):
        """Validate trade against risk parameters"""
        try:
            # Check position size limits
            if not self._check_position_size(order_request):
                return False, "Position size exceeds limit"
                
            # Check daily drawdown
            if not self._check_drawdown_limit():
                return False, "Daily drawdown limit reached"
                
            # Check correlation risk
            if not self._check_correlation_risk(order_request.symbol):
                return False, "High correlation risk"
                
            # Check volatility limits
            if not self._check_volatility_limits(order_request.symbol):
                return False, "Volatility exceeds limits"
                
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}"

    async def _calculate_var(self, balances, confidence_level):
        """Calculate Value at Risk using historical simulation"""
        try:
            returns = []
            for balance in balances:
                symbol = f"{balance['currency']}-USDT"
                historical_data = await get_historical_prices(symbol, days=30)
                daily_returns = self._calculate_daily_returns(historical_data)
                returns.append(daily_returns)
                
            portfolio_returns = np.sum(returns, axis=0)
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return None

    def generate_risk_report(self):
        """Generate comprehensive risk report"""
        try:
            report = {
                'summary': {
                    'total_risk_score': self._calculate_risk_score(),
                    'risk_level': self._determine_risk_level(),
                    'main_risk_factors': self._identify_risk_factors()
                },
                'metrics': self.risk_metrics,
                'recommendations': self._generate_risk_recommendations(),
                'alerts': self._generate_risk_alerts(),
                'timestamp': datetime.now().isoformat()
            }
            return report
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return None

class MLStrategyOptimizer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

    def prepare_features(self, df):
        """Prepare features for ML models"""
        try:
            features = pd.DataFrame()
            
            # Technical indicators
            features['rsi'] = df['RSI']
            features['macd'] = df['MACD']
            features['bb_position'] = (df['close'] - df['BB_middle']) / (df['BB_upper'] - df['BB_lower'])
            features['volume_sma_ratio'] = df['volume'] / df['Volume_MA']
            
            # Price action features
            features['returns'] = df['close'].pct_change()
            features['volatility'] = df['returns'].rolling(window=20).std()
            features['trend'] = df['EMA_Short'] - df['EMA_Long']
            
            # Market microstructure
            features['spread'] = (df['high'] - df['low']) / df['close']
            features['volume_price_corr'] = df['volume'].rolling(20).corr(df['close'])
            
            # Target variable (next period return)
            features['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    async def optimize_strategy(self, symbol, timeframe="1d", optimization_period=90):
        """Optimize trading strategy using ML"""
        try:
            # Fetch historical data
            historical_data = await get_historical_data(symbol, timeframe, optimization_period)
            df = prepare_technical_indicators(historical_data)
            
            # Prepare features
            features = self.prepare_features(df)
            if features is None:
                return None
                
            # Split data
            X = features.drop('target', axis=1)
            y = features['target']
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Optimize different models
            rf_model = await self.optimize_random_forest(X, y, tscv)
            lstm_model = await self.optimize_lstm(X, y, tscv)
            ensemble = self.create_ensemble([rf_model, lstm_model])
            
            # Store models and performance
            self.models[symbol] = {
                'random_forest': rf_model,
                'lstm': lstm_model,
                'ensemble': ensemble
            }
            
            return {
                'models': self.models[symbol],
                'feature_importance': self.analyze_feature_importance(rf_model, X.columns),
                'performance': self.performance_metrics[symbol]
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            return None

    async def optimize_random_forest(self, X, y, tscv):
        """Optimize Random Forest model using Optuna"""
        try:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = RandomForestClassifier(**params, random_state=42)
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    scores.append(score)
                    
                return np.mean(scores)
                
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            
            # Train final model with best parameters
            best_rf = RandomForestClassifier(**study.best_params, random_state=42)
            best_rf.fit(X, y)
            
            return best_rf
            
        except Exception as e:
            logger.error(f"Error optimizing Random Forest: {str(e)}")
            return None

    async def optimize_lstm(self, X, y, tscv):
        """Optimize LSTM model using Optuna"""
        try:
            def create_lstm_model(trial):
                model = Sequential([
                    LSTM(
                        trial.suggest_int('lstm_units', 32, 256),
                        input_shape=(X.shape[1], 1),
                        return_sequences=True
                    ),
                    Dropout(trial.suggest_float('dropout1', 0.1, 0.5)),
                    LSTM(trial.suggest_int('lstm_units2', 16, 128)),
                    Dropout(trial.suggest_float('dropout2', 0.1, 0.5)),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                return model
                
            def objective(trial):
                model = create_lstm_model(trial)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Reshape data for LSTM
                    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_val_reshaped = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
                    
                    model.fit(
                        X_train_reshaped, y_train,
                        epochs=trial.suggest_int('epochs', 10, 50),
                        batch_size=trial.suggest_int('batch_size', 16, 128),
                        validation_data=(X_val_reshaped, y_val),
                        verbose=0
                    )
                    
                    score = model.evaluate(X_val_reshaped, y_val, verbose=0)[1]
                    scores.append(score)
                    
                return np.mean(scores)
                
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            # Train final model with best parameters
            best_lstm = create_lstm_model(study.best_trial)
            X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
            best_lstm.fit(X_reshaped, y, epochs=study.best_params['epochs'],
                         batch_size=study.best_params['batch_size'])
            
            return best_lstm
            
        except Exception as e:
            logger.error(f"Error optimizing LSTM: {str(e)}")
            return None

    def create_ensemble(self, models):
        """Create ensemble model from optimized models"""
        class EnsembleModel:
            def __init__(self, models):
                self.models = models
                
            def predict(self, X):
                predictions = []
                for model in self.models:
                    if isinstance(model, RandomForestClassifier):
                        pred = model.predict_proba(X)[:, 1]
                    else:  # LSTM
                        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
                        pred = model.predict(X_reshaped).flatten()
                    predictions.append(pred)
                    
                return np.mean(predictions, axis=0) > 0.5
                
        return EnsembleModel(models)

    def analyze_feature_importance(self, model, feature_names):
        """Analyze feature importance from Random Forest model"""
        try:
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return None

class MarketAnalyzer:
    def __init__(self):
        self.market_data = {}
        self.alerts = []
        self.websocket = None
        self.alert_thresholds = {
            'volatility': 2.0,  # Standard deviations
            'volume_spike': 3.0,  # Times average volume
            'price_change': 0.02  # 2% price change
        }

    async def start_real_time_analysis(self, symbols):
        """Start real-time market analysis"""
        try:
            # Initialize data structures for each symbol
            for symbol in symbols:
                self.market_data[symbol] = {
                    'price_history': [],
                    'volume_history': [],
                    'indicators': {},
                    'patterns': [],
                    'alerts': []
                }

            # Start WebSocket connection
            await self.connect_websocket(symbols)
            
            # Start analysis loops
            await asyncio.gather(
                self.update_technical_analysis(),
                self.monitor_market_conditions(),
                self.detect_patterns(),
                self.generate_alerts()
            )
            
        except Exception as e:
            logger.error(f"Error in real-time analysis: {str(e)}")

    async def connect_websocket(self, symbols):
        """Connect to KuCoin WebSocket for real-time data"""
        try:
            # Get WebSocket token
            endpoint = "/api/v1/bullet-public"
            url = BASE_URL + endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(url) as response:
                    token_data = await response.json()
                    
            # Connect to WebSocket
            ws_url = f"{token_data['data']['instanceServers'][0]['endpoint']}?token={token_data['data']['token']}"
            self.websocket = await websockets.connect(ws_url)
            
            # Subscribe to market data
            for symbol in symbols:
                subscribe_message = {
                    "type": "subscribe",
                    "topic": f"/market/ticker:{symbol}",
                    "privateChannel": False,
                    "response": True
                }
                await self.websocket.send(json.dumps(subscribe_message))
                
            # Start message handler
            asyncio.create_task(self.handle_websocket_messages())
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")

    async def handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data['type'] == 'message':
                    symbol = data['topic'].split(':')[1]
                    price_data = data['data']
                    
                    # Update market data
                    self.market_data[symbol]['price_history'].append({
                        'timestamp': datetime.now(),
                        'price': float(price_data['price']),
                        'volume': float(price_data['volume'])
                    })
                    
                    # Trim history to keep last 1000 points
                    if len(self.market_data[symbol]['price_history']) > 1000:
                        self.market_data[symbol]['price_history'].pop(0)
                        
                    # Trigger real-time analysis
                    await self.analyze_tick_data(symbol, price_data)
                    
        except Exception as e:
            logger.error(f"WebSocket message handling error: {str(e)}")

    async def analyze_tick_data(self, symbol, tick_data):
        """Analyze incoming tick data"""
        try:
            # Convert price history to DataFrame
            df = pd.DataFrame(self.market_data[symbol]['price_history'])
            
            # Calculate real-time indicators
            indicators = {
                'vwap': self.calculate_vwap(df),
                'price_momentum': self.calculate_momentum(df['price']),
                'volume_momentum': self.calculate_momentum(df['volume']),
                'bid_ask_imbalance': self.calculate_order_imbalance(tick_data)
            }
            
            # Update indicators
            self.market_data[symbol]['indicators'] = indicators
            
            # Check for significant events
            await self.check_market_events(symbol, indicators)
            
        except Exception as e:
            logger.error(f"Tick analysis error: {str(e)}")

    async def check_market_events(self, symbol, indicators):
        """Check for significant market events"""
        try:
            events = []
            
            # Check volume spike
            if indicators['volume_momentum'] > self.alert_thresholds['volume_spike']:
                events.append({
                    'type': 'volume_spike',
                    'severity': 'high',
                    'message': f'Unusual volume detected in {symbol}'
                })
                
            # Check price volatility
            price_std = np.std([x['price'] for x in self.market_data[symbol]['price_history'][-20:]])
            if price_std > self.alert_thresholds['volatility']:
                events.append({
                    'type': 'high_volatility',
                    'severity': 'medium',
                    'message': f'High volatility detected in {symbol}'
                })
                
            # Check order imbalance
            if abs(indicators['bid_ask_imbalance']) > 2.0:
                events.append({
                    'type': 'order_imbalance',
                    'severity': 'medium',
                    'message': f'Significant order imbalance in {symbol}'
                })
                
            # Generate alerts for significant events
            for event in events:
                await self.generate_alert(symbol, event)
                
        except Exception as e:
            logger.error(f"Market event check error: {str(e)}")

    async def generate_alert(self, symbol, event):
        """Generate and distribute market alerts"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': event['type'],
                'severity': event['severity'],
                'message': event['message']
            }
            
            # Store alert
            self.alerts.append(alert)
            
            # Broadcast alert to connected clients
            if len(active_connections) > 0:
                alert_message = {
                    'type': 'market_alert',
                    'data': alert
                }
                await self.broadcast_message(alert_message)
                
        except Exception as e:
            logger.error(f"Alert generation error: {str(e)}")

    def calculate_vwap(self, df):
        """Calculate Volume-Weighted Average Price"""
        try:
            df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
            return df['vwap'].iloc[-1]
        except Exception as e:
            logger.error(f"VWAP calculation error: {str(e)}")
            return None

    def calculate_momentum(self, series, period=20):
        """Calculate momentum indicator"""
        try:
            return (series.iloc[-1] / series.iloc[-period] - 1) if len(series) >= period else 0
        except Exception as e:
            logger.error(f"Momentum calculation error: {str(e)}")
            return 0

    def calculate_order_imbalance(self, tick_data):
        """Calculate order book imbalance"""
        try:
            bid_volume = float(tick_data['bidSize'])
            ask_volume = float(tick_data['askSize'])
            return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        except Exception as e:
            logger.error(f"Order imbalance calculation error: {str(e)}")
            return 0

