import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------ DATABASE ------------------------------
USER_DB = {"service_providers": {}, "remote_users": {}}
USER_INFO = {"remote_users": {}, "service_providers": {}}
WATCHLIST_DB = {}

def save_user_db():
    try:
        with open("user_db.json", "w") as f:
            json.dump(USER_DB, f)
        with open("user_info.json", "w") as f:
            json.dump(USER_INFO, f)
        with open("watchlist_db.json", "w") as f:
            json.dump(WATCHLIST_DB, f)
    except Exception as e:
        print(f"Error saving database: {e}")

def load_user_db():
    global USER_DB, USER_INFO, WATCHLIST_DB
    try:
        if os.path.exists("user_db.json"):
            with open("user_db.json", "r") as f:
                USER_DB = json.load(f)
        if os.path.exists("user_info.json"):
            with open("user_info.json", "r") as f:
                USER_INFO = json.load(f)
        if os.path.exists("watchlist_db.json"):
            with open("watchlist_db.json", "r") as f:
                WATCHLIST_DB = json.load(f)
    except Exception as e:
        print(f"Error loading database: {e}")

load_user_db()

# ------------------------------ STREAMLIT CONFIG ------------------------------
st.set_page_config(page_title="📈 Stock Trend Predictor", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #e5f4ea; }
        .stButton>button { background-color: #0e4429; color: white; border-radius: 8px; }
        .stTextInput>div>input { background-color: #f1fff7; }
        h1, h2, h3, .stMarkdown { color: #114e3b; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------ AUTH ------------------------------
def login_user(role):
    st.subheader(f"🔐 {role.title().replace('_', ' ')} Login")
    username = st.text_input("Username", key=f"{role}_username")
    password = st.text_input("Password", type="password", key=f"{role}_password")
    if st.button("Login", key=f"{role}_login"):
        if username in USER_DB[role] and USER_DB[role][username] == password:
            st.session_state.logged_in = True
            st.session_state.user_role = role
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

def register_user():
    st.subheader("📝 Register (Remote User)")
    new_user = st.text_input("Choose Username")
    new_pass = st.text_input("Choose Password", type="password")
    email = st.text_input("Email")
    mobile = st.text_input("Mobile Number")
    if st.button("Register"):
        if new_user in USER_DB['remote_users']:
            st.warning("Username already exists")
        else:
            USER_DB['remote_users'][new_user] = new_pass
            USER_INFO['remote_users'][new_user] = {"email": email, "mobile": mobile}
            WATCHLIST_DB[new_user] = []
            save_user_db()
            st.success("Registration successful. You can now login.")

def register_service_provider():
    st.subheader("📝 Register (Service Provider)")
    new_user = st.text_input("Choose SP Username")
    new_pass = st.text_input("Choose SP Password", type="password")
    email = st.text_input("SP Email")
    mobile = st.text_input("SP Mobile")
    if st.button("Register SP"):
        if new_user in USER_DB['service_providers']:
            st.warning("Service Provider already exists")
        else:
            USER_DB['service_providers'][new_user] = new_pass
            USER_INFO['service_providers'][new_user] = {"email": email, "mobile": mobile}
            save_user_db()
            st.success("Service Provider registered successfully")

# ------------------------------ PREDICTION HELPERS ------------------------------
@st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, start="2013-01-01")
        if df.empty:
            raise ValueError("No data found for this ticker")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Failed to load data for {ticker}: {str(e)}")

def add_indicators(df):
    try:
        df = df.copy()
        df['SMA'] = df['Close'].rolling(window=14).mean()
        df['EMA'] = df['Close'].ewm(span=14).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['StdDev'] = df['Close'].rolling(window=10).std()
        df['ROC'] = df['Close'].pct_change(periods=10)
        high_max = df['High'].rolling(14).max()
        low_min = df['Low'].rolling(14).min()
        df['Williams %R'] = (high_max - df['Close']) / (high_max - low_min) * -100
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Failed to add indicators: {str(e)}")

def create_labels(df):
    try:
        df = df.copy()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df = df[:-1]  # Remove last row
        return df
    except Exception as e:
        raise Exception(f"Failed to create labels: {str(e)}")

def preprocess(df):
    try:
        features = ['SMA', 'EMA', 'RSI', 'MACD', 'OBV', 'Momentum', 'StdDev', 'ROC', 'Williams %R']
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = df[features].copy()
        y = df['Target'].copy()
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            raise ValueError(f"Insufficient data: only {len(X)} samples available, need at least 50")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        test_size = min(0.2, 20/len(X))
        split_result = train_test_split(X_scaled, y, shuffle=False, test_size=test_size)
        return split_result, X_scaled
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")

def train_ml_models(X_train, y_train):
    try:
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        rf = RandomForestClassifier(random_state=42, n_estimators=50)
        svm = SVC(probability=True, random_state=42, kernel='rbf', C=1.0)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=50)
        rf.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        return rf, svm, xgb
    except Exception as e:
        print(f"ML training error: {str(e)}")
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        return dummy, dummy, dummy

def reshape(X):
    return X.reshape((X.shape[0], 1, X.shape[1]))

def build_lstm(input_shape):
    model = Sequential([LSTM(64, input_shape=input_shape), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_deep_model(model, X_train, y_train):
    try:
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(monitor='loss', patience=3)])
        return model
    except Exception as e:
        print(f"LSTM training error: {str(e)}")
        return model

def get_model_prediction(model, X_test, y_test=None, deep=False):
    try:
        if deep:
            y_pred = model.predict(X_test, verbose=0)
            y_pred = y_pred.flatten()
        else:
            y_pred = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred > 0.5).astype(int)
        if y_test is not None and len(y_test) > 0:
            acc = accuracy_score(y_test, y_pred_class)
        else:
            acc = 0.5
        return int(y_pred_class[-1]), float(acc)
    except Exception as e:
        print(f"Model prediction error: {str(e)}")
        return 0, 0.5

def get_final_prediction(X_train, X_test, y_train, y_test):
    try:
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Empty training or test data")
        if len(np.unique(y_train)) < 2:
            return int(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0])
        
        rf, svm, xgb = train_ml_models(X_train, y_train)
        X_train_rnn = reshape(X_train)
        X_test_rnn = reshape(X_test)
        lstm = train_deep_model(build_lstm(X_train_rnn.shape[1:]), X_train_rnn, y_train)
        
        preds = {}
        try:
            preds["RF"] = get_model_prediction(rf, X_test, y_test)
        except:
            preds["RF"] = (0, 0.5)
        try:
            preds["SVM"] = get_model_prediction(svm, X_test, y_test)
        except:
            preds["SVM"] = (0, 0.5)
        try:
            preds["XGB"] = get_model_prediction(xgb, X_test, y_test)
        except:
            preds["XGB"] = (0, 0.5)
        try:
            preds["LSTM"] = get_model_prediction(lstm, X_test_rnn, y_test, deep=True)
        except:
            preds["LSTM"] = (0, 0.5)
        
        votes, weights = zip(*preds.values())
        final = int(np.average(votes, weights=weights) >= 0.5)
        return final
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 1

def get_stock_info(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return None
            
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        day_high = float(hist['High'].iloc[-1])
        day_low = float(hist['Low'].iloc[-1])
        volume = int(hist['Volume'].iloc[-1])
        
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
        week_52_high = float(hist['High'].max())
        week_52_low = float(hist['Low'].min())
        avg_volume = int(hist['Volume'].tail(30).mean())
        volatility = float(hist['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100)
        
        market_cap = info.get('marketCap')
        pe_ratio = info.get('trailingPE')
        dividend_yield = info.get('dividendYield')
        beta = info.get('beta')
        
        metrics = {
            'current_price': current_price,
            'prev_close': prev_close,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'day_high': day_high,
            'day_low': day_low,
            'volume': volume,
            'avg_volume': avg_volume,
            'week_52_high': week_52_high,
            'week_52_low': week_52_low,
            'volatility': volatility,
            'market_cap': market_cap if market_cap and isinstance(market_cap, (int, float)) else None,
            'pe_ratio': pe_ratio if pe_ratio and isinstance(pe_ratio, (int, float)) else None,
            'dividend_yield': dividend_yield if dividend_yield and isinstance(dividend_yield, (int, float)) else None,
            'beta': beta if beta and isinstance(beta, (int, float)) else None,
            'company_name': info.get('longName', ticker_symbol)
        }
        return metrics
    except Exception as e:
        print(f"Error fetching stock info for {ticker_symbol}: {str(e)}")
        return None

def calculate_support_resistance(df, window=20):
    try:
        recent_data = df.tail(100)
        support = float(recent_data['Low'].rolling(window=window).min().iloc[-1])
        resistance = float(recent_data['High'].rolling(window=window).max().iloc[-1])
        return support, resistance
    except Exception as e:
        print(f"Error calculating support/resistance: {str(e)}")
        return None, None

def plot_price_with_indicators(df, symbol, prediction, stock_info=None):
    try:
        if 'Date' not in df.columns:
            df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        
        last_year = df[df['Date'] >= (df['Date'].max() - pd.Timedelta(days=365))].copy()
        
        if len(last_year) == 0:
            st.warning("Not enough data to plot chart")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        ax1.plot(last_year['Date'], last_year['Close'], label='Close Price', linewidth=2, color='blue')
        
        if 'SMA' in last_year.columns and not last_year['SMA'].isna().all():
            ax1.plot(last_year['Date'], last_year['SMA'], label='SMA (14)', linewidth=1, alpha=0.7, color='orange')
        
        if 'EMA' in last_year.columns and not last_year['EMA'].isna().all():
            ax1.plot(last_year['Date'], last_year['EMA'], label='EMA (14)', linewidth=1, alpha=0.7, color='red')
        
        support, resistance = calculate_support_resistance(last_year)
        if support is not None and resistance is not None:
            ax1.axhline(y=support, color='green', linestyle='--', alpha=0.7, label=f'Support: {support:.2f}')
            ax1.axhline(y=resistance, color='red', linestyle='--', alpha=0.7, label=f'Resistance: {resistance:.2f}')
        
        signal = float(last_year['Close'].iloc[-1])
        next_day = last_year['Date'].iloc[-1] + pd.Timedelta(days=1)
        color = 'green' if prediction == 1 else 'red'
        marker = '^' if prediction == 1 else 'v'
        label = 'Predicted Up 📈' if prediction == 1 else 'Predicted Down 📉'
        ax1.scatter(next_day, signal, color=color, s=150, label=label, marker=marker, zorder=5)
        
        ax1.set_title(f"Stock Analysis: {symbol}", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Price", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        if 'RSI' in last_year.columns and not last_year['RSI'].isna().all():
            ax2.plot(last_year['Date'], last_year['RSI'], color='purple', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.set_title("RSI (Relative Strength Index)", fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.legend(fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'RSI data not available', transform=ax2.transAxes, ha='center', va='center')
        
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("RSI", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def display_trading_metrics(stock_info, df):
    """Display comprehensive trading metrics"""
    if not stock_info:
        st.warning("Unable to fetch detailed stock information")
        return
        
    st.subheader("📊 Trading Metrics & Key Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"₹{stock_info['current_price']:.2f}",
            delta=f"{stock_info['price_change']:+.2f} ({stock_info['price_change_pct']:+.2f}%)"
        )
    
    with col2:
        st.metric(
            label="📈 Day's High",
            value=f"₹{stock_info['day_high']:.2f}"
        )
    
    with col3:
        st.metric(
            label="📉 Day's Low", 
            value=f"₹{stock_info['day_low']:.2f}"
        )
    
    with col4:
        volume_str = f"{stock_info['volume']:,}"
        st.metric(
            label="📊 Volume",
            value=volume_str
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = stock_info['market_cap']
        if market_cap and isinstance(market_cap, (int, float)):
            market_cap_cr = market_cap / 10000000
            st.metric(label="🏢 Market Cap", value=f"₹{market_cap_cr:.0f}Cr")
        else:
            st.metric(label="🏢 Market Cap", value="N/A")
    
    with col2:
        pe_ratio = stock_info['pe_ratio']
        if pe_ratio and isinstance(pe_ratio, (int, float)):
            st.metric(label="📊 P/E Ratio", value=f"{pe_ratio:.2f}")
        else:
            st.metric(label="📊 P/E Ratio", value="N/A")
    
    with col3:
        st.metric(
            label="🎯 52W High",
            value=f"₹{stock_info['week_52_high']:.2f}"
        )
    
    with col4:
        st.metric(
            label="🎯 52W Low",
            value=f"₹{stock_info['week_52_low']:.2f}"
        )
    
    st.subheader("🔍 Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        latest_rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not df['RSI'].isna().all() else None
        latest_macd = float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not df['MACD'].isna().all() else None
    except:
        latest_rsi = None
        latest_macd = None
    
    with col1:
        if latest_rsi is not None:
            rsi_status = "Overbought 🔴" if latest_rsi > 70 else "Oversold 🟢" if latest_rsi < 30 else "Neutral 🟡"
            st.metric(label="RSI", value=f"{latest_rsi:.1f}", help=f"Status: {rsi_status}")
        else:
            st.metric(label="RSI", value="N/A")
    
    with col2:
        if latest_macd is not None:
            st.metric(label="MACD", value=f"{latest_macd:.3f}")
        else:
            st.metric(label="MACD", value="N/A")
    
    with col3:
        volatility = stock_info['volatility']
        if volatility and not np.isnan(volatility):
            st.metric(label="Volatility", value=f"{volatility:.1f}%")
        else:
            st.metric(label="Volatility", value="N/A")
    
    with col4:
        beta = stock_info['beta']
        if beta and isinstance(beta, (int, float)):
            st.metric(label="Beta", value=f"{beta:.2f}")
        else:
            st.metric(label="Beta", value="N/A")
    
    st.subheader("🚦 Trading Signals")
    col1, col2 = st.columns(2)
    
    with col1:
        if latest_rsi is not None:
            if latest_rsi > 70:
                st.warning("⚠️ RSI indicates OVERBOUGHT - Consider selling")
            elif latest_rsi < 30:
                st.success("✅ RSI indicates OVERSOLD - Consider buying")
            else:
                st.info("ℹ️ RSI is in neutral zone")
        else:
            st.info("ℹ️ RSI data not available")
    
    with col2:
        if latest_macd is not None:
            if latest_macd > 0:
                st.success("✅ MACD is positive - Bullish signal")
            else:
                st.warning("⚠️ MACD is negative - Bearish signal")
        else:
            st.info("ℹ️ MACD data not available")

def analyze_stock(symbol):
    """Centralized function to analyze a single stock"""
    try:
        # Load data
        df = load_data(symbol)
        if df.empty:
            raise ValueError("No data available")
        
        # Add indicators
        df = add_indicators(df)
        df = create_labels(df)
        
        # Preprocess
        (X_train, X_test, y_train, y_test), _ = preprocess(df)
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Insufficient data for analysis")
        
        # Get prediction
        prediction = get_final_prediction(X_train, X_test, y_train, y_test)
        
        # Get current price
        current_price = float(df['Close'].iloc[-1])
        
        return {
            'prediction': prediction,
            'current_price': current_price,
            'df': df,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'prediction': None,
            'current_price': None,
            'df': None,
            'success': False,
            'error': str(e)
        }
    if not stock_info:
        st.warning("Unable to fetch detailed stock information")
        return
        
    st.subheader("📊 Trading Metrics & Key Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"₹{stock_info['current_price']:.2f}",
            delta=f"{stock_info['price_change']:+.2f} ({stock_info['price_change_pct']:+.2f}%)"
        )
    
    with col2:
        st.metric(
            label="📈 Day's High",
            value=f"₹{stock_info['day_high']:.2f}"
        )
    
    with col3:
        st.metric(
            label="📉 Day's Low", 
            value=f"₹{stock_info['day_low']:.2f}"
        )
    
    with col4:
        volume_str = f"{stock_info['volume']:,}"
        st.metric(
            label="📊 Volume",
            value=volume_str
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = stock_info['market_cap']
        if market_cap and isinstance(market_cap, (int, float)):
            market_cap_cr = market_cap / 10000000
            st.metric(label="🏢 Market Cap", value=f"₹{market_cap_cr:.0f}Cr")
        else:
            st.metric(label="🏢 Market Cap", value="N/A")
    
    with col2:
        pe_ratio = stock_info['pe_ratio']
        if pe_ratio and isinstance(pe_ratio, (int, float)):
            st.metric(label="📊 P/E Ratio", value=f"{pe_ratio:.2f}")
        else:
            st.metric(label="📊 P/E Ratio", value="N/A")
    
    with col3:
        st.metric(
            label="🎯 52W High",
            value=f"₹{stock_info['week_52_high']:.2f}"
        )
    
    with col4:
        st.metric(
            label="🎯 52W Low",
            value=f"₹{stock_info['week_52_low']:.2f}"
        )
    
    st.subheader("🔍 Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        latest_rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not df['RSI'].isna().all() else None
        latest_macd = float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not df['MACD'].isna().all() else None
    except:
        latest_rsi = None
        latest_macd = None
    
    with col1:
        if latest_rsi is not None:
            rsi_status = "Overbought 🔴" if latest_rsi > 70 else "Oversold 🟢" if latest_rsi < 30 else "Neutral 🟡"
            st.metric(label="RSI", value=f"{latest_rsi:.1f}", help=f"Status: {rsi_status}")
        else:
            st.metric(label="RSI", value="N/A")
    
    with col2:
        if latest_macd is not None:
            st.metric(label="MACD", value=f"{latest_macd:.3f}")
        else:
            st.metric(label="MACD", value="N/A")
    
    with col3:
        volatility = stock_info['volatility']
        if volatility and not np.isnan(volatility):
            st.metric(label="Volatility", value=f"{volatility:.1f}%")
        else:
            st.metric(label="Volatility", value="N/A")
    
    with col4:
        beta = stock_info['beta']
        if beta and isinstance(beta, (int, float)):
            st.metric(label="Beta", value=f"{beta:.2f}")
        else:
            st.metric(label="Beta", value="N/A")
    
    st.subheader("🚦 Trading Signals")
    col1, col2 = st.columns(2)
    
    with col1:
        if latest_rsi is not None:
            if latest_rsi > 70:
                st.warning("⚠️ RSI indicates OVERBOUGHT - Consider selling")
            elif latest_rsi < 30:
                st.success("✅ RSI indicates OVERSOLD - Consider buying")
            else:
                st.info("ℹ️ RSI is in neutral zone")
        else:
            st.info("ℹ️ RSI data not available")
    
    with col2:
        if latest_macd is not None:
            if latest_macd > 0:
                st.success("✅ MACD is positive - Bullish signal")
            else:
                st.warning("⚠️ MACD is negative - Bearish signal")
        else:
            st.info("ℹ️ MACD data not available")

# ------------------------------ MAIN UI ------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

st.title("💰 Stock Market Trend Prediction")

if not st.session_state.logged_in:
    tab1, tab2, tab3, tab4 = st.tabs(["SP Login", "Remote User Login", "Register User", "Register SP"])
    
    with tab1: 
        login_user("service_providers")
    
    with tab2: 
        login_user("remote_users")
    
    with tab3: 
        register_user()
    
    with tab4: 
        register_service_provider()

else:
    st.sidebar.success(f"Logged in as {st.session_state.username} ({st.session_state.user_role.replace('_',' ').title()})")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown(f"### 👋 Hello, **{st.session_state.username}**!")

    # Service Provider Dashboard
    if st.session_state.user_role == "service_providers":
        st.subheader("📊 Service Provider Dashboard")
        st.info("Welcome to the Service Provider dashboard! Here you can view all registered remote users.")
        
        st.subheader("👥 Registered Remote Users")
        remote_users = USER_INFO.get("remote_users", {})
        
        if remote_users:
            user_data = []
            for username, info in remote_users.items():
                user_data.append({
                    "Username": username,
                    "Email": info.get("email", "N/A"),
                    "Mobile": info.get("mobile", "N/A")
                })
            
            df_users = pd.DataFrame(user_data)
            st.dataframe(df_users, use_container_width=True)
            st.success(f"Total registered remote users: {len(remote_users)}")
        else:
            st.info("No remote users registered yet.")
            empty_df = pd.DataFrame(columns=["Username", "Email", "Mobile"])
            st.dataframe(empty_df, use_container_width=True)

    # Remote User Features
    elif st.session_state.user_role == "remote_users":
        st.subheader("⭐ Your Watchlist")
        watchlist = WATCHLIST_DB.get(st.session_state.username, [])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_stock = st.text_input("Add stock to watchlist (e.g., INFY.NS, TCS.NS, RELIANCE.NS)")
        with col2:
            st.write("")
            if st.button("Add to Watchlist", type="primary"):
                if new_stock and new_stock.upper() not in watchlist:
                    watchlist.append(new_stock.upper())
                    WATCHLIST_DB[st.session_state.username] = watchlist
                    save_user_db()
                    st.success(f"Added {new_stock.upper()} to watchlist!")
                    st.rerun()
                elif new_stock and new_stock.upper() in watchlist:
                    st.warning("Stock already in watchlist!")
                else:
                    st.error("Please enter a valid stock symbol!")

        if watchlist:
            st.write("**Your Current Watchlist:**", ", ".join(watchlist))
            
            if st.button("🔄 Analyze Watchlist", type="primary"):
                uptrend, downtrend, errors = [], [], []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, symbol in enumerate(watchlist):
                    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(watchlist)})")
                    progress_bar.progress((i+1)/len(watchlist))
                    
                    # Use centralized analysis function
                    result = analyze_stock(symbol)
                    
                    if result['success']:
                        prediction = result['prediction']
                        current_price = result['current_price']
                        
                        if prediction == 1:
                            uptrend.append(f"{symbol} (₹{current_price:.2f})")
                        else:
                            downtrend.append(f"{symbol} (₹{current_price:.2f})")
                    else:
                        errors.append(f"{symbol}: {result['error']}")
                
                status_text.text("Analysis complete!")
                progress_bar.empty()
                status_text.empty()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if uptrend:
                        st.success(f"📈 **Uptrend Stocks:**")
                        for stock in uptrend:
                            st.write(f"• {stock}")
                    else:
                        st.info("📈 No stocks showing uptrend")
                
                with col2:
                    if downtrend:
                        st.error(f"📉 **Downtrend Stocks:**")
                        for stock in downtrend:
                            st.write(f"• {stock}")
                    else:
                        st.info("📉 No stocks showing downtrend")
                
                if errors:
                    st.warning("⚠️ **Errors encountered:**")
                    for error in errors:
                        st.write(f"• {error}")
        else:
            st.info("Your watchlist is empty. Add some stocks to get started!")

        # Stock Search and Prediction Section
        st.markdown("---")
        st.subheader("🔍 Search and Predict Individual Stock")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_symbol = st.text_input("Enter stock symbol (e.g., UPL.NS, TCS.NS, RELIANCE.NS)")
        with col2:
            st.write("")
            predict_button = st.button("🔮 Predict Trend", type="primary")

        if predict_button and search_symbol:
            try:
                with st.spinner(f"Analyzing {search_symbol.upper()}..."):
                    # Use centralized analysis function
                    result = analyze_stock(search_symbol)
                    
                    if not result['success']:
                        st.error(f"❌ {result['error']}")
                        st.info("💡 For Indian stocks, use format: RELIANCE.NS, TCS.NS, INFY.NS")
                    else:
                        prediction = result['prediction']
                        df = result['df']
                        
                        if prediction == 1:
                            st.success(f"📈 **{search_symbol.upper()}** is predicted to have an **UPTREND**!")
                        else:
                            st.error(f"📉 **{search_symbol.upper()}** is predicted to have a **DOWNTREND**!")
                        
                        stock_info = get_stock_info(search_symbol)
                        display_trading_metrics(stock_info, df)
                        plot_price_with_indicators(df, search_symbol.upper(), prediction, stock_info)
                
            except Exception as e:
                st.error(f"❌ Error analyzing {search_symbol}: {str(e)}")
                st.info("💡 Please check if the stock symbol is correct (e.g., use .NS for NSE stocks like RELIANCE.NS)")
                print(f"Full error details: {str(e)}")

        elif predict_button:
            st.warning("Please enter a stock symbol to analyze!")

# Footer
st.markdown("---")
st.markdown("Stock Market Prediction App")
