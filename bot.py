"""
IBKR RL Trading Bot - Complete Implementation in One File
Includes: Data Pipeline, Neural Networks, RL Agent, Sentiment Analysis, IBKR Connector, and Trading Bot
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta, time
import logging
import json
import os
import pickle

try:
    import yfinance as yf
    from ib_insync import IB, Stock, LimitOrder, MarketOrder
except ImportError:
    pass

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class DataPipeline:
    """Fetches and preprocesses historical stock data."""
    
    def __init__(self, symbols: List[str], lookback_days: int = 365):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.data = {}
        self.normalized_data = {}
        self.scaler_params = {}
    
    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data from yfinance."""
        logger.info(f"Fetching historical data for {len(self.symbols)} symbols...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        for symbol in self.symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                df.columns = [col.lower() for col in df.columns]
                df = df.dropna()
                self.data[symbol] = df
                logger.info(f"Fetched {len(df)} days for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return self.data
    
    def normalize_data(self, method: str = 'minmax') -> Dict[str, np.ndarray]:
        """Normalize data to 0-1 range."""
        logger.info(f"Normalizing data using {method}...")
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
            
            df = self.data[symbol]
            ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values
            
            if method == 'minmax':
                min_vals = ohlcv.min(axis=0)
                max_vals = ohlcv.max(axis=0)
                normalized = (ohlcv - min_vals) / (max_vals - min_vals + 1e-8)
                self.scaler_params[symbol] = {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
            else:  # zscore
                mean_vals = ohlcv.mean(axis=0)
                std_vals = ohlcv.std(axis=0)
                normalized = (ohlcv - mean_vals) / (std_vals + 1e-8)
                self.scaler_params[symbol] = {'mean': mean_vals, 'std': std_vals, 'method': 'zscore'}
            
            self.normalized_data[symbol] = normalized
        
        return self.normalized_data
    
    def create_sequences(self, sequence_length: int = 30) -> Tuple[Dict, Dict]:
        """Create sequences for neural network training."""
        logger.info(f"Creating sequences with length {sequence_length}...")
        
        X = {}
        y = {}
        
        for symbol in self.symbols:
            if symbol not in self.normalized_data:
                continue
            
            data = self.normalized_data[symbol]
            X_seq = []
            y_seq = []
            
            for i in range(len(data) - sequence_length):
                X_seq.append(data[i:i+sequence_length])
                y_seq.append(data[i+sequence_length, 3])
            
            X[symbol] = np.array(X_seq, dtype=np.float32)
            y[symbol] = np.array(y_seq, dtype=np.float32)
            logger.info(f"Created {len(X_seq)} sequences for {symbol}")
        
        return X, y
    
    def denormalize(self, symbol: str, normalized_value: float) -> float:
        """Denormalize a value back to original scale."""
        if symbol not in self.scaler_params:
            return normalized_value
        
        params = self.scaler_params[symbol]
        
        if params['method'] == 'minmax':
            min_val = params['min'][3]
            max_val = params['max'][3]
            return normalized_value * (max_val - min_val) + min_val
        else:
            mean_val = params['mean'][3]
            std_val = params['std'][3]
            return normalized_value * std_val + mean_val


class RealTimeDataBuffer:
    """Maintains a rolling buffer of real-time price data."""
    
    def __init__(self, symbols: List[str], buffer_size: int = 30):
        self.symbols = symbols
        self.buffer_size = buffer_size
        self.buffers = {symbol: deque(maxlen=buffer_size) for symbol in symbols}
        self.latest_prices = {symbol: None for symbol in symbols}
    
    def add_data(self, symbol: str, ohlcv: np.ndarray):
        """Add OHLCV data to buffer."""
        if symbol in self.buffers:
            self.buffers[symbol].append(ohlcv)
            self.latest_prices[symbol] = ohlcv[3]
    
    def get_buffer_array(self, symbol: str) -> Optional[np.ndarray]:
        """Get buffer as array."""
        if symbol not in self.buffers or len(self.buffers[symbol]) < self.buffer_size:
            return None
        return np.array(list(self.buffers[symbol]), dtype=np.float32)
    
    def is_ready(self, symbol: str) -> bool:
        """Check if buffer has enough data."""
        return len(self.buffers[symbol]) >= self.buffer_size


class SentimentBuffer:
    """Maintains a rolling buffer of sentiment scores."""
    
    def __init__(self, symbols: List[str], window_size: int = 30):
        self.symbols = symbols
        self.window_size = window_size
        self.buffers = {symbol: deque(maxlen=window_size) for symbol in symbols}
        self.current_sentiment = {symbol: 0.0 for symbol in symbols}
    
    def add_sentiment(self, symbol: str, sentiment_score: float):
        """Add sentiment score."""
        if symbol in self.buffers:
            self.buffers[symbol].append(sentiment_score)
            self.current_sentiment[symbol] = sentiment_score
    
    def get_sentiment_array(self, symbol: str) -> np.ndarray:
        """Get sentiment buffer as array."""
        if symbol not in self.buffers:
            return np.zeros(self.window_size, dtype=np.float32)
        
        buffer = list(self.buffers[symbol])
        if len(buffer) < self.window_size:
            buffer = [0.0] * (self.window_size - len(buffer)) + buffer
        
        return np.array(buffer[-self.window_size:], dtype=np.float32)
    
    def get_current_sentiment(self, symbol: str) -> float:
        """Get current sentiment score."""
        return self.current_sentiment.get(symbol, 0.0)


class OnlineLearningBuffer:
    """Maintains a rolling buffer of recent trading experiences."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add_sample(self, price_sequence: np.ndarray, sentiment_sequence: np.ndarray,
                   next_price: float, action: int, reward: float, timestamp: datetime):
        """Add a sample to the buffer."""
        self.buffer.append({
            'price_sequence': price_sequence,
            'sentiment_sequence': sentiment_sequence,
            'next_price': next_price,
            'action': action,
            'reward': reward,
            'timestamp': timestamp
        })
    
    def get_batch(self, batch_size: int) -> Optional[Dict]:
        """Get a random batch from the buffer."""
        if len(self.buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        return {
            'price_sequences': np.array([self.buffer[i]['price_sequence'] for i in indices]),
            'sentiment_sequences': np.array([self.buffer[i]['sentiment_sequence'] for i in indices]),
            'next_prices': np.array([self.buffer[i]['next_price'] for i in indices]),
            'actions': np.array([self.buffer[i]['action'] for i in indices]),
            'rewards': np.array([self.buffer[i]['reward'] for i in indices])
        }
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


# ============================================================================
# NEURAL NETWORKS
# ============================================================================

class HybridPricePredictor(nn.Module):
    """Hybrid LSTM + CNN neural network for price prediction."""
    
    def __init__(self, sequence_length: int = 30, num_features: int = 5, sentiment_dim: int = 30):
        super().__init__()
        
        # CNN Branch
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # LSTM Branch
        self.lstm = nn.LSTM(num_features, 64, num_layers=2, batch_first=True, dropout=0.3)
        
        # Dense layers
        cnn_output_size = 128 * sequence_length
        lstm_output_size = 64
        total_input = cnn_output_size + lstm_output_size + sentiment_dim
        
        self.dense = nn.Sequential(
            nn.Linear(total_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        combined = torch.cat([cnn_out, lstm_out, sentiment], dim=1)
        output = self.dense(combined)
        
        return output


class PricePredictor:
    """Wrapper for price prediction model."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def build_model(self, sequence_length: int = 30, num_features: int = 5, sentiment_dim: int = 30):
        """Build the neural network model."""
        self.model = HybridPricePredictor(sequence_length, num_features, sentiment_dim)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        logger.info("Price predictor model built")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 10, batch_size: int = 32):
        """Train the model on historical data."""
        if self.model is None:
            self.build_model()
        
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                batch_sentiment = np.random.randn(len(batch_X), 30).astype(np.float32)
                
                X_tensor = torch.from_numpy(batch_X).to(self.device)
                y_tensor = torch.from_numpy(batch_y).unsqueeze(1).to(self.device)
                sentiment_tensor = torch.from_numpy(batch_sentiment).to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(X_tensor, sentiment_tensor)
                loss = self.criterion(predictions, y_tensor)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= (len(X_train) // batch_size)
            self.train_losses.append(train_loss)
            
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]
                    batch_sentiment = np.random.randn(len(batch_X), 30).astype(np.float32)
                    
                    X_tensor = torch.from_numpy(batch_X).to(self.device)
                    y_tensor = torch.from_numpy(batch_y).unsqueeze(1).to(self.device)
                    sentiment_tensor = torch.from_numpy(batch_sentiment).to(self.device)
                    
                    predictions = self.model(X_tensor, sentiment_tensor)
                    loss = self.criterion(predictions, y_tensor)
                    val_loss += loss.item()
                
                val_loss /= (len(X_val) // batch_size)
                self.val_losses.append(val_loss)
            
            self.model.train()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    def predict(self, X: np.ndarray, sentiment: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device)
            sentiment_tensor = torch.from_numpy(sentiment).to(self.device)
            predictions = self.model(X_tensor, sentiment_tensor)
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Model loaded from {filepath}")


# ============================================================================
# REINFORCEMENT LEARNING
# ============================================================================

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO agent."""
    
    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class PPOAgent:
    """PPO (Proximal Policy Optimization) agent for trading."""
    
    def __init__(self, state_dim: int = 32, action_dim: int = 3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.network = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action based on state."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=1).item()
            log_prob = torch.log(action_probs[0, action]).item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def train(self, epochs: int = 3, batch_size: int = 16):
        """Train the agent using PPO."""
        if len(self.states) == 0:
            return
        
        returns = []
        advantages = []
        
        R = 0
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.gamma * R * (1 - self.dones[t])
            returns.insert(0, R)
            advantages.insert(0, R - self.values[t])
        
        returns = np.array(returns)
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(self.states))
            
            for i in range(0, len(self.states), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                states = torch.from_numpy(np.array([self.states[j] for j in batch_indices])).float().to(self.device)
                actions = torch.tensor([self.actions[j] for j in batch_indices]).to(self.device)
                old_log_probs = torch.tensor([self.log_probs[j] for j in batch_indices]).to(self.device)
                batch_returns = torch.from_numpy(returns[batch_indices]).float().to(self.device)
                batch_advantages = torch.from_numpy(advantages[batch_indices]).float().to(self.device)
                
                action_probs, values = self.network(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                
                critic_loss = ((values.squeeze() - batch_returns) ** 2).mean()
                
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save(self.network.state_dict(), filepath)
        logger.info(f"RL agent saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"RL agent loaded from {filepath}")


# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

class SentimentEngine:
    """Aggregates market sentiment from multiple sources."""
    
    def __init__(self, newsapi_key: Optional[str] = None, twitter_bearer_token: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.twitter_bearer_token = twitter_bearer_token
        self.vader = SentimentIntensityAnalyzer()
    
    def get_news_sentiment(self, symbol: str) -> float:
        """Get sentiment from news articles."""
        if not self.newsapi_key:
            return 0.0
        
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&apiKey={self.newsapi_key}"
            response = requests.get(url, timeout=5)
            articles = response.json().get('articles', [])
            
            if not articles:
                return 0.0
            
            sentiments = []
            for article in articles[:10]:
                text = article.get('title', '') + ' ' + article.get('description', '')
                scores = self.vader.polarity_scores(text)
                sentiments.append(scores['compound'])
            
            return np.mean(sentiments) if sentiments else 0.0
        
        except Exception as e:
            logger.warning(f"Error getting news sentiment for {symbol}: {e}")
            return 0.0
    
    def get_market_sentiment(self, volume: float, volatility: float, price_change: float) -> float:
        """Calculate market sentiment from technical metrics."""
        sentiment = np.tanh(price_change * 10)
        volume_factor = min(volume / 1e7, 1.0)
        volatility_factor = -min(volatility * 10, 0.5)
        
        combined = sentiment * volume_factor + volatility_factor
        return np.clip(combined, -1, 1)
    
    def get_combined_sentiment(self, symbol: str, market_data: Dict) -> float:
        """Get combined sentiment from all sources."""
        news_sentiment = self.get_news_sentiment(symbol)
        
        volume = market_data.get('volume', 0)
        volatility = market_data.get('volatility', 0.02)
        price_change = (market_data.get('last', 0) - market_data.get('open', market_data.get('last', 0))) / max(market_data.get('open', 1), 1)
        
        market_sentiment = self.get_market_sentiment(volume, volatility, price_change)
        
        combined = (0.4 * news_sentiment + 0.6 * market_sentiment)
        return np.clip(combined, -1, 1)


# ============================================================================
# INTERACTIVE BROKERS CONNECTOR
# ============================================================================

class IBKRConnector:
    """Manages connection to Interactive Brokers and executes trades."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1, account_id: str = None):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account_id = account_id
        
        self.ib = IB()
        self.connected = False
        
        self.market_data = {}
        self.positions = {}
        self.orders = {}
    
    async def connect(self) -> bool:
        """Connect to Interactive Brokers."""
        try:
            logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")
            
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )
            
            self.connected = True
            logger.info("Connected to IBKR successfully")
            
            if not self.account_id:
                accounts = self.ib.managedAccounts()
                if accounts:
                    self.account_id = accounts[0]
                    logger.info(f"Using account: {self.account_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
    
    def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols."""
        if not self.connected:
            logger.warning("Not connected to IBKR")
            return
        
        try:
            for symbol in symbols:
                contract = Stock(symbol, 'SMART', 'USD')
                ticker = self.ib.reqMktData(contract)
                self.market_data[symbol] = ticker
                logger.info(f"Subscribed to market data for {symbol}")
        
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get current market data for a symbol."""
        if symbol not in self.market_data:
            logger.warning(f"No market data for {symbol}")
            return {}
        
        ticker = self.market_data[symbol]
        
        return {
            'symbol': symbol,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'last': ticker.last,
            'volume': ticker.volume,
            'open': ticker.open,
            'high': ticker.high,
            'low': ticker.low,
            'close': ticker.close,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        if not self.connected or not self.account_id:
            return 0.0
        
        try:
            account_values = self.ib.accountValues()
            for value in account_values:
                if value.account == self.account_id and value.tag == 'NetLiquidation':
                    return float(value.value)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
        
        return 0.0
    
    def get_cash_balance(self) -> float:
        """Get available cash balance."""
        if not self.connected or not self.account_id:
            return 0.0
        
        try:
            account_values = self.ib.accountValues()
            for value in account_values:
                if value.account == self.account_id and value.tag == 'AvailableFunds':
                    return float(value.value)
        except Exception as e:
            logger.error(f"Error getting cash balance: {e}")
        
        return 0.0
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        if not self.connected:
            return {}
        
        try:
            positions = {}
            for position in self.ib.positions():
                if position.account == self.account_id:
                    symbol = position.contract.symbol
                    positions[symbol] = {
                        'symbol': symbol,
                        'quantity': position.position,
                        'avg_cost': position.avgCost,
                        'market_price': self.get_market_data(symbol).get('last', 0),
                        'market_value': position.position * position.avgCost,
                        'unrealized_pnl': position.unrealizedPNL
                    }
            
            self.positions = positions
            return positions
        
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def place_market_order(self, symbol: str, quantity: int, action: str = 'BUY') -> Optional[str]:
        """Place a market order."""
        if not self.connected or not self.account_id:
            logger.warning("Not connected to IBKR")
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder(action, quantity)
            
            trade = self.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            logger.info(f"Placed {action} order for {quantity} shares of {symbol} (ID: {order_id})")
            
            self.orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'action': action,
                'order_type': 'MARKET',
                'timestamp': datetime.now().isoformat()
            }
            
            return str(order_id)
        
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, quantity: int, limit_price: float, action: str = 'BUY') -> Optional[str]:
        """Place a limit order."""
        if not self.connected or not self.account_id:
            logger.warning("Not connected to IBKR")
            return None
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            order = LimitOrder(action, quantity, limit_price)
            
            trade = self.ib.placeOrder(contract, order)
            order_id = trade.order.orderId
            
            logger.info(f"Placed {action} limit order for {quantity} shares of {symbol} at ${limit_price} (ID: {order_id})")
            
            self.orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'limit_price': limit_price,
                'action': action,
                'order_type': 'LIMIT',
                'timestamp': datetime.now().isoformat()
            }
            
            return str(order_id)
        
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.connected:
            return False
        
        try:
            self.ib.cancelOrder(int(order_id))
            logger.info(f"Cancelled order {order_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()
        
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if now.weekday() >= 5:
            return False
        
        return market_open <= now.time() <= market_close


class OrderManager:
    """Manages order execution and tracking."""
    
    def __init__(self, connector: IBKRConnector):
        self.connector = connector
        self.order_history = []
        self.active_orders = {}
    
    def buy(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> Optional[str]:
        """Execute a buy order."""
        if limit_price:
            order_id = self.connector.place_limit_order(symbol, quantity, limit_price, 'BUY')
        else:
            order_id = self.connector.place_market_order(symbol, quantity, 'BUY')
        
        if order_id:
            self.active_orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'action': 'BUY',
                'timestamp': datetime.now()
            }
        
        return order_id
    
    def sell(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> Optional[str]:
        """Execute a sell order."""
        if limit_price:
            order_id = self.connector.place_limit_order(symbol, quantity, limit_price, 'SELL')
        else:
            order_id = self.connector.place_market_order(symbol, quantity, 'SELL')
        
        if order_id:
            self.active_orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'action': 'SELL',
                'timestamp': datetime.now()
            }
        
        return order_id
    
    def close_position(self, symbol: str) -> Optional[str]:
        """Close a position by selling all shares."""
        positions = self.connector.get_positions()
        
        if symbol in positions:
            quantity = int(positions[symbol]['quantity'])
            if quantity > 0:
                return self.sell(symbol, quantity)
        
        return None


# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, symbols: List[str], config_file: str = 'config.json',
                 models_dir: str = './models', logs_dir: str = './logs'):
        self.symbols = symbols
        self.config_file = config_file
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        self.config = self._load_config()
        
        # Initialize data components
        self.data_pipeline = DataPipeline(symbols, lookback_days=365)
        self.real_time_buffer = RealTimeDataBuffer(symbols, buffer_size=30)
        self.sentiment_buffer = SentimentBuffer(symbols, window_size=30)
        self.online_learning_buffer = OnlineLearningBuffer(max_size=1000)
        
        # Initialize sentiment engine
        self.sentiment_engine = SentimentEngine(
            newsapi_key=self.config.get('newsapi_key'),
            twitter_bearer_token=self.config.get('twitter_bearer_token')
        )
        
        # Initialize models
        self.price_predictor = PricePredictor()
        self.price_predictor.build_model(sequence_length=30, num_features=5, sentiment_dim=30)
        
        self.rl_agent = PPOAgent(state_dim=32)
        
        # Initialize IBKR connector
        self.ibkr = IBKRConnector(
            host=self.config.get('ibkr_host', '127.0.0.1'),
            port=self.config.get('ibkr_port', 7497),
            client_id=self.config.get('ibkr_client_id', 1)
        )
        self.order_manager = OrderManager(self.ibkr)
        
        # Trading state
        self.is_running = False
        self.last_trade_time = {}
        self.trade_log = []
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        
        return {
            'ibkr_host': '127.0.0.1',
            'ibkr_port': 7497,
            'ibkr_client_id': 1,
            'max_position_size': 0.3,
            'max_total_exposure': 0.9,
            'stop_loss_percent': 0.05,
            'take_profit_percent': 0.10,
            'min_trade_interval': 300,
            'trading_enabled': False,
            'trading_start_hour': 9,
            'trading_end_hour': 16,
            'newsapi_key': None,
            'twitter_bearer_token': None,
            'confidence_threshold': 0.6,
            'initial_balance': 100000
        }
    
    async def initialize(self) -> bool:
        """Initialize the trading bot."""
        logger.info("Initializing trading bot...")
        
        try:
            if not await self.ibkr.connect():
                logger.error("Failed to connect to IBKR")
                return False
            
            if os.path.exists(os.path.join(self.models_dir, 'price_predictor_initial.pt')):
                self.price_predictor.load_model(os.path.join(self.models_dir, 'price_predictor_initial.pt'))
                logger.info("Loaded existing price predictor model")
            
            logger.info("Subscribing to market data...")
            self.ibkr.subscribe_market_data(self.symbols)
            
            logger.info("Trading bot initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    async def run(self):
        """Main trading loop."""
        if not await self.initialize():
            logger.error("Failed to initialize trading bot")
            return
        
        self.is_running = True
        logger.info("Trading bot started")
        
        try:
            while self.is_running:
                try:
                    if not self.ibkr.is_market_open():
                        logger.info("Market is closed, waiting...")
                        await asyncio.sleep(3600)
                        continue
                    
                    if not self.config.get('trading_enabled', False):
                        logger.info("Trading is disabled")
                        await asyncio.sleep(300)
                        continue
                    
                    await self._trading_cycle()
                    await asyncio.sleep(60)
                
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Trading bot interrupted by user")
        
        finally:
            await self.shutdown()
    
    async def _trading_cycle(self):
        """Execute one trading cycle."""
        try:
            market_data = {}
            for symbol in self.symbols:
                data = self.ibkr.get_market_data(symbol)
                if data:
                    market_data[symbol] = data
            
            if not market_data:
                logger.warning("No market data available")
                return
            
            for symbol in self.symbols:
                await self._make_trading_decision(symbol, market_data.get(symbol, {}))
        
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _make_trading_decision(self, symbol: str, market_data: Dict):
        """Make trading decision for a symbol."""
        try:
            price_sequence = self.real_time_buffer.get_buffer_array(symbol)
            if price_sequence is None:
                return
            
            sentiment_sequence = self.sentiment_buffer.get_sentiment_array(symbol)
            
            state = np.concatenate([
                sentiment_sequence,
                np.array([
                    self.ibkr.get_cash_balance() / self.config.get('initial_balance', 100000),
                    self.ibkr.get_portfolio_value() / self.config.get('initial_balance', 100000)
                ])
            ]).astype(np.float32)
            
            action, _ = self.rl_agent.select_action(state, deterministic=False)
            
            price_pred = self.price_predictor.predict(
                price_sequence.reshape(1, 30, 5),
                sentiment_sequence.reshape(1, 30)
            )[0, 0]
            
            sentiment_score = self.sentiment_buffer.get_current_sentiment(symbol)
            confidence = abs(sentiment_score) * 0.3
            
            if confidence > self.config.get('confidence_threshold', 0.6):
                await self._execute_trade(symbol, action, market_data, confidence)
            
            next_price = market_data.get('last', 0)
            reward = (next_price - market_data.get('open', next_price)) / max(market_data.get('open', 1), 1)
            
            self.online_learning_buffer.add_sample(
                price_sequence,
                sentiment_sequence,
                next_price,
                action,
                reward,
                datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Error making trading decision for {symbol}: {e}")
    
    async def _execute_trade(self, symbol: str, action: int, market_data: Dict, confidence: float):
        """Execute a trade."""
        try:
            last_trade = self.last_trade_time.get(symbol, datetime.now() - timedelta(hours=1))
            if (datetime.now() - last_trade).total_seconds() < self.config.get('min_trade_interval', 300):
                return
            
            portfolio_value = self.ibkr.get_portfolio_value()
            max_position_value = portfolio_value * self.config.get('max_position_size', 0.3)
            
            current_price = market_data.get('last', 0)
            if current_price <= 0:
                return
            
            if action == 1:  # Buy
                shares = int(max_position_value / current_price)
                
                if shares > 0:
                    order_id = self.order_manager.buy(symbol, shares)
                    
                    if order_id:
                        logger.info(f"BUY {shares} shares of {symbol} at ${current_price} (Confidence: {confidence:.2%})")
                        self.last_trade_time[symbol] = datetime.now()
                        self.trade_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price,
                            'confidence': confidence,
                            'order_id': order_id
                        })
            
            elif action == 2:  # Sell
                order_id = self.order_manager.close_position(symbol)
                
                if order_id:
                    logger.info(f"SELL position in {symbol} (Confidence: {confidence:.2%})")
                    self.last_trade_time[symbol] = datetime.now()
                    self.trade_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': 'SELL',
                        'confidence': confidence,
                        'order_id': order_id
                    })
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def shutdown(self):
        """Shutdown the trading bot."""
        logger.info("Shutting down trading bot...")
        
        self.is_running = False
        
        log_file = os.path.join(self.logs_dir, f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(log_file, 'w') as f:
            json.dump(self.trade_log, f, indent=2)
        
        self.price_predictor.save_model(os.path.join(self.models_dir, 'price_predictor_final.pt'))
        self.rl_agent.save_model(os.path.join(self.models_dir, 'rl_agent_final.pt'))
        
        self.ibkr.disconnect()
        
        logger.info("Trading bot shutdown complete")


async def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("IBKR RL Trading Bot - Starting")
    logger.info("=" * 80)
    
    symbols = ['AVGO', 'TSLA', 'MU', 'COST', 'ABBV', 'NFLX', 'INTC', 'IBM']
    logger.info(f"Trading symbols: {', '.join(symbols)}")
    
    bot = TradingBot(symbols)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Trading bot stopped")


if __name__ == '__main__':
    asyncio.run(main())
