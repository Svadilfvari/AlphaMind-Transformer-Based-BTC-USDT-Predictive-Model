import os
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import time
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def validate_model(model, val_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu())
            trues.append(yb.cpu())
    y_true = torch.cat(trues).numpy()
    y_pred = torch.cat(preds).numpy()
    mse = mean_squared_error(y_true, y_pred)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    acc = (direction_true == direction_pred).mean()
    return mse, acc
# --- Fetch Binance OHLCV ---
def get_binance_klines(symbol, interval='1h', limit=1000, lookback_days=1095):
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - lookback_days * 24 * 60 * 60 * 1000
    all_data = []
    print(f"Fetching data for {symbol}...")
    with tqdm(total=(lookback_days * 24) // limit + 1, desc=f"Fetching {symbol}") as pbar:
        while start_time < end_time:
            params = {'symbol': symbol, 'interval': interval, 'startTime': start_time, 'limit': limit}
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    break
                all_data.extend(data)
                last_time = data[-1][0]
                start_time = last_time + 1
                time.sleep(0.5)
                pbar.update(1)
            else:
                print(f"Error fetching {symbol}: {response.status_code}")
                break
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    print(f"Completed fetching {symbol}, total candles: {len(df)}")
    return df

# --- Multi-Asset Support ---
symbols = {'bitcoin': 'BTCUSDT'}

market_data = {}
if os.path.exists('BTCUSDT_cached.csv'):
    print("✅ Loading cached dataset...")
    data = pd.read_csv('BTCUSDT_cached.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
else:
    print("🚀 No cache found. Fetching...")
    df = get_binance_klines('BTCUSDT')
    df.to_csv('BTCUSDT_cached.csv')
    data = df

# --- Feature Engineering ---
data['returns'] = np.log(data['close'] / data['close'].shift(1))
data['future_returns'] = np.log(data['close'].shift(-6) / data['close'])  # 6 hours ahead

# Simple Moving Averages and indicators
data['ma10'] = data['close'].rolling(10).mean()
data['ma50'] = data['close'].rolling(50).mean()
data['volatility_10d'] = data['returns'].rolling(10).std()

data.dropna(inplace=True)

features = ['close', 'volume', 'ma10', 'ma50', 'volatility_10d']
def add_technical_features(df):
    # Lagged returns
    for lag in [1, 2, 3, 6, 12, 24]:  # 1h, 2h, 3h, 6h, 12h, 24h
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)

    # RSI (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12-26 EMA difference)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    # Bollinger Bands width (20-period)
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bollinger_width'] = (2 * std20) / ma20

    # High-Low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']

    return df

data = add_technical_features(data)
scaler = MinMaxScaler()
X = scaler.fit_transform(data[features])
y = data['future_returns'].values
def create_sequences(features, targets, seq_len=24):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i+seq_len])
        y_seq.append(targets[i+seq_len])  # predict after the sequence
    return np.array(X_seq), np.array(y_seq)

seq_len = 24  # past 24 hours
X_seq, y_seq = create_sequences(X, y, seq_len)
print("X_seq shape:", X_seq.shape)  # (samples, seq_len, features)
print("y_seq shape:", y_seq.shape)  # (samples,)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
# --- Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
               # Register as buffer so it moves automatically with .to(device)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerSeqModel(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=500)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        x = self.embedding(x)         # -> [batch, seq_len, hidden_dim]
        x = self.pos_encoder(x)       # add positional info
        x = self.transformer(x)       # -> [batch, seq_len, hidden_dim]
        x = self.fc_out(x[:, -1, :])  # use last timestep representation
        return x.squeeze()

# Instantiate
model = TransformerSeqModel(num_features=X_seq.shape[-1]).to(device)

# --- Training ---
def train_model(model, train_loader, val_loader, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    best_val_acc = 0.0   # or float("inf") if you want best MSE
    best_epoch = -1

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        all_trues = []
        all_preds = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            all_preds.append(pred.detach().cpu())
            all_trues.append(yb.detach().cpu())
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Training metrics
        y_true = torch.cat(all_trues).numpy()
        y_pred = torch.cat(all_preds).numpy()
        train_acc = (np.sign(y_true) == np.sign(y_pred)).mean()
        train_corr = np.corrcoef(y_true, y_pred)[0,1]

        # ✅ Validate model
        val_mse, val_acc = validate_model(model, val_loader)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss/len(train_loader):.6f} | "
              f"Train Acc={train_acc:.2%} | Train Corr={train_corr:.3f} | "
              f"Val MSE={val_mse:.6f} | Val Acc={val_acc:.2%}")

        # ✅ Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ New best model saved at epoch {best_epoch} (Val Acc={best_val_acc:.2%})")

    print(f"🎯 Best model was at epoch {best_epoch} with Val Acc={best_val_acc:.2%}")
train_model(model, train_loader, val_loader)

# --- Evaluate and Plot ---
def evaluate_and_plot_returns(model, val_loader, save_dir="results"):
    # ✅ Create output folder
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu())
            trues.append(yb.cpu())

    # ✅ Concatenate properly
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()

    # ✅ Compute metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    accuracy = (direction_true == direction_pred).mean()
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    # ✅ Print metrics
    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation R² Score: {r2:.4f}")
    print(f"Directional Accuracy: {accuracy:.2%}")
    print(f"Return Correlation: {corr:.4f}")

    # ✅ Save metrics as JSON
    metrics = {
        "Validation_MSE": float(mse),
        "Validation_R2": float(r2),
        "Directional_Accuracy": float(accuracy),
        "Return_Correlation": float(corr)
    }
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved to {metrics_path}")

    # === Plot 1: Full True vs Predicted ===
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='True Returns (6h)', color='blue')
    plt.plot(y_pred, label='Predicted Returns (6h)', color='orange')
    plt.title('True vs Predicted Returns (6h ahead)')
    plt.xlabel('Time Steps')
    plt.ylabel('Return')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "true_vs_predicted_full.png"), dpi=300)
    plt.show()

    # === Plot 2: Sample (first 500 points) ===
    plt.figure(figsize=(15,5))
    plt.plot(y_true[:500], label="True Returns", color="blue", alpha=0.7)
    plt.plot(y_pred[:500], label="Predicted Returns", color="orange", alpha=0.7)
    plt.title("True vs Predicted 6h Returns (sample of 500)")
    plt.xlabel("Time steps")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "true_vs_predicted_sample.png"), dpi=300)
    plt.show()

    # === Plot 3: Rolling Directional Accuracy ===
    window = 200
    rolling_accuracy = [
        (np.sign(y_true[i:i+window]) == np.sign(y_pred[i:i+window])).mean()
        for i in range(len(y_true)-window)
    ]
    plt.figure(figsize=(12,4))
    plt.plot(rolling_accuracy, color="green")
    plt.axhline(0.5, color="red", linestyle="--", label="Random baseline")
    plt.title("Rolling Directional Accuracy (window=200)")
    plt.xlabel("Time steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "rolling_directional_accuracy.png"), dpi=300)
    plt.show()

    # === Plot 4: Scatter Plot ===
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.xlabel("True Returns")
    plt.ylabel("Predicted Returns")
    plt.title(f"True vs Predicted (corr={corr:.3f})")
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "scatter_true_vs_pred.png"), dpi=300)
    plt.show()

    # === Plot 5: Cumulative Strategy PnL ===
    strategy_returns = np.sign(y_pred) * y_true  # long if predicted >0, short if <0
    cum_pnl = np.cumsum(strategy_returns)
    cum_benchmark = np.cumsum(y_true)

    plt.figure(figsize=(10,5))
    plt.plot(cum_pnl, label="Strategy PnL")
    plt.plot(cum_benchmark, label="Buy & Hold")
    plt.legend()
    plt.title("Cumulative Strategy PnL vs Buy & Hold")
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "cumulative_pnl_vs_buyhold.png"), dpi=300)
    plt.show()

    print(f"✅ All plots saved to {plot_dir}")


evaluate_and_plot_returns(model, val_loader)
