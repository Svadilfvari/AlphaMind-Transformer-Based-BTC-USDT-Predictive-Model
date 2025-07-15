from alphamind.data import load_or_fetch
from alphamind.features_engineering import add_basic_features, add_technical_features
from alphamind.dataset import prepare_dataset
from alphamind.model import TransformerSeqModel
from alphamind.train import train_model
from alphamind.evaluate import evaluate_and_plot_returns
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 1. Load data
df = load_or_fetch()
print("\n=== CHECKPOINT 1: RAW DATA ===")
print("NaNs in raw data:")
print(df.isna().sum())

# ✅ 2. Add features
df = add_basic_features(df)
df['future_returns'] = (df['close'].shift(-6)/df['close']).apply(lambda x: torch.log(torch.tensor(x)))
df = df.dropna()  # now also drops last 6 NaNs
df = add_technical_features(df)
# ✅ Drop NaNs from indicators + drop last 6 rows with missing future_returns
df = df.dropna()
df = df.iloc[:-6]   # <-- Drop last 6 rows explicitly
print("\n=== CHECKPOINT 2: AFTER FEATURE ENGINEERING ===")
print("NaNs after adding indicators:")
print(df.isna().sum())

# ✅ Drop rows with NaNs introduced by rolling windows
df = df.dropna()
print("NaNs after cleaning:", df.isna().sum().sum())  # should be 0

# ✅ 3. Prepare dataset
features = ['close', 'volume', 'ma10', 'ma50', 'volatility_10d']
dataset, scaler = prepare_dataset(df, features)

# Check dataset tensor for NaNs
X_check = dataset.tensors[0].numpy()
y_check = dataset.tensors[1].numpy()
print("\n=== CHECKPOINT 3: AFTER SCALING + SEQUENCES ===")
print("NaNs in X_seq:", np.isnan(X_check).sum())
print("NaNs in y_seq:", np.isnan(y_check).sum())

# ✅ 4. Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# ✅ 5. Instantiate model
model = TransformerSeqModel(num_features=len(features)).to(device)

# ✅ Add debug hook inside train_model to catch NaNs mid-training
def debug_nan_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"⚠️ NaNs detected in layer {module.__class__.__name__}")
for layer in model.modules():
    if isinstance(layer, torch.nn.Linear):
        layer.register_forward_hook(debug_nan_hook)

# ✅ 6. Train with debug checks
train_model(model, train_loader, val_loader, epochs=1000)

# ✅ 7. Load best model for evaluation
model.load_state_dict(torch.load("best_model.pth"))

# ✅ 8. Evaluate (will print if NaNs happen in outputs)
evaluate_and_plot_returns(model, val_loader, save_dir="results")