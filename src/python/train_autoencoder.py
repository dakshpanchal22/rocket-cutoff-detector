import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# 0. Paths (always relative to project root)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'rocket_data.npz')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
data = np.load(DATA_FILE, allow_pickle=True)
thrust_list = data['thrust']        # array of arrays (variable length)
pressure_list = data['pressure']
fuel_flow_list = data['fuel_flow']
labels_curve = data['labels']       # 0 = normal, 1 = anomaly

print(f"Total curves loaded: {len(thrust_list)}")
print(f"Normal: {np.sum(labels_curve==0)}, Anomalous: {np.sum(labels_curve==1)}")

# ------------------------------------------------------------
# 2. Windowing parameters
# ------------------------------------------------------------
WINDOW_SIZE = 80     # time steps per window
STRIDE = 40          # slide by 40 steps

# ------------------------------------------------------------
# 3. Convert curves to fixed‑size windows
# ------------------------------------------------------------
windows = []
labels_window = []

for i, (th, pr, ff) in enumerate(zip(thrust_list, pressure_list, fuel_flow_list)):
    length = len(th)
    if length < WINDOW_SIZE:
        continue   # skip curves that are too short

    # Combine the three channels into a single array (length x 3)
    curve = np.column_stack((th, pr, ff))   # shape (length, 3)

    # Slide a window
    for start in range(0, length - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window = curve[start:end]           # shape (WINDOW_SIZE, 3)
        windows.append(window.flatten())    # flatten to 1D (240,)
        labels_window.append(labels_curve[i])

windows = np.array(windows)        # shape (num_windows, 240)
labels_window = np.array(labels_window)  # shape (num_windows,)

print(f"\nTotal windows: {len(windows)}")
print(f"Normal windows: {np.sum(labels_window==0)}, Anomalous: {np.sum(labels_window==1)}")

# ------------------------------------------------------------
# 4. Train / validation / test split
# ------------------------------------------------------------
# Train: normal windows only
normal_idx = np.where(labels_window == 0)[0]
np.random.seed(42)   # reproducibility
np.random.shuffle(normal_idx)
split = int(0.8 * len(normal_idx))
train_idx = normal_idx[:split]         # 80% of normal
val_idx = normal_idx[split:]           # 20% of normal

# Test: all anomalous windows + the normal validation windows
anom_idx = np.where(labels_window == 1)[0]
test_idx = np.concatenate([val_idx, anom_idx])

X_train = windows[train_idx]
X_val = windows[val_idx]
X_test = windows[test_idx]
y_test = labels_window[test_idx]      # only needed for evaluation

print(f"Train windows (normal): {len(X_train)}")
print(f"Val windows (normal): {len(X_val)}")
print(f"Test windows (mixed): {len(X_test)} (includes {np.sum(y_test==1)} anomalies)")

# ------------------------------------------------------------
# 5. PyTorch tensors and data loaders
# ------------------------------------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t), batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------
# 6. Define the autoencoder
# ------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=240, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)   # no activation on output (linear)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
print(f"\nUsing device: {device}")
print(model)

# ------------------------------------------------------------
# 7. Training loop
# ------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 150

train_losses = []
val_losses = []

for epoch in range(1, epochs+1):
    # Training
    model.train()
    train_loss = 0.0
    for (batch_x,) in train_loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
    train_loss /= len(X_train_t)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_rec = model(X_val_t.to(device))
        val_loss = criterion(val_rec, X_val_t.to(device)).item()
    val_losses.append(val_loss)

    if epoch % 30 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

print("Training complete.")

# ------------------------------------------------------------
# 8. Reconstruction error on test set
# ------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t.to(device))
    # MSE per window
    mse_per_window = torch.mean((test_pred - X_test_t.to(device))**2, dim=1).cpu().numpy()

# ------------------------------------------------------------
# 9. Threshold from training errors
# ------------------------------------------------------------
with torch.no_grad():
    train_pred = model(X_train_t.to(device))
    train_mse = torch.mean((train_pred - X_train_t.to(device))**2, dim=1).cpu().numpy()

threshold = np.percentile(train_mse, 95)
print(f"\n95th percentile training MSE (threshold): {threshold:.6f}")

# ------------------------------------------------------------
# 10. Evaluate
# ------------------------------------------------------------
pred_anomalies = (mse_per_window > threshold).astype(int)
true_anomalies = y_test.astype(int)

correct = np.sum((pred_anomalies == 1) & (true_anomalies == 1))
recall = correct / np.sum(true_anomalies) if np.sum(true_anomalies) > 0 else 0
precision = correct / np.sum(pred_anomalies) if np.sum(pred_anomalies) > 0 else 0
print(f"True anomalies in test: {np.sum(true_anomalies)}")
print(f"Predicted anomalies:    {np.sum(pred_anomalies)}")
print(f"Correctly caught:       {correct}")
print(f"Recall:    {recall:.2%}")
print(f"Precision: {precision:.2%}")

# ------------------------------------------------------------
# 11. Plot reconstruction error
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))
normal_mask = (y_test == 0)
anom_mask = (y_test == 1)
plt.plot(np.arange(len(mse_per_window))[normal_mask], mse_per_window[normal_mask],
         'bo', markersize=3, alpha=0.6, label='Normal window')
plt.plot(np.arange(len(mse_per_window))[anom_mask], mse_per_window[anom_mask],
         'ro', markersize=4, label='Anomaly window')
plt.axhline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.4f})')
plt.xlabel('Test window index')
plt.ylabel('Reconstruction MSE')
plt.title('Autoencoder Anomaly Detection – Rocket Engine')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'autoencoder_anomaly_detection.png'), dpi=150)
plt.show()

# ------------------------------------------------------------
# 12. Save the model (PyTorch format)
# ------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, 'autoencoder.pth')
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")