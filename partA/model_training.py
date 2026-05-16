import torch
import torch.nn as nn
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# DATASET   = "train_64_cp.mat"
# X_KEY     = "X_64_cp"
# Y_KEY     = "Y_64_cp"
# MODEL_OUT = "ofdm_models_64.pth"
# MEAN_OUT  = "norm_mean_64.npy"
# STD_OUT   = "norm_std_64.npy"

# DATASET   = "train_8_cp.mat"
# X_KEY     = "X_8_cp"
# Y_KEY     = "Y_8_cp"
# MODEL_OUT = "ofdm_models_8.pth"
# MEAN_OUT  = "norm_mean_8.npy"
# STD_OUT   = "norm_std_8.npy"

DATASET   = "train_64_noCP.mat"
X_KEY     = "X_64_noCP"
Y_KEY     = "Y_64_noCP"
MODEL_OUT = "ofdm_models_noCP.pth"
MEAN_OUT  = "norm_mean_noCP.npy"
STD_OUT   = "norm_std_noCP.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

with h5py.File(DATASET, 'r') as f:
    X_train = np.array(f[X_KEY])
    Y_train = np.array(f[Y_KEY])


if X_train.shape[0] == 256:
    X_train = X_train.T
if Y_train.shape[0] == 128:
    Y_train = Y_train.T

print("Train shape:", X_train.shape, Y_train.shape)

mean = np.mean(X_train, axis=0, keepdims=True)
std = np.std(X_train, axis=0, keepdims=True) + 1e-8

X_train = (X_train - mean) / std

np.save(MEAN_OUT, mean)
np.save(STD_OUT, std)

X_tr, X_val, Y_tr, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=42
)

X_tr = torch.tensor(X_tr, dtype=torch.float32)
Y_tr = torch.tensor(Y_tr, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)

class OFDMNet16(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 120),
            nn.ReLU(),
            nn.Linear(120, 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

num_chunks = 8
models = []

print("\n Training 8 chunk models...\n")

for chunk_id in range(num_chunks):

    print(f"\n====== Model {chunk_id+1}/8 ======\n")

    model = OFDMNet16().to(device)

    if chunk_id == 0:
        train_losses = []
        val_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    Y_chunk_tr = Y_tr[:, chunk_id*16:(chunk_id+1)*16]
    Y_chunk_val = Y_val[:, chunk_id*16:(chunk_id+1)*16]

    dataset = torch.utils.data.TensorDataset(X_tr, Y_chunk_tr)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    epochs = 100
    patience = 30
    counter = 0

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(loader)

        if chunk_id == 0:
            train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            X_val_gpu = X_val.to(device)
            Y_val_gpu = Y_chunk_val.to(device)

            val_out = model(X_val_gpu)
            val_loss = criterion(val_out, Y_val_gpu).item()

        if chunk_id == 0:
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"⏹ Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    if chunk_id == 0:
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve (Model 1)')
        plt.legend()
        plt.grid()
        plt.show()

    models.append(model)

torch.save([m.state_dict() for m in models], MODEL_OUT)

print(f"\nTraining complete. Models saved to {MODEL_OUT}")