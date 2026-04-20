import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ================= MODEL =================
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

# ================= LOAD MODELS + NORMS =================
def load_models(model_path):
    states = torch.load(model_path, map_location=device, weights_only=True)
    models = []
    for i in range(8):
        model = OFDMNet16().to(device)
        model.load_state_dict(states[i])
        model.eval()
        models.append(model)
    return models

# Load all 3 models
models_64   = load_models("ofdm_models_64.pth")
models_8    = load_models("ofdm_models_8.pth")
models_noCP = load_models("ofdm_models_noCP.pth")

# Load normalization
mean_64 = np.load("norm_mean_64.npy")
std_64  = np.load("norm_std_64.npy")

mean_8 = np.load("norm_mean_8.npy")
std_8  = np.load("norm_std_8.npy")

mean_noCP = np.load("norm_mean_noCP.npy")
std_noCP  = np.load("norm_std_noCP.npy")

# ================= FUNCTION =================
def compute_dnn_ber(f, X_cell, Y_cell, snr_list, models, mean, std):

    ber = []

    for s in range(len(snr_list)):

        # MATLAB cell indexing
        X_ref = X_cell[0, s]
        Y_ref = Y_cell[0, s]

        X = np.array(f[X_ref])
        Y = np.array(f[Y_ref])

        # Fix orientation
        if X.shape[0] == 256:
            X = X.T
        if Y.shape[0] == 128:
            Y = Y.T

        # Normalize (CASE-SPECIFIC)
        X = (X - mean) / std

        X = torch.tensor(X, dtype=torch.float32).to(device)

        outputs = []

        with torch.no_grad():
            for i in range(8):
                out = models[i](X).cpu().numpy()
                outputs.append(out)

        pred = np.hstack(outputs)
        bits_pred = (pred > 0.5).astype(int)

        errors = np.sum(bits_pred != Y)
        ber.append(errors / Y.size)

    return np.array(ber)

# ================= MAIN =================
with h5py.File("test_all_data.mat", 'r') as f:

    snr_list = np.array(f['snr_list']).flatten()

    # 64 pilots
    X_test_64 = f['X_test_64']
    Y_test_64 = f['Y_test_64']
    BER_LS_64 = np.array(f['BER_LS_64']).flatten()
    BER_MMSE_64 = np.array(f['BER_MMSE_64']).flatten()

    # 8 pilots
    X_test_8 = f['X_test_8']
    Y_test_8 = f['Y_test_8']
    BER_LS_8 = np.array(f['BER_LS_8']).flatten()
    BER_MMSE_8 = np.array(f['BER_MMSE_8']).flatten()

    # no CP
    X_test_noCP = f['X_test_noCP']
    Y_test_noCP = f['Y_test_noCP']
    BER_LS_noCP = np.array(f['BER_LS_noCP']).flatten()
    BER_MMSE_noCP = np.array(f['BER_MMSE_noCP']).flatten()

    print("\nRunning DNN...\n")

    BER_DNN_64 = compute_dnn_ber(f, X_test_64, Y_test_64, snr_list,
                                models_64, mean_64, std_64)

    BER_DNN_8 = compute_dnn_ber(f, X_test_8, Y_test_8, snr_list,
                               models_8, mean_8, std_8)

    BER_DNN_noCP = compute_dnn_ber(f, X_test_noCP, Y_test_noCP, snr_list,
                                  models_noCP, mean_noCP, std_noCP)

# ================= PLOTS =================

# ---- 64 pilots ----
plt.figure()
plt.semilogy(snr_list, BER_LS_64, '-o', label='LS')
plt.semilogy(snr_list, BER_MMSE_64, '-s', label='MMSE')
plt.semilogy(snr_list, BER_DNN_64, '-^', label='DNN')
plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("64 Pilots (With CP)")
plt.legend()

# ---- 8 pilots ----
plt.figure()
plt.semilogy(snr_list, BER_LS_8, '-o', label='LS')
plt.semilogy(snr_list, BER_MMSE_8, '-s', label='MMSE')
plt.semilogy(snr_list, BER_DNN_8, '-^', label='DNN')
plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("8 Pilots (With CP)")
plt.legend()

# ---- CP vs No CP ----
plt.figure()

# LS
plt.semilogy(snr_list, BER_LS_64, '-o', label='LS CP')
plt.semilogy(snr_list, BER_LS_noCP, '--o', label='LS NoCP')

# MMSE
plt.semilogy(snr_list, BER_MMSE_64, '-s', label='MMSE CP')
plt.semilogy(snr_list, BER_MMSE_noCP, '--s', label='MMSE NoCP')

# DNN
plt.semilogy(snr_list, BER_DNN_64, '-^', label='DNN CP')
plt.semilogy(snr_list, BER_DNN_noCP, '--^', label='DNN NoCP')

plt.grid()
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("CP vs No CP (Full Comparison)")
plt.legend()

plt.show()