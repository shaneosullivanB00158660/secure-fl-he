import numpy as np
import torch
import phe
import time

# Settings
USE_HE = True # Toggle encryption
NUM_CLIENTS = 3
ROUNDS = 5
N_FEATURES = 5
LR = 0.05
SAMPLES = 20

print(f"Starting demo | USE_HE={USE_HE}, CLIENTS={NUM_CLIENTS}, ROUNDS={ROUNDS}")

# Synthetic data per client
np.random.seed(42)
X_clients = [np.random.randn(SAMPLES, N_FEATURES) for _ in range(NUM_CLIENTS)]
true_weights = np.array([1.0, -2.0, 0.5, 0.1, 2.0])
y_clients = [X @ true_weights + np.random.randn(SAMPLES) * 0.1 for X in X_clients]

class LinReg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(N_FEATURES, 1, bias=False)
    def forward(self, x):
        return self.fc(x)

def rmse(model, Xs, ys):
    """Root mean squared error over all client data."""
    model.eval()
    X_all = np.vstack(Xs)
    y_all = np.hstack(ys)
    with torch.no_grad():
        preds = model(torch.tensor(X_all, dtype=torch.float32)).numpy().flatten()
    return np.sqrt(np.mean((preds - y_all) ** 2))

# Encryption setup
if USE_HE:
    pub, priv = phe.paillier.generate_paillier_keypair()
    def encrypt(vec): return [pub.encrypt(float(x)) for x in vec]
    def decrypt(vec): return np.array([priv.decrypt(x) for x in vec], dtype=np.float32)
else:
    def encrypt(vec): return vec
    def decrypt(vec): return np.array(vec)

def client_update(global_model, X, y):
    """Train on local data and return parameter delta."""
    model = LinReg(); model.load_state_dict(global_model.state_dict())
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    opt.zero_grad()
    out = model(X_t)
    loss = torch.nn.MSELoss()(out, y_t)
    loss.backward()
    opt.step()
    return (model.fc.weight.data - global_model.fc.weight.data).numpy().flatten()

global_model = LinReg()

for rnd in range(ROUNDS):
    t0 = time.time()

    # Each client trains and encrypts update
    t1 = time.time()
    updates = [encrypt(client_update(global_model, X_clients[i], y_clients[i])) for i in range(NUM_CLIENTS)]
    t2 = time.time()

    # Aggregate updates
    if USE_HE:
        agg = [sum(w) / NUM_CLIENTS for w in zip(*updates)]
    else:
        agg = np.mean(updates, axis=0)
    t3 = time.time()

    # Decrypt
    avg_update = decrypt(agg)
    t4 = time.time()

    # Update global model
    with torch.no_grad():
        global_model.fc.weight += torch.tensor(avg_update).reshape_as(global_model.fc.weight)

    error = rmse(global_model, X_clients, y_clients)

    print(
        f"Round {rnd+1}: RMSE={error:.4f} | "
        f"Enc: {t2-t1:.4f}s, Agg: {t3-t2:.4f}s, Dec: {t4-t3:.4f}s, Total: {t4-t0:.4f}s"
    )
