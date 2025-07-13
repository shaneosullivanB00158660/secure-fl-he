import time
import numpy as np
from main import generate_keys, federated_round

NUM_CLIENTS = 3
VECTOR_SIZE = 10
ROUNDS = 3

def run_benchmark(use_he):
    if use_he:
        pubkey, privkey = generate_keys()
    else:
        pubkey = privkey = None

    global_model = np.zeros(VECTOR_SIZE, dtype=np.float32)
    times = []

    for _ in range(ROUNDS):
        start = time.time()
        global_model = federated_round(
            global_model, NUM_CLIENTS, VECTOR_SIZE, use_he, pubkey, privkey
        )
        times.append(time.time() - start)
    return np.mean(times)

if __name__ == "__main__":
    t_plain = run_benchmark(use_he=False)
    print(f"Average time per round without HE: {t_plain:.4f}s")

    t_he = run_benchmark(use_he=True)
    print(f"Average time per round with HE: {t_he:.4f}s")
