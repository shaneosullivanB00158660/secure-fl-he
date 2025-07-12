import numpy as np
import phe
import time

USE_HE = True  # Toggle homomorphic encryption
NUM_CLIENTS = 3
VECTOR_SIZE = 10
ROUNDS = 3

if USE_HE:
    # Generate Paillier keys
    pubkey, privkey = phe.paillier.generate_paillier_keypair()
    def encrypt(vec):
        # Encrypt a vector using Paillier
        return [pubkey.encrypt(float(x)) for x in vec]
    def decrypt(vec):
        # Decrypt a vector using Paillier
        return np.array([privkey.decrypt(x) for x in vec], dtype=np.float32)
else:
    def encrypt(vec):
        # No encryption, return the input as is
        return vec
    def decrypt(vec):
        # No decryption, return as NumPy array
        return np.array(vec)

# Initialise the global model
global_model = np.zeros(VECTOR_SIZE, dtype=np.float32)

for rnd in range(ROUNDS):
    start_time = time.time()
    updates = []

    # Encrypt updates
    t_enc_start = time.time()
    for _ in range(NUM_CLIENTS):
        # Each client makes a small change to the global model
        local_model = global_model + np.random.randn(VECTOR_SIZE) * 0.1
        update = local_model - global_model
        updates.append(encrypt(update))
    t_enc = time.time() - t_enc_start

    # Aggregate updates
    t_agg_start = time.time()
    if USE_HE:
        # Aggregate encrypted updates elementwise
        agg = []
        for elements in zip(*updates):
            s = elements[0]
            for e in elements[1:]:
                s += e
            agg.append(s / NUM_CLIENTS)
    else:
        # Average updates in plain text
        agg = np.mean(updates, axis=0)
    t_agg = time.time() - t_agg_start

    # Decrypt aggregate
    t_dec_start = time.time()
    if USE_HE:
        avg_update = decrypt(agg)
    else:
        avg_update = agg
    t_dec = time.time() - t_dec_start

    # Update the global model
    global_model += avg_update
    total_time = time.time() - start_time

    print(
        f"Round {rnd+1}: global_model[:3] = {global_model[:3]} | "
        f"Enc: {t_enc:.4f}s, Agg: {t_agg:.4f}s, Dec: {t_dec:.4f}s, Total: {total_time:.4f}s"
    )
