import numpy as np
import phe

def generate_keys():
    pubkey, privkey = phe.paillier.generate_paillier_keypair()
    return pubkey, privkey

def encrypt_vector(vec, pubkey):
    return [pubkey.encrypt(float(x)) for x in vec]

def decrypt_vector(vec, privkey):
    return np.array([privkey.decrypt(x) for x in vec], dtype=np.float32)

def federated_round(global_model, num_clients, vector_size, use_he, pubkey=None, privkey=None):
    updates = []
    for _ in range(num_clients):
        local_model = global_model + np.random.randn(vector_size) * 0.1
        update = local_model - global_model
        if use_he:
            updates.append(encrypt_vector(update, pubkey))
        else:
            updates.append(update)

    if use_he:
        agg = []
        for elements in zip(*updates):
            s = elements[0]
            for e in elements[1:]:
                s += e
            agg.append(s / num_clients)
        avg_update = decrypt_vector(agg, privkey)
    else:
        avg_update = np.mean(updates, axis=0)
    global_model += avg_update
    return global_model
