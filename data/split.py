import numpy as np

def split_patients(patient_dirs, ssl_frac=0.8, train_frac=0.1, val_frac = 0.05, seed=42):
    rng = np.random.default_rng(seed)
    patient_dirs = list(patient_dirs)
    rng.shuffle(patient_dirs)

    n = len(patient_dirs)
    n_ssl = int(n * ssl_frac)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    ssl   = patient_dirs[:n_ssl]
    train = patient_dirs[n_ssl : n_ssl + n_train]
    val   = patient_dirs[n_ssl + n_train : n_ssl + n_train + n_val]
    test  = patient_dirs[n_ssl + n_train + n_val :]

    return ssl, train, val, test