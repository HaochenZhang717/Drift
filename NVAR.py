import argparse
import itertools
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from aeon.datasets import load_classification


def to_numpy_time_series(X):
    """
    aeon usually returns shape:
        (N, C, T)
    We convert to:
        (N, T, C)
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 2:
        # (N, T) -> (N, T, 1)
        X = X[:, :, None]
    elif X.ndim == 3:
        # aeon: (N, C, T) -> (N, T, C)
        X = np.transpose(X, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported X shape: {X.shape}")

    return X


def standardize_per_series(X):
    """
    Standardize each individual time series to zero mean and unit variance.
    X shape: (N, T, C)
    """
    X_out = np.empty_like(X, dtype=np.float64)

    for i in range(X.shape[0]):
        mean = X[i].mean(axis=0, keepdims=True)
        std = X[i].std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X_out[i] = (X[i] - mean) / std

    return X_out


def make_feature_pool(dx, k):
    """
    Build candidate terms.

    Linear lag term:
        ("lag", a, p)
        means x^a_{t - p}

    Nonlinear product term:
        ("prod", a, p, b, q)
        means x^a_{t - p} * x^b_{t - q}

    Here p, q are lag indices. Actual lag is p * s.
    p = 0 means current time t.
    """
    lag_terms = []
    for a in range(dx):
        for p in range(1, k + 1):
            lag_terms.append(("lag", a, p))

    prod_terms = []
    base_terms = [(a, p) for a in range(dx) for p in range(0, k + 1)]

    for idx1, (a, p) in enumerate(base_terms):
        for idx2 in range(idx1, len(base_terms)):
            b, q = base_terms[idx2]
            prod_terms.append(("prod", a, p, b, q))

    return lag_terms + prod_terms


def sample_feature_terms(dx, k, d_bar, seed):
    """
    The final embedding always includes original x_t, so we sample only
    d_bar - dx additional terms from lag + nonlinear pool.
    """
    rng = np.random.default_rng(seed)
    pool = make_feature_pool(dx, k)

    n_extra = max(0, d_bar - dx)
    n_extra = min(n_extra, len(pool))

    if n_extra == 0:
        return []

    indices = rng.choice(len(pool), size=n_extra, replace=False)
    return [pool[i] for i in indices]


def build_nvar_embedding_one_series(X, terms, k, s):
    """
    X shape:
        (T, dx)

    Return:
        R shape (T_eff, D)
    where T_eff = T - k * s.
    """
    T, dx = X.shape
    max_lag = k * s

    if T <= max_lag + 1:
        raise ValueError(
            f"Time series too short: T={T}, but k*s={max_lag}. "
            f"Choose smaller k or s."
        )

    rows = []

    for t in range(max_lag, T):
        features = []

        # Original x_t
        features.extend(X[t].tolist())

        # Sampled lag / nonlinear terms
        for term in terms:
            if term[0] == "lag":
                _, a, p = term
                features.append(X[t - p * s, a])

            elif term[0] == "prod":
                _, a, p, b, q = term
                features.append(X[t - p * s, a] * X[t - q * s, b])

            else:
                raise ValueError(f"Unknown term: {term}")

        rows.append(features)

    return np.asarray(rows, dtype=np.float64)


def ocrep_lambda(R):
    """
    OCReP-like ridge regularization:
        lambda = sigma_min * sigma_max

    The paper uses this heuristic for readout regularization.
    """
    try:
        svals = np.linalg.svd(R, compute_uv=False)
        s_min = float(np.min(svals))
        s_max = float(np.max(svals))
        lam = s_min * s_max
    except np.linalg.LinAlgError:
        lam = 1e-6

    if not np.isfinite(lam) or lam <= 1e-12:
        lam = 1e-6

    return lam


def fit_readout_representation(R):
    """
    Fit ridge regression:
        R[:-1] -> R[1:]

    Readout:
        Y = X W + c

    Return vectorized [W, c].
    """
    X_in = R[:-1]
    Y_out = R[1:]

    n, d = X_in.shape

    # Add bias column
    X_aug = np.concatenate([X_in, np.ones((n, 1))], axis=1)

    lam = ocrep_lambda(X_aug)

    A = X_aug.T @ X_aug
    B = X_aug.T @ Y_out

    reg = lam * np.eye(A.shape[0])
    reg[-1, -1] = 0.0  # do not regularize bias

    W_aug = np.linalg.solve(A + reg, B)

    return W_aug.reshape(-1)


def compute_representations(X, terms, k, s):
    """
    X shape: (N, T, dx)
    """
    reps = []

    for i in range(X.shape[0]):
        R = build_nvar_embedding_one_series(X[i], terms, k=k, s=s)
        theta = fit_readout_representation(R)
        reps.append(theta)

    return np.vstack(reps)


def rbf_kernel_from_representations(A, B, gamma):
    """
    K_ij = exp(-||theta_i - theta_j||^2 / (2 gamma^2))
    """
    dist2 = pairwise_distances(A, B, metric="sqeuclidean")
    K = np.exp(-dist2 / (2.0 * gamma ** 2 + 1e-12))
    return K


def median_lengthscale(train_reps):
    dists = pairwise_distances(train_reps, train_reps, metric="euclidean")
    upper = dists[np.triu_indices_from(dists, k=1)]

    upper = upper[np.isfinite(upper)]
    upper = upper[upper > 1e-12]

    if len(upper) == 0:
        return 1.0

    return float(np.median(upper))


def run_experiment(dataset, k, s, d_bar, C, seed):
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    X_train = to_numpy_time_series(X_train)
    X_test = to_numpy_time_series(X_test)

    X_train = standardize_per_series(X_train)
    X_test = standardize_per_series(X_test)

    dx = X_train.shape[-1]

    terms = sample_feature_terms(dx=dx, k=k, d_bar=d_bar, seed=seed)

    train_reps = compute_representations(X_train, terms, k=k, s=s)
    test_reps = compute_representations(X_test, terms, k=k, s=s)

    gamma = median_lengthscale(train_reps)

    K_train = rbf_kernel_from_representations(train_reps, train_reps, gamma)
    K_test = rbf_kernel_from_representations(test_reps, train_reps, gamma)

    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)

    pred = clf.predict(K_test)
    acc = accuracy_score(y_test, pred)

    print(f"Dataset: {dataset}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"dx: {dx}")
    print(f"k: {k}")
    print(f"s: {s}")
    print(f"d_bar: {d_bar}")
    print(f"Number of sampled terms: {len(terms)}")
    print(f"Train representations: {train_reps.shape}")
    print(f"Test representations: {test_reps.shape}")
    print(f"RBF gamma: {gamma:.6f}")
    print(f"C: {C}")
    print(f"Accuracy: {acc:.6f}")

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GunPoint")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--d_bar", type=int, default=75)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        k=args.k,
        s=args.s,
        d_bar=args.d_bar,
        C=args.C,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()