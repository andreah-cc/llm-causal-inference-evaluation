"""Engine 5: Overlap/positivity stress tests layered on Engines 1-4."""

import numpy as np
from .common import sigmoid
from .engine1 import gen_outcome_1
from .engine2 import gen_outcome_2
from .engine3 import gen_outcome_3
from .engine4 import gen_outcome_4


def gen_treat_5(X, w, a, k, ep, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(w, float)
    w = w / np.linalg.norm(w)
    w = w.reshape(-1, 1)
    score = a + k * (w.T @ X)
    e = sigmoid(score)
    e = ep + (1 - 2 * ep) * e
    return rng.binomial(1, e)


def main_5(
    X, w, a, k, ep, t, engine, *,
    eps, b=None, SY=None, j_star=None, rng=None
):
    """
    Apply an overlap stress test to the treatment assignment while reusing
    outcome mechanisms from Engines 1-4.

    `eps`, `SY`, and `j_star` are explicit parameters here rather than globals.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = gen_treat_5(X, w, a, k, ep, rng=rng)

    if engine == "Engine 1":
        if b is None:
            raise ValueError("Engine 1 requires b.")
        Y0, Y1, Y = gen_outcome_1(X, A, b, t, eps)

    elif engine == "Engine 2":
        Y0, Y1, Y = gen_outcome_2(X, A, t, eps)

    elif engine == "Engine 3":
        tau0, tau1, tau2 = t
        X_np = X.T
        A_n = np.asarray(A).reshape(-1)
        sd = float(np.std(eps))
        Y0, Y1, Y = gen_outcome_3(
            X_np, A_n, tau0, tau1, tau2, sd, rng=rng
        )

    elif engine == "Engine 4":
        if b is None or SY is None or j_star is None:
            raise ValueError("Engine 4 requires b, SY, and j_star.")
        try:
            t0, t1 = t
        except Exception as exc:
            raise ValueError("Engine 4 requires t=(t0, t1).") from exc
        Y0, Y1, Y = gen_outcome_4(X, A, SY, b, t0, t1, j_star, eps)

    else:
        raise ValueError("engine must be 'Engine 1', 'Engine 2', 'Engine 3', or 'Engine 4'.")

    return X, A, Y0, Y1, Y
