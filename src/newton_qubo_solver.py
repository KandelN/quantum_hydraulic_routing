"""Newton-linearized QUBO solver for small linear systems."""

import numpy as np


def build_T(n, m, s):
    weights = s * 2.0 ** (1 - np.arange(1, m + 1))
    T = np.zeros((n, n * m))
    for j in range(n):
        T[j, j * m:(j + 1) * m] = weights
    return T


def qubo_bruteforce(Q):
    n = Q.shape[0]
    best_q = None
    best_val = np.inf
    for i in range(2 ** n):
        q = np.array(list(np.binary_repr(i, width=n))).astype(int)
        val = q @ Q @ q
        if val < best_val:
            best_val = val
            best_q = q
    return best_q


def linear_to_qubo_solve(A, b, m=3, s=1.0):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[1]

    print('Condition number of J:', np.linalg.cond(A))

    D = np.diag(1 / np.linalg.norm(A, axis=1))
    A_r = D @ A
    b_r = D @ b
    print('After Row Scaling Condition Number:', np.linalg.cond(A_r))

    E = np.diag(1 / np.linalg.norm(A_r, axis=0))
    A_s = A_r @ E
    print('After Column Scaling Condition Number:', np.linalg.cond(A_s))
    print('Scaling Vector (Column):', np.round(np.diag(E), 6))

    T = build_T(n, m, s)
    x0 = -s * np.ones(n)
    A_bin = A_s @ T
    b_prime = b_r - A_s @ x0

    H = A_bin.T @ A_bin
    g = A_bin.T @ b_prime
    Q = H - 2 * np.diag(g)

    print('Binary Variables:', Q.shape[0])
    print('QUBO Coefficient Range:', (np.min(Q), np.max(Q)))

    Q = Q / np.max(np.abs(Q))
    q_sol = qubo_bruteforce(Q)
    x_scaled = x0 + T @ q_sol
    x_sol = E @ x_scaled

    classical_sol = np.linalg.solve(A, b)
    print('QUBO Solution     :', np.round(x_sol, 6))
    print('Classical Solution:', np.round(classical_sol, 6))
    print('Error Norm        :', np.linalg.norm(x_sol - classical_sol))
    return x_sol
