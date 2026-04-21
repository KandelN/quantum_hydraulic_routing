"""HHL-based linear solver helper for Jacobian systems.

This keeps the older workflow that relies on an older Qiskit stack and a local
install of the matching ``quantum_linear_solvers`` package.
"""

import numpy as np


def quantum_solver(A, b):
    from linear_solvers import NumPyLinearSolver, HHL
    from qiskit.quantum_info import Statevector

    np.set_printoptions(precision=5, linewidth=120)
    print('Matrix A:\n', A)
    print('Vector b:\n', b)
    print('Condition Number:', np.linalg.cond(A))

    normalization_factor = np.linalg.norm(b)
    b_normalized = b / normalization_factor
    print('Normalized Vector b:\n', b_normalized)

    result = HHL().solve(A, b)
    print('Result Object Returned:\n', result)
    print('Result Circuit:\n')
    print(result.state)

    result_sv = Statevector(result.state).data
    states = [2 ** (result.state.num_qubits - 1) + i for i in range(len(b))]
    result_vector = np.array([result_sv[i] for i in states])

    print('Result states:', states)
    print('Result Euclidean Norm:', result.euclidean_norm)
    print('Result Vector |x>:', result_vector)
    print('Real Component of Result Vector |x>:', result_vector.real)

    quantum_solution = result.euclidean_norm * result_vector.real / np.linalg.norm(result_vector.real)
    classical_solution = NumPyLinearSolver().solve(A, b_normalized)

    denormalized_classical = classical_solution.state * normalization_factor
    denormalized_quantum = quantum_solution * normalization_factor

    print('Denormalized Quantum Solution:', denormalized_quantum)
    print('True Classical Solution x:', np.round(denormalized_classical, 6))
    print('Quantum Solution x:', np.round(denormalized_quantum, 6))
    print('Error Vector:', denormalized_quantum - denormalized_classical)
    print('Error Norm:', np.linalg.norm(denormalized_quantum - denormalized_classical))
    return denormalized_quantum


def hermitian_block_matrix(mat, vec):
    mat_dag = mat.conj().T
    zero = np.zeros_like(mat)
    H = np.block([[zero, mat], [mat_dag, zero]])
    b_ext = np.concatenate([vec, np.zeros_like(vec)])
    return H, b_ext


def solve_qls(A, b):
    if np.allclose(A, A.conj().T):
        print('System is already Hermitian.')
        return quantum_solver(A, b)

    print('System is not Hermitian. Constructing Hermitian block matrix.')
    print('Original Condition Number of A:', np.linalg.cond(A))

    D = np.diag(1 / np.linalg.norm(A, axis=1))
    As = D @ A
    bs = D @ b
    E = np.diag(1 / np.linalg.norm(As, axis=0))
    As = As @ E

    print('Condition Number after Scaling:', np.linalg.cond(As))
    H, B = hermitian_block_matrix(As, bs)
    print('Condition Number of Hermitian Block Matrix H:', np.linalg.cond(H))

    x_quantum = quantum_solver(H, B)[-len(b):]
    if np.linalg.cond(A) > 1e10:
        print('Warning: Original matrix A is ill-conditioned. Quantum solution may be inaccurate.')
    return E @ x_quantum
