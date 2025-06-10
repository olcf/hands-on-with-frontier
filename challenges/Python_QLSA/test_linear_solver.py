"""
Script to test the quantum linear solver installation.

Sample run script:
python test_linear_solver.py -nq 2
"""

import time
import argparse
import numpy as np
from scipy.sparse import diags
# Importing Qiskit libraries
from qiskit_aer import AerSimulator
from linear_solvers import NumPyLinearSolver, HHL

#pylint: disable=invalid-name, line-too-long

parser = argparse.ArgumentParser()
parser.add_argument("-nq", "--NQ_MATRIX", type=int, default=2, required=True,
    help="Numer of qubits to determine size of linear system of quations " + \
        "(A*x=b) being solved. Size of A matrix = 2^NQ_MATRIX.")
args = parser.parse_args()

if __name__ == '__main__':
    # Generate matrix and vector
    # We use a sample tridiagonal system. It's 2x2 version is:
    # matrix = np.array([ [1, -1/3], [-1/3, 1] ])
    # vector = np.array([1, 0])
    n_qubits_matrix = args.NQ_MATRIX
    MATRIX_SIZE = 2 ** n_qubits_matrix
    # entries of the tridiagonal Toeplitz symmetric matrix
    a = 1
    b = -1/3
    matrix = diags([b, a, b],
                [-1, 0, 1],
                shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    vector = np.array([1] + [0]*(MATRIX_SIZE - 1))

    # ============
    # Select backend: Using different simulators (default in `linear_solvers`
    # is statevector simulation)
    backend = AerSimulator(method='statevector')
    backend.set_options(precision='single')

    # ============
    # Setup HHL solver
    hhl = HHL(1e-3, quantum_instance=backend)
    print(f'Simulator: {backend}')

    # ============
    # Solutions
    print('======================')
    # Classical
    t = time.time()
    classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
    elpsdt = time.time() - t
    print(f'Time elapsed for classical:\n{int(elpsdt/60)} min {elpsdt%60:.2f} sec')
    # HHL
    t = time.time()
    hhl_solution = hhl.solve(matrix, vector)
    elpsdt = time.time() - t
    print(f'Time elapsed for HHL:\n{int(elpsdt/60)} min {elpsdt%60:.2f} sec')

    # ============
    # Circuits
    print('======================')
    print('HHL circuit:')
    print(hhl_solution.state)

    # ============
    # Comparing the observable - Euclidean norm
    print('======================')
    print(f'Euclidean norm classical:\n{classical_solution.euclidean_norm}')
    print(f'Euclidean norm HHL:\n{hhl_solution.euclidean_norm} (diff (%): {np.abs(classical_solution.euclidean_norm-hhl_solution.euclidean_norm)*100/classical_solution.euclidean_norm:1.3e})')

    # ============
    # Comparing the solution vectors component-wise
    print('======================')
    from qiskit.quantum_info import Statevector
    def get_solution_vector(solution, nstate):
        """
        Extracts and normalizes simulated state vector
        from LinearSolverResult.
        """
        # solution_vector = Statevector(solution.state).data[-nstate:].real
        temp = Statevector(solution.state)
        ID = np.where(np.abs(temp.data[:].real)<1e-10)[0]
        A = temp.data[:]
        A[ID] = 0+0j
        B = temp.data[:].real
        B[ID] = 0
        # print(f'# of elements in solution vector: {len(B)}')
        istart = int(len(B)/2)
        solution_vector = temp.data[istart:istart+nstate].real
        norm = solution.euclidean_norm
        return norm * solution_vector / np.linalg.norm(solution_vector)

    print(f'Classical solution vector:\n{classical_solution.state}')
    solvec_hhl = get_solution_vector(hhl_solution, MATRIX_SIZE)
    print(f'HHL solution vector:\n{solvec_hhl}')
    print(f'diff (%):\n{np.abs(classical_solution.state-solvec_hhl)*100/classical_solution.state}')
