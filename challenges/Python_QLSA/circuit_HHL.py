#pylint: disable = invalid-name
"""
Introduction
Script to generate HHL circuit that solves any Ax=b problem.
Function `func_matrix_vector.py` is used to define A and b.
Sample code run script:
python circuit_HHL.py -case sample-tridiag -casefile input_vars.yaml --savedata
python circuit_HHL.py -case hele-shaw -casefile input_vars.yaml --savedata
"""

import time
import argparse
import pickle

import numpy as np

# Importing standard Qiskit libraries
from qiskit import transpile
from qiskit import qpy
from qiskit_aer import AerSimulator
from linear_solvers import HHL
# library to generate matrix and vector for linear system of equations
import func_matrix_vector as matvec



parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case_name",  type=str, default='ideal',
    required=False, help="Name of the problem case: 'sample-tridiag', 'hele-shaw'")
parser.add_argument("-casefile", "--case_variable_file",  type=str, default='ideal',
    required=False, help="YAML file containing variables for the case: 'input_vars.yaml'")
parser.add_argument("--gpu", default=False, action='store_true',
    help="Use GPU backend for Aer simulator.")
parser.add_argument("--gpumultiple", default=False, action='store_true',
    help="Use multiple GPUs for the backend of Aer simulator.")
parser.add_argument("--drawcirc", default=False, action='store_true', help="Draw circuit.")

parser.add_argument("--savedata", default=False, action='store_true',
    help="Save data at `models/<filename>` with `<filename>` based on parameters.")
args = parser.parse_args()

if __name__ == '__main__':
    # Get system matrix and vector
    matrix, vector, input_vars = matvec.get_matrix_vector(args)
    MATRIX_SIZE = matrix.shape[0]
    n_qubits_matrix = int(np.log2(MATRIX_SIZE))

    # setup quantum backend
    backend_type = 'ideal'
    backend_method = 'statevector'
    print(f'Using \'{backend_type}\' simulator with \'{backend_method}\' backend')
    if args.gpu:
        backend = AerSimulator(method=backend_method, device='GPU')
    elif args.gpumultiple:
        backend = AerSimulator(method=backend_method, device='GPU',
            blocking_enable=True, blocking_qubits=18)
    else: backend = AerSimulator(method=backend_method)
    print(f'Backend: {backend}')

    # setup HHL solver
    # backend_init = qc_backend('ideal', 'statevector', args)
    hhl = HHL(quantum_instance=backend)

    # Generate HHL circuit
    print('==================Generating HHL circuit================', flush=True)
    t = time.time()
    circ = hhl.construct_circuit(matrix, vector)
    t_circ = time.time() - t
    print(f'Time elapsed for generating HHL circuit:  {int(t_circ/60)} min {t_circ%60:.2f} sec')

    # Save data
    if args.savedata:
        circ_transpile = transpile(circ, backend)
        # save metadata (DON'T USE Pickle to save the circuit - only works for a given version)
        save_data = {   'args'                  : args,
                        'input_vars'            : input_vars,
                        'matrix'                : matrix,
                        'vector'                : vector,
                        't_circ'                : t_circ}
        filename = input_vars['savefilename'].format(**input_vars)
        savefilename = f'{filename}_circ_nqmatrix{n_qubits_matrix}'
        file = open(f'{savefilename}.pkl', 'wb')
        pickle.dump(save_data, file)
        file.close()
        # save circuit as QPY file
        with open(f'{savefilename}.qpy', 'wb') as fd:
            qpy.dump(circ_transpile, fd)
        print("===========Circuit saved===========")

    # Plot circuit
    if args.drawcirc:
        circ.measure_all()
        print(f'Circuit:\n{circ.draw()}', flush=True)
