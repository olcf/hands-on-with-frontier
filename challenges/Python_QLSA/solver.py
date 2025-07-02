# Introduction
'''
Script to run quantum linear solver on various hardware - IBM, IonQ.
Function `func_matrix_vector.py` is used to define A and b.
Function `func_qc.py` is used to generate, transpile, and run the quantum circuit.
Sample code run script:
python solver.py -case sample-tridiag -casefile input_vars.yaml -s 100  
    -backtyp real-iqm -backmet garnet:mock
python solver.py -case hele-shaw -casefile input_vars.yaml -s 100  
    -backtyp ideal 
'''

import time
import argparse
import os

import numpy as np

# Importing standard Qiskit libraries
# library to load circuit for given matrix and vector, transpile circuit with
# given backend, and run shot-based  simulations
from func_qc import qc_circ
import func_matrix_vector as matvec

#pylint: disable = line-too-long


parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case_name",  type=str, default='ideal',
    required=False, help="Name of the problem case: 'sample-tridiag', 'hele-shaw'")
parser.add_argument("-casefile", "--case_variable_file",  type=str,
    default='ideal', required=False,
    help="YAML file containing variables for the case: 'input_vars.yaml'")

parser.add_argument("-s", "--SHOTS", type=int, default=1000, required=True,
    help="Numer of shots for the simulator.")
parser.add_argument("--gpu", default=False, action='store_true',
    help="Use GPU backend for Aer simulator.")
parser.add_argument("--gpumultiple", default=False, action='store_true',
    help="Use multiple GPUs for the backend of Aer simulator.")
parser.add_argument("-backtyp", "--backend_type",  type=str, default='ideal',
    required=False, help="Type of the backend: 'ideal' 'real-ibm' 'real-iqm'")
parser.add_argument("-backmet", "--backend_method",  type=str, default='statevector',
    required=False,
    help="Method/name of the backend. E.g. 'statevector' 'fake_sherbrooke' 'ibm_sherbrooke' 'garnet' 'fake_garnet' ")
parser.add_argument("--drawcirc", default=False, action='store_true', help="Draw circuit.")
parser.add_argument("--plothist", default=False, action='store_true', help="Draw circuit.")

parser.add_argument("--savedata", default=False, action='store_true',
    help="Save data at `models/<filename>` with `<filename>` based on parameters.")
parser.add_argument("--loadcirctranspile", default=False, action='store_true',
    help="Load transpiled circuit at `models/<filename>` with `<filename>` based on parameters.")
args = parser.parse_args()

if __name__ == '__main__':
    # Get system matrix and vector
    matrix, vector, input_vars = matvec.get_matrix_vector(args)
    n_qubits_matrix = int(np.log2(matrix.shape[0]))

    # Solutions
    # classical soultion
    t = time.time()
    classical_solution = np.linalg.solve(matrix, vector/np.linalg.norm(vector))
    t_classical = time.time() - t
    print(f'Time elapsed for classical:  {int(t_classical/60)} min {t_classical%60:.2f} sec',
        flush=True)

    if args.backend_type in ('real-iqm'):
        os.environ["USING_IQM"] = "1"
    else:
        os.environ["USING_IQM"] = "0"

    # quantum solution
    qc_circ(n_qubits_matrix, classical_solution, args, input_vars)
