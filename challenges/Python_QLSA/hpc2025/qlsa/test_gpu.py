'''
Sample code to test the GPU (and multi-GPU) capability of aer_simulator
Reference: https://qiskit.org/ecosystem/aer/howtos/running_gpu.html

Sample run script:
python test_gpu.py -nq 2 --gpu
'''

import argparse
import time
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QuantumVolume
from qiskit import transpile

#pylint: disable=line-too-long

parser = argparse.ArgumentParser()
parser.add_argument("-nq", "--NUM_QUBITS", type=int, default=2, required=True,
    help="Numer of qubits to determine size of circuit.")
parser.add_argument("--gpu", default=False, action='store_true',
    help="Use GPU backend for Aer simulator.")
parser.add_argument("--gpumultiple", default=False, action='store_true',
    help="Use multiple GPUs for the backend of Aer simulator.")
args = parser.parse_args()

if __name__ == '__main__':
    # Select backend
    if args.gpu:
        backend = AerSimulator(method='statevector', device='GPU')
    elif args.gpumultiple:
        backend = AerSimulator(method='statevector', device='GPU',
            blocking_enable=True, blocking_qubits=18)
    else:
        backend = AerSimulator(method='statevector')
    print(f'Simulator: {backend}')

    # Generate circuit and transpile
    qubits = args.NUM_QUBITS
    t = time.time()
    qc = QuantumVolume(qubits, seed = 0)
    qc.measure_all()
    qc = transpile(qc, backend)
    elpsdt1 = time.time() - t

    # Run circuit
    t = time.time()
    result = backend.run(qc).result()
    elpsdt2 = time.time() - t

    print(f'N qubits: {qubits}; GPU: {args.gpu}; multiple-GPU: {args.gpumultiple};\nTime elapsed 1:  {int(elpsdt1/60)} min {elpsdt1%60:.2f} sec\nTime elapsed 2:  {int(elpsdt2/60)} min {elpsdt2%60:.2f} sec')
