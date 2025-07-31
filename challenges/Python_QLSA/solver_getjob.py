'''
Script to extract and process job results from IBM, IQM, and IonQ.
'''

import numpy as np
from func_qc import get_ancillaqubit, fidelity
import func_matrix_vector as matvec
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case_name",  type=str, default='ideal',
    required=False, help="Name of the problem case: 'sample-tridiag', 'hele-shaw'")
parser.add_argument("-casefile", "--case_variable_file",  type=str,
    default='ideal', required=False,
    help="YAML file containing variables for the case: 'input_vars.yaml'")
parser.add_argument("-s", "--SHOTS", type=int, default=1000, required=True,
    help="Numer of shots for the simulator.")
parser.add_argument("-backtyp", "--backend_type",  type=str, default='ideal',
    required=False, help="Type of the backend: 'ideal' 'real-ibm' 'real-iqm' 'real-ionq'")
parser.add_argument("-backmet", "--backend_method",  type=str, default='statevector',
    required=False,
    help="Method/name of the backend. E.g. 'statevector' " \
    "'ibm_sherbrooke' 'fake_sherbrooke' 'garnet' 'fake_garnet' 'qpu.aria-1' 'aria-1'")
parser.add_argument("-jid", "--job_id",  type=str, required=True, 
                    help="Job ID for the backend")
parser.add_argument("--savedata", default=False, action='store_true',
    help="Save data at `models/<filename>` with `<filename>` based on parameters.")
args = parser.parse_args()

# Get system matrix and vector
matrix, vector, input_vars = matvec.get_matrix_vector(args)
n_qubits_matrix = int(np.log2(matrix.shape[0]))

# Setup quantum backend and get job info
backend_type = args.backend_type
backend_method = args.backend_method
job_id = args.job_id
if backend_type=='real-ibm':
    from qiskit_ibm_runtime import QiskitRuntimeService
    # save your IBMProvider accout for future loading
    API_KEY = os.getenv('IBMQ_API_KEY')
    instance = os.getenv('IBMQ_INSTANCE')
    # save your QiskitRuntimeService accout for future loading
    QiskitRuntimeService.save_account(
        token=API_KEY,
        instance=instance,
        overwrite=True
    )
    service = QiskitRuntimeService()  # To get back results
    backend = service.backend(backend_method)
    job = service.job(job_id)
elif backend_type=='real-iqm':
    from iqm.qiskit_iqm import IQMProvider
    # save your IQM account for future loading
    API_KEY = os.getenv('IQM_API_KEY') # ${IQM_TOKEN} can't be set when using `token` parameter below
    server_url = f"https://cocos.resonance.meetiqm.com/{backend_method}"
    if "fake" in backend_method: # "facade" backends only work for Adonis (switching to "fake" backends)
        raise Exception('IQM fake backend results cannot be retrieved using this code.')
    else:
        backend = IQMProvider(server_url, token=API_KEY).get_backend()
    from iqm.qiskit_iqm.iqm_job import IQMJob
    job = IQMJob(backend, job_id)
elif backend_type=='real-ionq':
    from qiskit_ionq import IonQProvider
    # save your IonQ accout for future loading
    API_KEY = os.getenv('IONQ_API_KEY')
    provider = IonQProvider(API_KEY)
    if "qpu." in backend_method: # real hardware
        backend = provider.get_backend(f"{backend_method}", gateset='qis')
    else: # emulator: add noise to simulator to obtain emulator
        backend = provider.get_backend("simulator", gateset='qis')
        backend.set_options(shots=args.SHOTS, sampler_seed=np.random.randint(100), noise_model=backend_method)
    from qiskit_ionq import ionq_job
    job = ionq_job.IonQJob(backend, job_id)
else:
    raise Exception(f'Backend type \'{backend_type}\' not implemented.')

# Retrieve Results
result = job.result()
if args.backend_type in ('real-iqm', 'real-ionq'):
    counts = result.get_counts()
elif args.backend_type in (['real-ibm']):
    counts = result[0].data.meas.get_counts()
print(f'counts:\n{counts}')

# Post-process counts with respect to solution system
# Get required counts vectors
counts_ancilla, _, _, counts_vector = get_ancillaqubit(counts, n_qubits_matrix)
classical_solution = np.linalg.solve(matrix, vector/np.linalg.norm(vector))
# normalize counts vector with true solution norm
counts_solution_vector = \
    np.linalg.norm(classical_solution) * counts_vector / np.linalg.norm(counts_vector)

# Save full data
filename = input_vars['savefilename'].format(**input_vars)
shots = args.SHOTS
savefilename = \
    f'{filename}_circ-fullresults_nqmatrix{n_qubits_matrix}_backend-{args.backend_method}_shots{shots}'
if args.savedata:
    save_data = {   'args'                      : args,
                    'input_vars'                : input_vars,
                    'counts'                    : counts,
                    'counts_ancilla'            : counts_ancilla,
                    'counts_vector'             : counts_vector,
                    'counts_solution_vector'    : counts_solution_vector,
                    'classical_solution'        : classical_solution,
                    't_run'                     : None,  # use data from cloud server
                    'shots'                     : shots,
                    'fidelity'                  : fidelity(counts_solution_vector, classical_solution)
                    }
    # save results
    with open(f"{savefilename}.pkl", "wb") as file:
        pickle.dump(save_data, file)
    print("===========Full data saved===========")
