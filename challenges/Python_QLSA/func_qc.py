'''
Functions to perform quantum circuit loading,
transpiling the circuit for the specific backend, and
running exact and shots-based simulations.
NOTE: The current function qc_circ also computes 
the fidelity and solution of a QLSA problem. 
Thus, the number of qubits representing the system/matrix 
is also needed. If only the output is needed, 
any quantum circuit can be loaded and run.
'''

import time
import os
import pickle
import json

import numpy as np
# Importing standard Qiskit libraries
from qiskit import qpy
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import RuntimeEncoder
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from iqm.qiskit_iqm import IQMProvider

import matplotlib.pyplot as plt

# pylint: disable = invalid-name, missing-function-docstring, line-too-long
# pylint: disable = broad-exception-raised, consider-using-enumerate
# pylint: disable = import-outside-toplevel, unspecified-encoding

# get backend based on type and method
def qc_backend(backend_type, backend_method, args):
    if backend_type=='ideal':
        # ideal simulator
        if args.gpu:
            backend = AerSimulator(method=backend_method, device='GPU')
        elif args.gpumultiple:
            backend = AerSimulator(method=backend_method, device='GPU',
                blocking_enable=True, blocking_qubits=18)
        else: 
            backend = AerSimulator(method=backend_method)
    elif backend_type=='real-ibm':
        if "fake" in backend_method: # fake backend
            # fake backend
            backend = FakeProviderForBackendV2().backend(backend_method)
        else:
            # save your IBM account for future loading
            API_KEY = os.getenv('IBMQ_API_KEY')
            instance = os.getenv('IBMQ_INSTANCE')
            # save your QiskitRuntimeService accout for future loading
            QiskitRuntimeService.save_account(
              channel="ibm_quantum",
              instance=instance,
              token=API_KEY,
              overwrite=True
            )
            service = QiskitRuntimeService()
            backend = service.backend(backend_method)
    elif backend_type=='real-iqm':
        # save your IQM account for future loading
        API_KEY = os.getenv('IQM_API_KEY') # ${IQM_TOKEN} can't be set when using `token` parameter below
        server_url = f"https://cocos.resonance.meetiqm.com/{backend_method}"
        if "fake" in backend_method: # "facade" backends only work for Adonis (switching to "fake" backends)
            if backend_method=="fake_garnet":
                from iqm.qiskit_iqm import IQMFakeGarnet
                backend = IQMFakeGarnet()
            else:
                raise Exception('Unknown fake backend.')
        else:
            backend = IQMProvider(server_url, token=API_KEY).get_backend()
    else:
        raise Exception(f'Backend type \'{backend_type}\' not implemented.')
    return backend

# circuit generation, transpile, running
def qc_circ(n_qubits_matrix, classical_solution, args, input_vars):
    '''
    Function to load quantum circuit, transpile, and run.
    Input:
        n_qubits_matrix            num. of quibits representing the system/matrix
        classical_solution         classical solution of linear system of equations
        args                       input arguments containing details of shots, backend, etc.
        input_vars                 parameters of the system of equations being solved
    Output:
        job                        the job handle of Qiskit primitive (only Sampler for now)
    '''
    print('**************************Quantum circuit loading, transpile & ' +
       'running*************************', flush=True)
    # ============================
    # First setup quantum backend
    print(f'Using \'{args.backend_type}\' simulator with \'{args.backend_method}\' backend')
    backend = qc_backend(args.backend_type, args.backend_method, args)
    print(f'Backend: {backend}')


    # ============================
    # 1. Load generated circuit
    filename = input_vars['savefilename'].format(**input_vars)
    savefilename = f'{filename}_circ_nqmatrix{n_qubits_matrix}'
    t = time.time()
    with open(f'{savefilename}.qpy', 'rb') as fd:
        circ = qpy.load(fd)[0]
    t_load = time.time() - t
    print('===============Loaded circuit (before transpile) ==============')
    print(f'Time elapsed for loading circuit:  {int(t_load/60)} min {t_load%60:.2f} sec',
        flush=True)
    circ.measure_all()
    if args.drawcirc:
        print(f'Circuit:\n{circ.draw()}', flush=True)
    print(f"Circuit details:\n# qubits = {circ.num_qubits}\n# gates = {sum(circ.count_ops().values())}\n# CNOT = {circ.count_ops()['cx']}\nDepth = {circ.depth()}")


    # ============================
    # 2. Transpile circuit for simulator
    savefilename = f'{filename}_circ-transpile_nqmatrix{n_qubits_matrix}_backend-{args.backend_method}'
    if args.loadcirctranspile is True:
        t = time.time()
        file = open(savefilename, 'rb')
        data = pickle.load(file)
        file.close()
        circ = data['circ']
        t_load = time.time() - t
        print('===============Loaded transpiled circuit using pickled data==============')
        print(f'Time elapsed for loading circuit:  {int(t_load/60)} min {t_load%60:.2f} sec',
            flush=True)
    else:
        if args.backend_type in ('real-iqm'):
            from iqm.qiskit_iqm import transpile_to_IQM as transpile
            if args.backend_method in ('fake_garnet'):
                from qiskit.transpiler.passes import RemoveResetInZeroState
                circ = RemoveResetInZeroState()(circ.decompose())
        else:
            from qiskit import transpile
        t = time.time()
        isa_circ = transpile(circ, backend)
        t_transpile = time.time() - t
        print(f'Time elapsed for transpiling the circuit:  {int(t_transpile/60)} min {t_transpile%60:.2f} sec')

        # Save data
        if args.savedata:
            save_data = {   'args'                  : args,
                            'input_vars'            : input_vars,
                            't_transpile'           : t_transpile}
            file = open(f'{savefilename}.pkl', 'wb')
            pickle.dump(save_data, file)
            file.close()
            # save transpiled circuit
            with open(f'{savefilename}.qpy', 'wb') as fd:
                qpy.dump(isa_circ, fd)
            print("===========Transpiled Circuit saved===========", flush=True)


    # ============================
    # 3. Run and get counts
    shots = args.SHOTS
    # Setup Sampler
    t = time.time()
    
    # Run the job
    if args.backend_type in ('real-iqm'):
        job = backend.run(isa_circ, shots=shots)
    elif args.backend_type in (['real-ibm','ideal']):
        sampler = Sampler(backend)
        job = sampler.run([isa_circ], shots=shots)
   
   # Grab results from the job
    
    job_id = job.job_id()
    print(f"job_id: {job_id}")
    result = job.result()
    t_run = time.time() - t
    print(f'Time elapsed for running the circuit:  {int(t_run/60)} min {t_run%60:.2f} sec',
        flush=True)
    
    # Returns counts
    if args.backend_type in ('real-iqm'):
        counts = result.get_counts()
    elif args.backend_type in (['real-ibm','ideal']):
        counts = result[0].data.meas.get_counts()
    print(f'counts:\n{counts}')


    # Saving the final statevector if using ideal (qiskit) backend
    if args.backend_type=='ideal':
        isa_circ.remove_final_measurements()  # no measurements allowed
        statevector = Statevector(isa_circ)
        statevector = np.asarray(statevector)
        istart = int(len(statevector)/2)
        exact_vector = statevector[istart:istart+(int(2**n_qubits_matrix))].real

    # get counts based probabilistic/statistical state vector
    counts_ancilla, counts_total, probs_vector, counts_vector = \
        get_ancillaqubit(counts, n_qubits_matrix)
    print(f'All counts of ancila (only the first 2**nq represent solution vector):\n{counts_ancilla}')
    print("Counts vector should approach exact vector in infinite limit")
    print(f'counts_vector:\n{counts_vector}')
    if args.backend_type=='ideal':
        print(f'exact_vector/norm:\n{exact_vector/np.linalg.norm(exact_vector)}')
    
    # print solutions
    print(f'\ntrue solution:\n{classical_solution}')
    # normalize counts vector with true solution norm
    counts_solution_vector = \
        np.linalg.norm(classical_solution) * counts_vector / np.linalg.norm(counts_vector)
    print(f'\ncounts solution vector:\n{counts_solution_vector}')
    print(f'diff with true solution (%):\n{np.abs(classical_solution-counts_solution_vector)*100/(classical_solution+1e-15)}')
    print(f'Fidelity: {fidelity(counts_solution_vector, classical_solution)}')
    if args.backend_type=='ideal':
        exact_solution_vector = \
            np.linalg.norm(classical_solution) * exact_vector / np.linalg.norm(exact_vector)
        print(f'\nexact solution vector:\n{exact_solution_vector}')
        print(f'diff with true solution (%):\n{np.abs(classical_solution-exact_solution_vector)*100/(classical_solution+1e-15)}')
        print(f'Fidelity: {fidelity(exact_solution_vector, classical_solution)}')

    # plot histogram
    if args.plothist:

        plot_histogram(counts, figsize=(7, 7), color='tab:green',
            title=f'{args.backend_type}:{args.backend_method}')  # dodgerblue tab:green
        plt.savefig('Figs/temp_hist.png')

    # Save full data
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
                        't_run'                     : t_run,
                        'shots'                     : shots,
                        'fidelity'                  : fidelity(counts_solution_vector, classical_solution)
                        }
        # save results
        with open(f"{savefilename}.pkl", "wb") as file:
            pickle.dump(save_data, file)
        # with open(f"{savefilename}_result.json", "w") as file:
        #     json.dump(result, file, cls=RuntimeEncoder)
        print("===========Full data saved===========")

    # return job


# function to measure the qubits
def get_ancillaqubit(counts, nq):
    """
    NOTE: only count measurements when ancilla qubit (leftmost) is 1
    Input:
        counts   counts from the simulator
        nq       number of qubits used to represent the system or solution vector
    Output:
        counts_ancill     accounts of the measurements where ancilla qubit = 1
        other metrics for combination of nq qubits = 1
    """
    if not counts:
        # Handle empty input counts
        num_states = 2**nq
        return [], 0, np.zeros(num_states), np.zeros(num_states)

    # Determine the expected prefix for relevant states (Ancilla=1, Work=0...0)
    total_qubits = len(next(iter(counts)))
    num_work_qubits = total_qubits - nq - 1
    if num_work_qubits < 0:
        raise ValueError("Inconsistent qubit counts: total_qubits < num_solution_qubits + 1")
    bits_prefix = "1" + "0" * num_work_qubits

    def _print_the_array(arr, n):
        cache = ""
        for i in range(0, n):
            cache += str(arr[i])
        if bits_prefix + cache not in counts:
            counts[bits_prefix + cache] = 0

    # Function to generate all binary strings
    def _generate_all_binary_strings(n, arr, i):
        if i == n:
            _print_the_array(arr, n)
            return
        # First assign "0" at ith position
        # and try for all other permutations
        # for remaining positions
        arr[i] = 0
        _generate_all_binary_strings(n, arr, i + 1)
        # And then assign "1" at ith position
        # and try for all other permutations
        # for remaining positions
        arr[i] = 1
        _generate_all_binary_strings(n, arr, i + 1)

    arr = [None] * len(next(iter(counts)))

    # Print all binary strings
    _generate_all_binary_strings(nq, arr, 0)

    counts_list = list(counts.items())
    counts_ancilla = []
    ancilla_states = []
    # check the left most qubit
    for i in range(len(counts_list)):
        if counts_list[i][0][0] == "1":  # extract all ancilla qubits=1
            counts_ancilla += (counts_list[i],)
            ancilla_states += (counts_list[i][0],)
    # sort based on right most qubits. Find the argsort and manually rearrange the counts list.
    ancilla_states_sortedID = np.argsort(ancilla_states)
    counts_ancilla_sorted = []
    for i in range(len(counts_ancilla)):
        counts_ancilla_sorted += (counts_ancilla[ancilla_states_sortedID[i]],)
    counts_ancilla = counts_ancilla_sorted.copy()

    # At this point, all the states are sorted such that ancilla=1 and the
    # combination of nb qubits is 0 or 1
    # So, we take the first 2**nb states (OR size of the system)
    num_state = 2**nq
    
    # print(f'The number of counts of ancilla bits: {len(counts_ancilla)},
    # N.O num_state: {num_state}')
    # re-compute counts_total
    counts_total = 0
    for i in range(num_state):
        counts_total += counts_ancilla[i][1]
    # compute solution vectors
    probs_vector = []
    counts_vector = []
    for i in range(num_state):
        try:
            probs_vector += (counts_ancilla[i][1] / (1.0 * counts_total),)
            counts_vector += (np.sqrt(counts_ancilla[i][1] / (1.0 * counts_total)),)
        except ZeroDivisionError:
            print(f"\
                *****************WARNING************************\n\
                    ZeroDivisionError: counts_ancilla[{i}] = {counts_ancilla[i]}\n\
                    Could be a result of bad simulator results. Generating fake probs_vector\n\
                    and counts_vector...")
            probs_vector += (1.e-15 / (1.0 * 1.e-15),)
            counts_vector += (np.sqrt(1.e-15 / (1.0 * 1.e-15)),)

    return counts_ancilla, counts_total, np.array(probs_vector), np.array(counts_vector)

# function to compute fidelity of the solution
def fidelity(qfunc, true):
    '''
    Function to compute fidelity of solution state.
    Input:
        qfunc   quantum solution state
        true    classiccal/true solution state
    Output:
        fidelity   state fidelity
    '''
    solution_qfun_normed = qfunc / np.linalg.norm(qfunc)
    solution_true_normed = true / np.linalg.norm(true)
    return state_fidelity(solution_qfun_normed, solution_true_normed)
