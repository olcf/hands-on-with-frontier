'''
Sample code to test installation of Qiskit and additional plugins
to run a circuit on different 
backends (simulators, emulators, and real devices)

Sample run script:
python test_qiskit_installation.py -backtyp ideal
'''

import os
import argparse
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

#pylint: disable=broad-exception-raised, line-too-long, ungrouped-imports, invalid-name

parser = argparse.ArgumentParser()
parser.add_argument("-backtyp", "--backend_type",  type=str, default='ideal',
    required=False,
    help="Type of the backend: 'ideal', 'fake-ibm' 'real-ibm' 'real-iqm'")
args = parser.parse_args()

if __name__ == '__main__':
    backend_type = args.backend_type
    # Choose the simulator or backend to run the quantum circuit
    if backend_type=='ideal':
        # Using ideal simulator, AerSimulator (works even without IBMQ account,
        # don't have to wait in a queue)
        backend = AerSimulator()
    elif backend_type=='fake-ibm':
        # Using qiskit's fake provider (works even without IBMQ account,
        # don't have to wait in a queue)
        backend = FakeProviderForBackendV2().backend("fake_sherbrooke")
    elif backend_type=='real-ibm':
        # Using the latest qiskit_ibm_runtime
        #### IF YOU HAVE AN IBMQ ACCOUNT (using an actual backend) #####
        # save your IBM accout for future loading
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
        backend = service.backend("ibm_sherbrooke")
        print("WARNING: When using the real IBM backend, running the circuit and returning the results will take time due to the queue wait time. The job submission may time out and you will get a connection error. Use the online dashboard to see results.")
    elif backend_type=='fake-iqm':
        # from iqm.qiskit_iqm import IQMProvider
        # # save your IQM account for future loading
        # API_KEY = os.getenv('IQM_API_KEY') # ${IQM_TOKEN} can't be set when using `token` parameter below
        # server_url = "https://cocos.resonance.meetiqm.com/garnet:mock"
        # backend = IQMProvider(server_url, token=API_KEY).get_backend('facade_garnet')
        raise Exception('Backend type \'{backend_type}\' not implemented.')
    elif backend_type=='real-iqm':
        # from iqm.qiskit_iqm import IQMProvider
        # # save your IQM account for future loading
        # API_KEY = os.getenv('IQM_API_KEY') # ${IQM_TOKEN} can't be set when using `token` parameter below
        # server_url = "https://cocos.resonance.meetiqm.com/garnet"
        # backend = IQMProvider(server_url, token=API_KEY).get_backend()
        # print("WARNING: When using the real IQM backend, running the circuit and returning the results will take time due to the queue wait time. The job submission may time out and you will get a connection error. Use the online dashboard to see results.")
        raise Exception('Backend type \'{backend_type}\' not implemented.')
    else:
        raise Exception('Backend type \'{backend_type}\' not implemented.')

    print(f'Backend: {backend}')
    ######################################

    # Create the circuit - the Bell state
    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(2)
    # Add a H gate on qubit 0
    circuit.h(0)
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
    circuit.cx(0, 1)
    # Map the quantum measurement to the classical bits
    circuit.measure_all()

    # Draw the circuit
    print(circuit.draw())

    # Circuits must obey the ISA of the backend.
    # Convert to ISA circuits via transpilation for the specific backend.
    if 'iqm' in backend_type:
        # from iqm.qiskit_iqm import transpile_to_IQM as transpile
        raise Exception('Backend type \'{backend_type}\' not implemented.')
    else:
        from qiskit import transpile
    isa_circuit = transpile(circuit, backend)

    # Setup Sampler to submit job
    sampler = Sampler(backend)
    # Run the job - this will sit on this line synchronously until complete.
    # One could call sampler.run() and obtain the jobId, then subsequently
    # execute the remainder of this script to poll the jobId for the
    # completed result. A Jupyter notebook example in this project repo is
    # provided to demonstrate.
    job = sampler.run([isa_circuit], shots=1000)

    # Grab results from the job
    result = job.result()
    # Returns counts
    counts = result[0].data.meas.get_counts()
    print(f"\nTotal count for 00 and 11 are: {counts}")
