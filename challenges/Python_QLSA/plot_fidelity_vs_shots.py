import pickle
from matplotlib import pyplot as plt
import os
import numpy as np
import glob

def _get_backend_name(filename):
    """
    Helper function to get the backend name from the results filename.
    The backend name is expected to be between 'backend-' and '_shots'.
    For example, if the filename is 
    'sample_HHL_circ-fullresults_nqmatrix1_backend-garnet:mock_shots1000.pkl',
    the function will return 'garnet:mock'.

    Parameters:
    filename (str): The name of the file from which to extract the backend name.

    Returns:
    str: The extracted backend name, or None if the pattern is not found.
    """
    # Define the start and end patterns
    start_pattern = 'backend-'
    end_pattern = '_shots'
    
    # Find the starting index
    start_index = filename.find(start_pattern)
    if start_index == -1:
        return None  # Return None if 'backend-' is not found

    # Adjust start_index to point to the start of the desired string (after 'backend-')
    start_index += len(start_pattern)

    # Find the ending index
    end_index = filename.find(end_pattern, start_index)
    if end_index == -1:
        return None  # Return None if '.pkl' is not found

    # Extract and return the desired string
    return filename[start_index:end_index]

def get_results_files():
    """
    The function searches and returns a list of paths 
    for files with the extension '.pkl' in the 'models' directory.
    """

    files = None
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".pkl"):
                files = glob.glob("models/*fullresults*.pkl")
        if files == [] or None:
            print("No result files found in 'models'")
            print("Make sure to run the circuit and solver scripts first,\n" \
            "and set '--savedata' flag in the solver script.")
    else:
        print("Could not find directory 'models'")
        print("Make sure to run the circuit and solver scripts first.")
        exit(1)

    return files

def get_results(files):
    """
    The function takes a list of file paths, extracts the backend name from each file,
    and organizes the fidelity results by backend name.
    
    Parameters:
    files (list): A list of file paths to the results files.

    Returns:
    backend_results (dict): A nested dictionary where keys are backend names 
                            and values are dictionaries with keys as shot counts
                            and values as fidelity results.
    """
    backends = set([_get_backend_name(file) for file in files])
    backend_results = {}
    for backend in backends:
        fidelities = {}
        for file in files:
            if backend != _get_backend_name(file):
                continue
            with open(file, "rb") as f:
                results = pickle.load(f)
                fidelities[results['shots']] = results['fidelity']
        backend_results[backend] = fidelities
    return backend_results

if __name__ == "__main__":
    files = get_results_files()
    backend_results = get_results(files)

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    for backend, fidelities in backend_results.items():
        sorted_fidelities = dict(sorted(fidelities.items()))
        backend_label = backend
        if backend == 'statevector':
            backend_label = 'ideal simulator'
            ax[0].plot(list(sorted_fidelities.keys()), list(sorted_fidelities.values()), marker='o', label=backend_label)
            ax[0].set_xlabel("Number of shots")
            ax[0].set_ylabel("Fidelity")
            ax[0].legend()
        ax[1].plot(list(sorted_fidelities.keys()), list(sorted_fidelities.values()), marker='o', label=backend_label)
        ax[1].set_xlabel("Number of shots")
        ax[1].set_ylabel("Fidelity")
        ax[1].legend()
    plt.suptitle("Fidelity vs Number of Shots")
    plt.savefig("fidelity_vs_shots.png", format='png')
    print("DONE! Plot saved as 'fidelity_vs_shots.png'")
