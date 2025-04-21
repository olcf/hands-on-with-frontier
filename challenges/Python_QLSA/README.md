# Quantum Linear Solver Algorithm Challenges for HPC Crash Course 2025!

Welcome to the ORNL 2025 Crash Course on Quantum Linear Solver Algorithms (QLSA) and their implications for the future of quantum computing! As we stand on the brink of a new era in computational technology, understanding the concepts and applications of quantum algorithms has never been more critical. This course is designed to equip you with the knowledge and skills necessary to navigate the evolving landscape of quantum computing and give you experience with running circuits on REAL quantum devices! 

Here we provide a sample implementation of a QLSA, the [Harrow–Hassidim–Lloyd (HHL)](https://doi.org/10.1103/PhysRevLett.103.150502) algorithm, using [Qiskit's HHL implementation](https://github.com/Qiskit/textbook/blob/main/notebooks/ch-applications/hhl_tutorial.ipynb). The implementation uses Python scripts that take advantage of Qiskit libraries. The codes are designed to run the QLSA circuit on REAL quantum devices:

1. [IQM](https://www.meetiqm.com/) (used in this crash course)
2. [IBM quantum-computing](https://quantum-computing.ibm.com/)

An application to fluid dynamics is also provided. The fluid dynamics use case follows the work of [Bharadwaj & Srinivasan (2020)](https://www.sto.nato.int/publications/STO%20Educational%20Notes/STO-EN-AVT-377/EN-AVT-377-01.pdf) and [Gopalakrishnan Meena et al. (2024)](https://doi.org/10.1063/5.0231929). 

## Significance of the HHL Algorithm
&nbsp; The HHL algorithm represents a monumental breakthrough in quantum computing, allowing for the efficient solution of linear systems of equations, a ubiquitous subtask that underlies numerous scientific and engineering applications. Traditional algorithms struggle with large-scale data sets, often resulting in exponential (O(2<sup>n</sup>)) computational costs. In contrast, the HHL algorithm harnesses the power of quantum mechanics to deliver a polynomial (O(n<sup>2</sup>)) speedup, enabling faster computations that can solve complex problems in fields such as optimization, machine learning, and fluid dynamics!

&nbsp; Understanding the HHL algorithm not only showcases the unique advantages of quantum computing but also opens the door to innovative applications that were previously unimaginable in classical computing. As we work through this course, we will explore the foundational principles and mathematical theory behind the HHL algorithm. For specifics on how quantum computing works, please see our (Python_QML_Basics)[../Python_QML_Basics] challenge.

## Quantum Algorithms Primer

It may be prudent to discuss the differences between classical and quantum algorithms before jumping into specifics. In fact, one common misconception is that quantum computers will outright replace classical computers. The truth is actually much more nuanced! Quantum algorithms are not designed to replace traditional computing algorithms; instead, they excel at solving specific types of problems that classical computers struggle with. See our [`Python_QML_Basics`](../Python_QML_Basics) for a quick review of the differences between classical and quantum computing! Another misconception is that quantum computers can solve problems that classical computers cannot. This is also **not** true! Quantum computers might seem like magic, but they're really only capable of solving classical computations faster and (ideally) with fewer resources than normal computers.

The focus of quantum algorithms is to leverage quantum prinicples such as superposition, entanglement, and quantum interference to perform computations in ways classical computers cannot.



## The HHL Algorithm Code

## Setting Up Our Environment
First, we will move to the challenge directory, unload all current modules you may have previously loaded on Odo, and deactivate any previously loaded environments. 
```
# Move to the challenge directory (assuming you cloned the repo in your home directory)
cd ~/hands-on-with-odo/challenges/Python_QLSA/

# Run only if a previous environment was loaded
source deactivate

# Run regardless of loaded environment
module reset
```
Next, we will load our miniforge module (analagous to an open source minconda), and activate the appropriate conda environment for this exercise.
```
module load miniforge3
source activate /gpfs/wolf2/olcf/stf007/proj-shared/jwine/qlsa_testing/qsla-solver
```
You should now see (qlsa-solver) at the beginning of your terminal prompt.


## HPC Crash Course Challenges

1. Shots-based study
* Objective: Determine the convergence of the accuracy (fidelity) with the number of shots.
* Try changing the shots parameter and see how the fidelity of the results changes.
* Complete the following tasks to solve the tridiagonal Toeplitz matrix problem.
* Run on simulator only (i.e. --backend-method garnet:mock).

* **Task:**
Make a plot that demonstrates the convergence of fidelity for solving matrix of size 2 × 2. Shot range from 100 to 1, 000, 000.
Report your deduction of the converged shot value.

2. Backend evaluation
* Objective: Compare the results for running the circuits on simulators, emulators, and real devices.
* Complete the following tasks to solve the tridiagonal Toeplitz matrix problem.
* Use IQM’s emulator and real device.

* **Task:**
Compare fidelity and uncertainty quantification for various backends (matrix size 2 × 2). Use guidance from `Task
1`. 

**Hint:** [`plot_fidelity_vs_shots.py`](plot_fidelity_vs_shots.py) can be executed after running all of the production runs for every shot and backend combination for Tasks 1 and 2.

## Running the Code

Before running the code, it is important to obtain your IQM token so that you can run the circuits on actual quantum devices.

### Obtaining your IQM Key

1. Log in to your IQM account at https://resonance.meetiqm.com

2. In the "Your Account" menu (click on your initials in the top-right corner), click the circular arrows in the "API Token" section.

3. Generate a new token and copy it. 

4. Open the (`keys.sh`)[keys.sh] file, and replace "my_iqm_api_key" with the token you just copied. (Note: make sure your token is encapsulated by the double-quotes)

5. Save and close the `keys.sh` file. 

6. In your terminal, do the following:
```
$ source keys.sh
```

### (Optional) Testing the Code 

It is also advisable to test the code first to ensure the environment is setup correctly.

1. Test the quantum linear solver package: [`test_linear_solver.py`](test_linear_solver.py)
      ```
      python test_linear_solver.py -nq 2
      ```
      <details><summary>Sample output from the test code:</summary>
      
      ```
      Simulator: AerSimulator('aer_simulator')
      ======================
      Time elapsed for classical:  
      0 min 0.00 sec
      Time elapsed for HHL:  
      0 min 0.21 sec
      ======================
      HHL circuit:
            ┌──────────────┐┌──────┐        ┌─────────┐
      q9_0: ┤0             ├┤4     ├────────┤4        ├
            │  circuit-165 ││      │        │         │
      q9_1: ┤1             ├┤5     ├────────┤5        ├
            └──────────────┘│      │┌──────┐│         │
      q10_0: ───────────────┤0     ├┤3     ├┤0        ├
                            │  QPE ││      ││  QPE_dg │
      q10_1: ───────────────┤1     ├┤2     ├┤1        ├
                            │      ││      ││         │
      q10_2: ───────────────┤2     ├┤1 1/x ├┤2        ├
                            │      ││      ││         │
      q10_3: ───────────────┤3     ├┤0     ├┤3        ├
                            └──────┘│      │└─────────┘
      q11: ─────────────────────────┤4     ├───────────
                                    └──────┘                 
      ====================== 
      Euclidean norm classical:    
      1.237833351044751
      Euclidean norm HHL:        
      1.2099806231118977 (diff (%): 2.250e+00)
      ======================
      Classical solution vector:
      [1.14545455 0.43636364 0.16363636 0.05454545]
      HHL solution vector:
      [1.11266151 0.43866345 0.16004585 0.08942688]
      diff (%): 
      [ 2.86288363  0.52703993  2.1942013  63.94928497]
      ```
      </details>

2. Test Qiskit installation: [`test_qiskit_installation.py`](test_qiskit_installation.py)
      ```
      python test_qiskit_installation.py -backtyp ideal
      ```
      <details><summary>Sample output from the test code:</summary>

      ```
      Backend: QasmSimulator('qasm_simulator')

      Total count for 00 and 11 are: {'00': 494, '11': 506}
              ┌───┐      ░ ┌─┐   
         q_0: ┤ H ├──■───░─┤M├───
              └───┘┌─┴─┐ ░ └╥┘┌─┐
         q_1: ─────┤ X ├─░──╫─┤M├
                   └───┘ ░  ║ └╥┘
      meas: 2/══════════════╩══╩═
                            0  1 
      ```
      </details>

      * Change `-backtyp` for different backends. Make sure to test all backend options offered.
      * **NOTE:** To run using IQM Resonance, you need to add your  IQM API KEY and instance to the [`keys.sh`](keys.sh) file and source activate it.


### Production Run

The instructions below are mainly for **running interactively** on OLCF Odo. The first time you run the Python scripts, it may take some time to load the libraries.

The general workflow is to 1) Start an interactive job (or batch job) to use Odo's compute nodes, 2) Load the appropriate Python `conda` environment, 3) Generate the circuit, 4) Run the QLSA solver with the circuit you just generated, 5) Analyze your results
Run the HHL Circuit

1. Start interactive job
    ```
    salloc -A trn037 -p batch -N 1 -t 1:00:00
    ```

2. Load Python environment:
    * When targeting real quantum backends, you must go through a [proxy server for connecting outside OLCF](https://docs.olcf.ornl.gov/quantum/quantum_software/hybrid_hpc.html#batch-jobs) due to the Odo compute nodes being closed off from the internet by default. 
      ```
      source proxies.sh
      ```
    * First, load the relevant conda module:
      ```
      module load miniforge3
      ```
      How to activate the environment needed for circuit generation and solution:
      ```
      source activate /gpfs/wolf2/olcf/trn037/proj-shared/81a/software/miniconda3-odo/envs/qlsa-circuit 
      ```
3. Run QLSA circuit generator script: [`circuit_HHL.py`](circuit_HHL.py)
    ```
    mkdir models
    srun -N1 -n1 -c1 python circuit_HHL.py -case sample-tridiag -casefile input_vars.yaml --savedata
    ```
    * **NOTE:** Make sure to save the circuit.
    * Make sure to have your `qlsa-circuit` conda environment activated.
    * Try different case settings in the case file [`input_vars.yaml`](input_vars.yaml).

4. Run the QLSA solver: [`solver.py`](solver.py)
    ```
    srun -N1 -n1 -c2 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 --savedata
    ```
    * **NOTE:** Before running the code activate the solver env (`qlsa-solver`).
    * Make sure to have your `qlsa-solver` conda environment activated.
    * Experiment with different parameters in the code.
    * **WARNING:** make sure to save the runs you want with `--savedata` flag; otherwise, you will be unable to generate a plot for the tasks.

5. Plot your results: [`plot_fidelity_vs_shots.py`](plot_fidelity_vs_shots.py)
    ```
    python plot_fidelity_vs_shots.py
    ```
  
> Note: Alternative to all of the above, you can use the batch script [`submit_odo.sh`](submit_odo.sh) to [submit a batch job on OLCF Odo](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#batch-scripts) using `sbatch --export=NONE submit_odo.sh`. The `submit_odo.sh` example batch script is already setup with the above steps; however, modifying that file is required if you want to change any python script arguments.

> Warning: For our purposes, we "hacked" Qiskit's `backend_sampler_v2.py` to workaround IQM returning results in raw strings instead of bytes. The fixed routine is here: `/gpfs/wolf2/olcf/trn037/world-shared/backend_sampler_v2.py`. (in original lines 211/212, switched `num_bytes` to be the length of the string instead)



# References
* A. W. Harrow, A. Hassidim, and S. Lloyd, "Quantum algorithm for linear systems of equations," [Phys. Rev. Lett. 103, 150502](https://doi.org/10.1103/PhysRevLett.103.150502) (2009).
* S. S. Bharadwaj and K. R. Sreenivasan, "Quantum computation of fluid dynamics," [arXiv:2007.09147](https://doi.org/10.48550/arXiv.2007.09147) (2020).
* M. Gopalakrishnan Meena, K. C. Gottiparthi, J. G. Lietz, A. Georgiadou, and E. A. Coello Pérez, "Solving the Hele-Shaw flow using the Harrow-Hassidim-Lloyd algorithm on superconducting devices: A study of efficiency and challenges," [Physics of Fluids, 36 (10): 101705](https://doi.org/10.1063/5.0231929), (2024). ([preprint](http://arxiv.org/abs/2409.10857), [code](https://doi.org/10.5281/zenodo.13738192) - the current repo is adapted from this code)
* [Qiskit - Getting started](https://qiskit.org/documentation/getting_started.html)
* [Qiskit on IQM](https://iqm-finland.github.io/qiskit-on-iqm/user_guide.html)

# Cite this work

* Gopalakrishnan Meena, M., Gottiparthi, K. C., Lietz, J. G., Georgiadou, A., and Coello Pérez, E. A. (2024). Solving the Hele–Shaw flow using the Harrow–Hassidim–Lloyd algorithm on superconducting devices: A study of efficiency and challenges. [Physics of Fluids, 36(10).](https://doi.org/10.1063/5.0231929)
* Gopalakrishnan Meena, M., Gottiparthi, K., & Lietz, J. (2024). qlsa-hele-shaw: Solving the Hele-Shaw flow using the Harrow-Hassidim-Lloyd algorithm on superconducting devices. Zenodo. https://doi.org/10.5281/zenodo.13738192



