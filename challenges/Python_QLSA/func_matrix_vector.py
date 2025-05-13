"""
func_matrix_vector
"""
import math
import pickle
import os
from itertools import islice

import yaml
import numpy as np
from scipy.sparse import diags

import func_HeleShaw as HS


#pylint: disable = missing-function-docstring, broad-exception-raised
#pylint: disable = line-too-long, invalid-name, unspecified-encoding

def get_matrix_vector(args):
    if args.case_name == 'sample-tridiag':
        doc_id = 0
        matrix, vector, filename = sample_tridiag(doc_id, args)
    elif args.case_name == 'hele-shaw':
        doc_id = 1
        matrix, vector, filename = Hele_Shaw(doc_id, args)
    else:
        raise Exception('Invalid case. Please see `func_matrix_vector.py` for varoius implementations.')
    return matrix, vector, filename

def get_yaml(input_file, doc_id):
    with open(input_file, 'r') as f:
        docs = yaml.safe_load_all(f)
        input_vars = next(islice(docs, doc_id, None))
    savedir = input_vars['savedir'].format(**input_vars)
    # Check if directory exists and create it if it doesn't
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        print(f"Created save directory: {savedir}")
    return input_vars

def sample_tridiag(doc_id, args):
    input_vars = get_yaml(args.case_variable_file, doc_id)
    print(f"Case: {input_vars['case_name']}")
    filename = input_vars['savefilename'].format(**input_vars)
    n_qubits_matrix = input_vars['NQ_MATRIX']
    # custom systems
    MATRIX_SIZE = 2 ** n_qubits_matrix

    # entries of the tridiagonal Toeplitz symmetric matrix
    a = 1
    b = -1/3
    matrix = diags([b, a, b],
                 [-1, 0, 1],
                 shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
    return matrix, vector, input_vars

def Hele_Shaw(doc_id, args):
    input_vars = get_yaml(args.case_variable_file, doc_id)
    print(f"Case: {input_vars['case_name']}")
    filename = input_vars['savefilename'].format(**input_vars)
    P_in = input_vars['P_in']            # pressure in
    P_out = input_vars['P_out']             # pressure out
    U_top = input_vars['U_top']             # velocity at top
    U_bottom = input_vars['U_bottom']          # velocity at bottom
    L = input_vars['L']                 # length of channel
    D = input_vars['D']                 # width/height of channel
    mu = input_vars['mu']                # fluid viscosity
    rho = input_vars['rho']               # fluid density
    nx = input_vars['nx']          # number of grid points in the x (horizontal) direction
    ny = input_vars['ny']          # number of grid points in the y (vertical/span-wise) direction
    var = input_vars['var']   # which variable to solve for? pressure or velocity
    if ny<3 and nx>2:
        raise Exception ('ny < 3. Due to the current flow setup, the 2nd order finite difference needs more than 2 cells in y-direction.')
        # x-direction doesnot need more than 2 cells as the Laplcaian matrix has
        # space to fill those values.
        # But the resulting matrix is incorrect and thus the solutions are wrong.
    # ## Analytical solution
    print('Solving analytically...')
    x = np.linspace(0, L, nx)
    y = np.linspace(0, D, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)
    P_analytic, U_analytic = HS.HeleShaw(xx ,yy, P_in, P_out, L, D, mu)

    # ## Numerical solution
    print('Solving numerically...')
    # initialize pressure and velocity matrices
    # pressure
    P = np.zeros((ny,nx))
    P[:,0] = P_in
    P[:,-1] = P_out
    # velocity-x
    U = np.zeros((ny,nx))
    U[0,:] = U_bottom
    U[-1,:] = U_top

    # System of equations using computed Laplacian and rhs
    if var == 'pressure':
        print('=====Solving for pressure...=====')
        A, B = HS.Laplacian_pressure(P, P_in, P_out, dx, dy)
    elif var == 'velocity':
        print('=====Solving for velocity...=====')
        A, B = HS.Laplacian_velocity_x(U, U_top, U_bottom, P_analytic, dx, dy)
    else:
        raise Exception('Invalid input for -var or --variable. Enter either prressure or velocity as the flow variables to solve for.')
    # for solving velocity, need to load pressure profile solved using HHL (if available)
    filename_pressure = input_vars['filename_pressure'].format(**input_vars)
    if var == 'velocity' and os.path.isfile(filename_pressure):
        # first find the corresponding pressure case
        file = open(f'{filename_pressure}', 'rb')
        data = pickle.load(file)
        file.close()
        filename_pressure = f"{filename_pressure[:-13]}_circ-fullresults_nq{data['NUM_QUBITS']}_backend-{data['args'].backend_method}_shots{data['args'].SHOTS}.pkl"
        # see if full data is available
        if os.path.isfile(filename_pressure):
            print('Using HHL solved pressure profile...')
            file = open(f'{filename_pressure}', 'rb')
            data = pickle.load(file)
            file.close()
            state_hhl = data['exact_solution_vector']
            state_true_norm = data['classical_solution'].state
            # check if matrix was padded for non-2 power size
            nsys = nx*ny
            nsys_nxtpower = next_power_of_2(nsys)
            if next_power_of_2(nx*ny)-(nx*ny) > 0:
                state_hhl = state_hhl[:-(nsys_nxtpower-nsys)]
                state_true_norm = state_true_norm[:-(nsys_nxtpower-nsys)]
            # check if matrix was Hermitian
            if len(state_hhl) > nx*ny:
                state_hhl = state_hhl[-nx*ny:]
                state_true_norm = state_true_norm[-nx*ny:]
            # print(f'States used for scaling:\nclassical: {state_true_norm}\nHHL: {state_hhl}')
            # Need to rescale pressure as the HHL solution is normalized
            factor = P_analytic[0,0]/state_true_norm.reshape(ny, nx)[0,0]
            offset = P_analytic[0,1] - (state_true_norm.reshape(ny, nx) * factor)[0,1]
            state_pressure_hhl = (state_hhl.reshape(ny, nx) * factor) + offset
            # print(f'Factor: {factor}; Offset: {offset}\nRe-scaled HHL: {state_pressure_hhl}')
            A, B = HS.Laplacian_velocity_x(U, U_top, U_bottom, state_pressure_hhl, dx, dy)
    else:
        print('Using analytical pressure profile...')
    # make sure size of A & B is power of 2
    print(f'Determinant of resulting matrix: {np.linalg.det(A)}\nCondition # of resulting matrix: {np.linalg.cond(A)}')
    # contidion # ??
    A, B = check_size_pow2(A, B)
    A_herm, B_herm = check_matrix_herm(A, B)
    print(f'Reformatted A_herm:\n{A_herm}\nB_herm:\n{B_herm}')
    print(f'Determinant of resulting matrix: {np.linalg.det(A_herm)}\nCondition # of resulting matrix: {np.linalg.cond(A_herm)}')

    if args.savedata is True:
        MATRIX_SIZE = A_herm.shape[0]
        n_qubits_matrix = int(np.log2(MATRIX_SIZE))
        save_data = {'P_in'                  : P_in,
                     'P_out'                 : P_out,
                     'U_top'                 : U_top,
                     'U_bottom'              : U_bottom, 
                     'L'                     : L,
                     'D'                     : D,
                     'mu'                    : mu,
                     'rho'                   : rho,
                     'nx'                    : nx,
                     'ny'                    : ny,
                     'A_herm'                : A_herm,
                     'B_herm'                : B_herm,
                     'n_qubits_matrix'       : n_qubits_matrix,
                     'args'                  : args}
        file = open(f'{filename}_metadata.pkl', 'wb')
        pickle.dump(save_data, file)
        file.close()
        print("===========Metadata saved===========")

    return A_herm, B_herm, input_vars

# Functions
def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def check_size_pow2(A, B):
    """To check if matrix (and vector) is of size power 2 and if not pad with zeros"""
    nsys = A.shape[0]
    # make sure nsys is power of 2
    nsys_nxtpower = next_power_of_2(nsys)
    if nsys_nxtpower-nsys > 0:
        print(f'Size of A & B are not power of 2:\nNext 2 power of {nsys} = {nsys_nxtpower}\nValue to be padded = {nsys_nxtpower-nsys}')
        # Desired number of rows and columns for the resulting square matrix
        desired_size = nsys_nxtpower

        # Calculate the required padding for rows and columns
        padding_rows = max(0, desired_size - A.shape[0])
        padding_cols = max(0, desired_size - A.shape[1])

        top_padding = padding_rows // 2
        bottom_padding = padding_rows - top_padding
        left_padding = padding_cols // 2
        right_padding = padding_cols - left_padding

        # Pad the array with zeros to achieve the desired size
        # padding symmetrically
        #   A_padded = np.pad(A, pad_width=((top_padding, bottom_padding), (left_padding, right_padding)), mode='constant', constant_values=0)
        #   B_padded = np.pad(B, pad_width=(top_padding, bottom_padding), mode='constant', constant_values=0)
        # padding to the bottom and right
        A_padded = np.pad(A, pad_width=((0, top_padding+bottom_padding), (0, left_padding+right_padding)), mode='constant', constant_values=0)
        B_padded = np.pad(B, pad_width=(0, top_padding+bottom_padding), mode='constant', constant_values=0)

        # Filling the diagonals of the padded region with 1s so that the matrix is not singular
        np.fill_diagonal(A_padded[-padding_rows:,-padding_cols:], 1.0)

        # Print the padded array
        print(f'Padded shape of A: {A.shape} -> {A_padded.shape}')
        print(f'Padded shape of B: {B.shape} -> {B_padded.shape}')
        print(f'Padded A with diag 1:\n{A_padded}')
        print(f'Padded B:\n{B_padded}')
        A = A_padded
        B = B_padded
    return A, B

def check_matrix_herm(A, B):
    # check if matrix is Hermitian
    nsys = A.shape[0]
    if np.allclose(A, A.conj().T):
        A_herm = A
        B_herm = B  + 5e-2*np.linalg.norm(B)  # offset for making states non-zero; for error modeling
    else:
        print('Matrix not Hermitian. Making it Hermitian...')
        A_herm = np.hstack( (np.vstack( (np.zeros((nsys,nsys)), A.conj().T) ), np.vstack( (A, np.zeros((nsys,nsys))) )) )
        B_herm = np.hstack( (B, np.zeros(nsys)) ) + 5e-2*np.linalg.norm(B)
    return A_herm, B_herm

def solve_numpylinalg(A, B):
    return np.linalg.solve(A, B)

def solve_qiskitlinalg(A, B, func):
    return func.solve(A, B).state

def post_process_sol(tags, phi_sol, id2rc):
    phi_mapped = np.zeros_like(tags) * np.nan
    for _id, rc in id2rc.items():
        phi_mapped[rc[0], rc[1]] = phi_sol[_id]
    return phi_mapped
