"""
func_HeleShaw
"""

#pylint: disable = invalid-name, missing-function-docstring

import numpy as np

def ij2idx(row_i, column_j, column):
    return (row_i*column) + column_j

def Laplacian_pressure(P, P_left, P_right, dx, dy):
    ny, nx = P.shape
    LP = np.zeros((ny*nx, ny*nx))
    b  = np.zeros((ny*nx))
    for i in range(ny):
        for j in range(nx):
            ii = ij2idx(i, j, nx)
            if j==0: # left boundary middle cells
                # forward in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += 1
                b[ii] = P_left
            elif j==nx-1: # right boundary middle cells
                # backward in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += 1
                b[ii] = P_right
            elif i==0: # bottom boundary
                if j==0: # left-bottom corner
                    # forward in x and forward in y
                    LP[ii,ij2idx(i  ,j  , nx)] += 1
                    b[ii] = P_left
                elif j==nx-1: # right-bottom corner
                    # backward in x and forward in y
                    LP[ii,ij2idx(i  ,j  , nx)] += 1
                    b[ii] = P_right
                else: # bottom middle cells
                    # central in x and forward in y
                    LP[ii,ij2idx(i  ,j  , nx)] += -2/(dx**2)
                    LP[ii,ij2idx(i  ,j-1, nx)] += 1/(dx**2)
                    LP[ii,ij2idx(i  ,j+1, nx)] += 1/(dx**2)
                    LP[ii,ij2idx(i  ,j  , nx)] += 1/(dy**2)
                    LP[ii,ij2idx(i+1,j  , nx)] += -2/(dy**2)
                    LP[ii,ij2idx(i+2,j  , nx)] += 1/(dy**2)
                    b[ii] = 0
            elif i==ny-1: # top boundary
                if j==0: # left-top corner
                    # forward in x and backward in y
                    LP[ii,ij2idx(i  ,j  , nx)] += 1
                    b[ii] = P_left
                elif j==nx-1: # right-top corner
                    # backward in x and backward in y
                    LP[ii,ij2idx(i  ,j  , nx)] += 1
                    b[ii] = P_right
                else: # top middle cells
                    # central in x and backward in y
                    LP[ii,ij2idx(i  ,j  , nx)] += -2/(dx**2)
                    LP[ii,ij2idx(i  ,j-1, nx)] += 1/(dx**2)
                    LP[ii,ij2idx(i  ,j+1, nx)] += 1/(dx**2)
                    LP[ii,ij2idx(i  ,j  , nx)] += 1/(dy**2)
                    LP[ii,ij2idx(i-1,j  , nx)] += -2/(dy**2)
                    LP[ii,ij2idx(i-2,j  , nx)] += 1/(dy**2)
                    b[ii] = 0
            else: # middle cells
                # central in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j-1, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j+1, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                b[ii] = 0
    return LP, b


def Laplacian_velocity_x(U, U_top, U_bottom, P, dx, dy):
    ny, nx = U.shape
    LP = np.zeros((ny*nx, ny*nx))
    b  = np.zeros((ny*nx))
    for i in range(ny):
        for j in range(nx):
            ii = ij2idx(i, j, nx)
            if i==0: # bottom boundary
                LP[ii,ij2idx(i  ,j  , nx)] += 1
                b[ii] = U_bottom
            elif i==ny-1: # top boundary
                LP[ii,ij2idx(i  ,j  , nx)] += 1
                b[ii] = U_top
            elif j==0: # left boundary middle cells
                # forward in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j+1, nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j+2, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                px_ii = (P[i,j+1] - P[i,j]) / dx
                b[ii] = px_ii
            elif j==nx-1: # right boundary middle cells
                # backward in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j-1, nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j-2, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                px_ii = (P[i,j] - P[i,j-1]) / dx
                b[ii] = px_ii
            else: # middle cells
                # central in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j-1, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j+1, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                px_ii = 0.5 * (P[i,j+1] - P[i,j-1]) / dx
                b[ii] = px_ii
    return LP, b

def Laplacian_velocity_y(V, V_top, V_bottom, P, dx, dy):
    ny, nx = V.shape
    LP = np.zeros((ny*nx, ny*nx))
    b  = np.zeros((ny*nx))
    for i in range(ny):
        for j in range(nx):
            ii = ij2idx(i, j, nx)
            if i==0: # bottom boundary
                LP[ii,ij2idx(i  ,j  , nx)] += 1
                b[ii] = V_bottom
            elif i==ny-1: # top boundary
                LP[ii,ij2idx(i  ,j  , nx)] += 1
                b[ii] = V_top
            elif j==0: # left boundary middle cells
                # forward in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j+1, nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j+2, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                py_ii = 0.5 * (P[i+1,j] - P[i-1,j])/(dy**2)
                b[ii] = py_ii
            elif j==nx-1: # right boundary middle cells
                # backward in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j-1, nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j-2, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                py_ii = 0.5 * (P[i+1,j] - P[i-1,j])/(dy**2)
                b[ii] = py_ii
            else: # middle cells
                # central in x and central in y
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dx**2)
                LP[ii,ij2idx(i  ,j-1, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j+1, nx)] += 1/(dx**2)
                LP[ii,ij2idx(i  ,j  , nx)] += -2/(dy**2)
                LP[ii,ij2idx(i-1,j  , nx)] += 1/(dy**2)
                LP[ii,ij2idx(i+1,j  , nx)] += 1/(dy**2)
                py_ii = 0.5 * (P[i+1,j] - P[i-1,j])/(dy**2)
                b[ii] = py_ii
    return LP, b

# # Comparing with the analytical solution
def HeleShaw(x, y, pin, pout, L, D, mu):
    '''Analytical solution to the Hele-Shaw flow'''
    P = pin + ((pout-pin)*x/L)
    U = -(0.5/mu) * ( (pout-pin) * y * (D-y) / L)
    return P, U
