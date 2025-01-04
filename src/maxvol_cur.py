"""An implementation of the CUR decomposition, with a way of finding the optimal indices."""
import numpy as np
from numpy import linalg as LA

def cur_decomposition(data: np.ndarray,
                      row_indices: np.ndarray,
                      column_indices: np.ndarray,
                      validate_rank: bool = False) -> tuple[np.ndarray]:
    """
    Returns the CUR decomposition of data with respect to the row and column indices.
    Optimal row and column indices are given by the maximum volume algorithm
    data is approximated by C @ inv(U) @ R, with equality when the rank of the data is equal to the number of indices.
    """
    num_rows = len(row_indices)
    if num_rows != len(column_indices):
        raise ValueError("Number or row and column indices must be equal.")
    columns = data[:, column_indices] # C
    submatrix = data[row_indices, :][:, column_indices] # U
    rows = data[row_indices, :] # R
    if validate_rank:
        submatrix_rank = np.linalg.matrix_rank(submatrix)
        if submatrix_rank != num_rows:
            print(f"Warning: Submatrix in row, column indices is not invertible. Rank {submatrix_rank} when expecting rank {num_rows}.")
    return columns, submatrix, rows

def alternating_maxvol(data: np.ndarray,
                       initial_row_indices: np.ndarray,
                       initial_column_indices: np.ndarray,
                       validate_rank: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Algorithm based on paper "How to Find a Good Submatrix" by Goreinov, Sergei A., et al.
    Alternating version of the one directional maxvol algorithm, modified for two-directional search.
    Finds a close to dominant r x r submatrix of data matrix.

    The submatrix in the resulting indices is the corresponding close to maximum volume submatrix.
    In other words, if A = data[row_indices,:][:,column_indices],
    then abs(det(A)): is close to maximum over all choices of submatrixes.
    """
    row_indices = initial_row_indices.copy()
    column_indices = initial_column_indices.copy()
    error_threshold = 1e-8 # tolerance

    num_indices = len(row_indices)
    if num_indices != len(column_indices): # checks that initial submatrix is r x r
        raise ValueError("initial submatrix is not r x r")

    submatrix = data[row_indices,:][:,column_indices] # initial submatrix
    if validate_rank:
        initial_rank = np.linalg.matrix_rank(submatrix)
        if initial_rank != num_indices: # initial submatrix should be nonsingular
            print(f"Warning: Submatrix in positions I and J is not invertible. Rank {initial_rank} when expecting rank {num_indices}.")

    row_dom = False # indicates if near dominant in rows
    column_dom = False # indicates if near dominant in columns

    max_num_iterations = 1000 #
    for k in range(max_num_iterations):
        Yh = LA.solve(submatrix.T, data[:,column_indices].T)
        Y = Yh.T # Y =  data[:,J]A^{-1}
        Ya = np.abs(Y) # entry-wise absolute value of Y
        y = np.amax(Ya) # largest element of Y in modulus
        if y > 1+error_threshold: # if A is not within the acceptable tolerance of a dominant submatrix in columns
            position_y = np.where(Ya == y) # indices of maximum element in Ya
            i = position_y[0][0]
            j = position_y[1][0] # (i,j) are the coordinates of y in Ya
            row_indices[j] = i # replaces jth row of A with the ith row of data[:,J]
            submatrix = data[row_indices,:][:,column_indices]
            column_dom = False # not near dominant in columns
        elif row_dom: # if near dominant in both rows and columns
            break
        else:
            column_dom = True # indicates that A is near dominant in columns

        Z = LA.solve(submatrix,data[row_indices,:]) #Z = A^{-1}data[I,:]
        Za = np.abs(Z) # entry-wise absolute value of Z
        z = np.amax(Za) # largest element of Z in modulus
        if z > 1+error_threshold: # if A is not within the acceptable tolerance of a dominant submatrix in rows
            position_z = np.where(Za == z) # indices of maximum element in Za
            p = position_z[0][0]
            q = position_z[1][0] # (p,q) are the coordinates of z in Za
            column_indices[p] = q # replace pth column of A with qth row of data[I,:]
            submatrix = data[row_indices,:][:,column_indices]
            row_dom = False # not near dominant in rows
        elif column_dom: # if near dominant in both rows and columns
            break
        else:
            row_dom = True # indicates that A is near dominant in rows

        if k == max_num_iterations-1:
            raise ValueError(f"alt_maxvol did not converge in {max_num_iterations} steps")
    return row_indices, column_indices
