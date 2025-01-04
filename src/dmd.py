"""Implementations of the dynamic mode decomposition (DMD)
including the CUR DMD developed by K. Allen and S. De Pascuale."""
import numpy as np
from numpy import linalg as LA
import sklearn.utils.extmath as skmath
from src import maxvol_cur as mc

def cur_dmd(data: np.ndarray,
            rank: int,
            initial_row_indices: np.ndarray = None,
            initial_column_indices: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the column-submatrix-row (CUR) based dynamic mode decomposition (DMD) on n x p time series data.
    If initial row or column indices are not supplied, they will be chosen randomly.

    Parameters:
        data (np.ndarray): Time series data where each column is a time snapshot.
        rank (int): Number of singular values for SVD decomposition
        initial_row_indices(np.ndarray, optional): Initial row indices for maxvol algorithm
        initial_column_indices(np.ndarray, optional): Initial column for maxvol algorithm

    Returns:
        DMD eigenvalues and DMD modes of the dataset.

    Example:
        dmd_eig_values, dmd_modes = dmd(np.random.rand(100, 10), 5)
    """
    # data matrices
    x = data[:,:-1] # All but last column
    y = data[:,1:] # All but first column

    # If the initial indices are not valid or do not exist, get valid ones
    row_indices, column_indices = get_valid_initial_indices(x, rank, initial_row_indices, initial_column_indices)
    optimal_row_indices, optimal_column_indices = mc.alternating_maxvol(x, row_indices, column_indices)
    c, u, r = mc.cur_decomposition(x, optimal_row_indices, optimal_column_indices)

    # Analogous to Koopman operator
    pinv_r = LA.pinv(r)
    a_tilde = LA.pinv(c) @ y @ pinv_r @ u
    dmd_eig_values, w = LA.eig(a_tilde)
    # Multiply in reverse order for efficiency
    dmd_modes = y @ (pinv_r @ (u @ w))

    return dmd_eig_values, dmd_modes

def get_valid_initial_indices(data: np.ndarray,
                              desired_rank: int,
                              initial_row_indices: np.ndarray = None,
                              initial_column_indices: np.ndarray = None,
                              max_attempts: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds valid valid row and column indices for use in the alternating maxvol algorithm.
    If initial_row_indices and initial_column_indices are provided, checks if they correspond to a valid submatrix first.
    Initial indices are considered valid if they correspond to a full rank submatrix in the data.
    The algorithm resamples indicies until a full rank submatrix is found.
    A more efficint implementation is possible, one idea is to first resample indices that correspond to rows/columns of all zeros.
    """
    valid_indices = False
    if initial_row_indices is not None and initial_column_indices is not None:
        try:
            if len(initial_row_indices) != len(initial_column_indices): # checks that initial submatrix is square
                raise ValueError("Initial submatrix is not square.")
            initial_submatrix = data[initial_row_indices,:][:,initial_column_indices]
            initial_submatrix_rank = np.linalg.matrix_rank(initial_submatrix)
            if initial_submatrix_rank==desired_rank:
                valid_indices = True
        except ValueError:
            print("Initial indicies are not valid, finding new indices...")

    n, p = data.shape
    num_attemps = 0
    while not valid_indices:
        if num_attemps > max_attempts:
            raise ValueError(f"Could not find a valid initial submatrix in {max_attempts} attempts.")
        initial_row_indices = np.random.randint(n, size=desired_rank)
        initial_column_indices = np.random.randint(p, size=desired_rank)
        submatrix = data[initial_row_indices,:][:,initial_column_indices]
        submatrix_rank = np.linalg.matrix_rank(submatrix)
        if submatrix_rank==desired_rank: # Rank is valid when submatrix is full rank.
            valid_indices = True
        num_attemps += 1
    return initial_row_indices, initial_column_indices

def svd_dmd(data: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the simgular value decomposition (SVD) based dynamic mode decomposition on n x p time series data.
    The case when rank = p corresponds to the exact DMD.

    Parameters:
        data (np.ndarray): Time series data where each column is a time snapshot.
        rank (int): Number of singular values for SVD decomposition

    Returns:
        DMD eigenvalues and DMD modes of the dataset.

    Example:
        dmd_eig_values, dmd_modes = dmd(np.random.rand(100, 10), 5)
    """
    if rank > min(data.shape):
        raise ValueError("The rank must be smaller than each dimension of the data.")

    # data matrices
    x = data[:,:-1] # All but last column
    y = data[:,1:] # All but first column

    # fixed rank SVD is faster than computing full SVD (Halko, et al. (2009))
    u, sigma, vh = skmath.randomized_svd(x, n_components = rank, random_state = None)

    inv_sigma = 1/sigma
    # Koopman operator
    a_tilde = u.conj().T @ y @ vh.conj().T * inv_sigma

    dmd_eig_values, w = LA.eig(a_tilde)
    # The DMD modes corresponds to eigenvectors of timestep operator a such that ax = y
    dmd_modes = y @ vh.conj().T * inv_sigma @ w

    return dmd_eig_values, dmd_modes

def forecast(data: np.ndarray, rank: int, num_forecasts: int, cur_or_svd: str = 'cur'):
    """Forecast timeseries data using the dynamic mode decomposition of given rank."""
    # TODO: Verify that dmd modes should not be scaled by eigenvalues
    # TODO: refactor multiplying by eigenvalues so it does not multiply by a diagonal matrix.
    if cur_or_svd == 'cur':
        eig_values, dmd_modes = cur_dmd(data, rank)
    elif cur_or_svd == 'svd':
        eig_values, dmd_modes = svd_dmd(data, rank)
    else:
        raise ValueError(f"cur_or_svd must be 'cur' or 'svd'. Instead got {cur_or_svd}.")
    diag_eig_values = np.diag(eig_values)

    initial_condition = data[:,-1] # uses last vector in data for initial condition
    data_dimension = initial_condition.shape[0]
    forecast_results = np.zeros((data_dimension, num_forecasts), dtype=np.complex_)

    # time stepping
    low_dim_forecast = LA.lstsq(dmd_modes, initial_condition, rcond=None)[0]
    for t in range(num_forecasts):
        low_dim_forecast = diag_eig_values @ low_dim_forecast
        forecast_results[:,t] =  dmd_modes @ low_dim_forecast

    forecast_results = forecast_results.real # take real component of results
    return forecast_results
