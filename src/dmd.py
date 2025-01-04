import numpy as np
from numpy import linalg as LA
import sklearn.utils.extmath as skmath

def svd_dmd(data: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Performs the Dynamic Mode Decomposition on n x p time series data.
    The case when rank = p corresponds to the exact DMD.

    Parameters:
        data (np.ndarray): Time series data where each column is a time snapshot.
        rank (int): Number of singular values for SVD decomposition

    Returns:
        Tuple[np.ndarray, np.ndarray]: Eigenvalues and DMD modes of the dataset.

    Example:
        dmd_eig_values, dmd_modes = dmd(np.random.rand(100, 10), 5)
    '''
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

def forecast(data: np.ndarray, rank: int, num_forecasts: int):
    '''Forecast timeseries data using the dynamic mode decomposition of given rank.'''
    eig_values, dmd_modes = svd_dmd(data, rank)
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
