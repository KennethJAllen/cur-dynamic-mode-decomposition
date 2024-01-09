import numpy as np
from numpy import linalg as LA
import sklearn.utils.extmath as skmath

def dmd(data: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Performs Dynamic Mode Decomposition on time series data.

    Parameters:
    data (np.ndarray): Time series data where each column is a time snapshot.
    rank (int): Number of singular values for SVD decomposition.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Eigenvalues and DMD modes of the dataset.

    Example:
    >>> eig_values, dmd_modes = dmd(np.random.rand(100, 10), 5)
    '''
    if rank > min(data.shape):
        raise ValueError("The rank must be smaller than each dimension of the data.")

    # data matrices
    x = data[:,:-1]
    y = data[:,1:]

    # SVD
    u, sigma, vt = skmath.randomized_svd(x, n_components = rank, random_state = None)

    # Koopman operator
    inv_sigma = np.reciprocal(sigma)
    a_tilde = u.T @ y @ vt.T * inv_sigma

    # DMD modes
    eig_values, w = LA.eig(a_tilde)
    dmd_modes = y @ vt.T * inv_sigma @ w # corresponds to eigenvectors of timestep operator a such that ax = y

    return eig_values, dmd_modes

def forecast(data, rank, num_forecasts):
    '''Forecast timeseries data using the dynamic mode decomposition.'''
    eig_values, dmd_modes = dmd(data, rank)
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
    forecast_results[forecast_results<0] = 0 # set negative values in result to zero
    return forecast_results
