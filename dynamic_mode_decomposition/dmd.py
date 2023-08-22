import numpy as np
from numpy import linalg as LA
from sklearn.utils.extmath import randomized_svd

def dmd(data, rank):
    '''Dynamic mode decomposition.'''
    #data matrices
    X = data[:,:-1]
    Y = data[:,1:]

    #SVD
    U, Sigma, VT = randomized_svd(X, n_components=rank, random_state=None)

    #Koopman operator
    inv_Sigma = np.reciprocal(Sigma)
    A_tilde = U.T @ Y @ VT.T * inv_Sigma

    #DMD modes
    eig_values, W = LA.eig(A_tilde)
    dmd_modes = Y @ VT.T * inv_Sigma @ W #corresponds to eigenvectors of timestep operator A such that AX=Y
    
    return eig_values, dmd_modes

def forecast(data, rank, num_forecasts):
    '''Forecast timeseries data using the dynamic mode decomposition.'''
    eig_values, dmd_modes = dmd(data, rank)
    diag_eig_values = np.diag(eig_values)

    initial_condition = data[:,-1] #uses last vector in data for initial condition
    data_dimension = initial_condition.shape[0]
    forecast_results = np.zeros((data_dimension, num_forecasts), dtype=np.complex_)

    #time stepping
    low_dim_forecast = LA.lstsq(dmd_modes, initial_condition, rcond=None)[0]
    for t in range(num_forecasts):
        low_dim_forecast = diag_eig_values @ low_dim_forecast
        forecast_results[:,t] =  dmd_modes @ low_dim_forecast
    forecast_results = forecast_results.real
    forecast_results[forecast_results<0] = 0 #set negative values in result to zero
    return forecast_results