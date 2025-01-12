"""A fluid dynamics forecasting demo for the CUR and SVD based dynamic mode decompositions."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from src import dmd

def generate_fluid_data(dims: tuple[int] = (100, 100), nt: int=200, noise_level=0.02) -> tuple[np.ndarray]:
    """
    Generate synthetic fluid dynamics data consisting of spatial modes that evolve over time.
    
    Parameters:
    -----------
    dims : tuple[int]
        Number of spatial points in x and y directions
    nt : int
        Number of timesteps
    noise_level : float
        Standard deviation of Gaussian noise to add
        
    Returns:
    --------
    data : np.ndarray
        Array of shape (nx*ny, nt) containing the synthetic data
    meshgrid : tuple[np.ndarray]
        Meshgrid = X, Y arrays for plotting
    """
    nx, ny = dims
    # Create spatial grid
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    meshgrid = np.meshgrid(x, y)
    X, Y = meshgrid

    # Time vector
    t = np.linspace(0, 4*np.pi, nt)

    # Initialize data array
    data = np.zeros((nx*ny, nt))

    # Create evolving spatial modes
    for tau in range(nt):
        # First mode: rotating vortex
        mode1 = np.exp(-((X-np.cos(t[tau]))**2 + (Y-np.sin(t[tau]))**2))

        # Second mode: standing wave
        mode2 = np.sin(2*np.pi*X/2)*np.cos(t[tau])

        # Third mode: traveling wave
        mode3 = np.sin(2*np.pi*(X - 0.5*t[tau]))

        # Combine modes with different weights
        field = 1.0*mode1 + 0.5*mode2 + 0.3*mode3

        # Add noise
        noise = noise_level * np.random.randn(nx, ny)
        field += noise

        # Reshape to column vector and store
        data[:, tau] = field.reshape(-1)

    return data, meshgrid

def calculate_forecast_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Calculates the error at each time step between actual data and forecasted data.
    input:5
        actual (np.ndarray): n x t actual timeseries data
        predicted (np.ndarray): n x t forecasted timeseries data
    output:
        error (np.ndarray): size t errors at each timestep
    """
    assert actual.shape == predicted.shape
    return np.mean(np.abs(actual-predicted), axis=1)

def create_animation(forecasts: dict[str,np.ndarray],
                     meshgrid: tuple[np.ndarray],
                     animation_title: str,
                     errors: dict[str,np.ndarray] = None,
                     interval: int=50) -> None:
    """
    Create an animation of the fluid field evolution.
    
    Parameters:
    -----------
    forecasts : dict[str,np.ndarray]
        Dictionary of named datasets to compare.
        Each dataset is of shape (nx*ny, nt).
    meshgrid : tuple[np.ndarray]
        meshgrid = X, Y arrays for plotting.
    errors : dict[np.ndarray], optional
        Dictionary with same keys as forecasts, containing error values for each frame.
        Each array should be of length nt.
    animation_title : str
        Animation title
    interval : int
        Delay between frames in milliseconds
    """
    X, Y = meshgrid

    # Set up the figure
    n_plots = len(forecasts)
    fig_width = min(5, 20/n_plots) * n_plots
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))

    axes = []
    pcms = []

    for index, (title, forecast) in enumerate(forecasts.items()):
        ax = fig.add_subplot(1, len(forecasts), index+1)
        axes.append(ax)

        field = forecast[:, 0].reshape(X.shape)
        pcm = ax.pcolor(X, Y, field, shading='auto')
        pcms.append(pcm)

        ax.set_title(title)
        ax.axis('equal')

    #fig.colorbar(pcms[0], ax=axes, label='Field Value')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Creates space for the suptitle
    fig.suptitle(animation_title)

    # Animation update function
    def update(frame: int) -> list:
        """Animation update function"""
        for (title, forecast), pcm, ax in zip(forecasts.items(), pcms, axes):
            if frame >= forecast.shape[1]:
                # Only update if frame exists
                continue
            field = forecast[:, frame].reshape(X.shape)
            pcm.set_array(field.ravel())
            if errors is not None and title in errors:
                error = errors[title][frame]
                title_with_error = f"{title} Error: {error:.1e}"
                ax.set_title(title_with_error)
        return pcms

    # Create animation
    num_frames = min(forecast.shape[1] for forecast in forecasts.values())
    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=True
    )

    out_dir = Path(__file__).parent / 'fluid_evolutions'
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = animation_title.lower().replace(' ', '_') + '.gif'
    file_path = out_dir / file_name

    writer = PillowWriter(fps=20)
    anim.save(file_path, writer=writer)
    plt.close(fig)

def main():
    """
    Generates synthetic fluid data.
    Forecasts with the CUR based DMD and the SVD based DMD.
    Saves forecasts as gifs.
    """
    # Generate Synthetic Fluid Data
    total_num_timesteps = 200
    fluid_data, meshgrid = generate_fluid_data(nt = total_num_timesteps)

    # Save original data
    create_animation({'Original Data': fluid_data}, meshgrid, animation_title='Synthetic Fluid Data')

    # Example of comparing original and forecasted data
    # Generate forecast
    num_training_timesteps = 150 # Use first num_training_steps timesteps
    training_data = fluid_data[:, :num_training_timesteps]
    rank = 15
    num_forecasts = total_num_timesteps - num_training_timesteps
    actual_data = fluid_data[:,num_training_timesteps:]
    # Forecast data
    actual_data_title = "Original Fluid Evolution"
    dmd_svd_title = "SVD DMD Forecast"
    dmd_cur_title = "CUR DMD Forecast"
    dmd_svd_forecast = dmd.forecast(training_data, rank=rank, num_forecasts=num_forecasts, cur_or_svd='svd')
    dmd_cur_forecast = dmd.forecast(training_data, rank=rank, num_forecasts=num_forecasts, cur_or_svd='cur')
    forecasts = {actual_data_title: actual_data,
                 dmd_svd_title: dmd_svd_forecast,
                 dmd_cur_title: dmd_cur_forecast}
    dmd_svd_error = calculate_forecast_error(actual_data, dmd_svd_forecast)
    dmd_cur_error = calculate_forecast_error(actual_data, dmd_cur_forecast)
    errors = {dmd_svd_title: dmd_svd_error,
              dmd_cur_title: dmd_cur_error}
    # Save forecasts as gifs
    forecast_title = f"Rank {rank} DMD Forecasts"
    create_animation(forecasts, meshgrid, animation_title=forecast_title, errors=errors)

# Example usage:
if __name__ == "__main__":
    main()
