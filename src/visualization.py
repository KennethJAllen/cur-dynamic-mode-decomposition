import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE  # Using t-SNE instead of UMAP

def visualize_timeseries_comparison(true_data, pred_data, timestamps=None):
    """
    Visualize comparison between true and predicted high-dimensional time series data
    using t-SNE for dimensionality reduction
    
    Parameters:
    true_data: np.array of shape (n_timepoints, n_dimensions)
    pred_data: np.array of shape (n_timepoints, n_dimensions)
    timestamps: optional array of timestamps
    """
    if timestamps is None:
        timestamps = np.arange(true_data.shape[0])
    
    # 1. Error Heatmap
    plt.figure(figsize=(12, 6))
    errors = np.abs(true_data - pred_data)
    sns.heatmap(errors.T, cmap='YlOrRd')
    plt.title('Absolute Error Heatmap')
    plt.xlabel('Time')
    plt.ylabel('Dimension')
    plt.show()
    
    # 2. t-SNE for dimensionality reduction
    reducer = TSNE(n_components=2, random_state=42)
    # Fit t-SNE on concatenated data to get consistent embeddings
    combined_data = np.vstack([true_data, pred_data])
    combined_embedded = reducer.fit_transform(combined_data)
    
    true_2d = combined_embedded[:true_data.shape[0]]
    pred_2d = combined_embedded[true_data.shape[0]:]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(true_2d[:, 0], true_2d[:, 1], c=timestamps, cmap='viridis', 
                          alpha=0.5, label='True')
    scatter2 = plt.scatter(pred_2d[:, 0], pred_2d[:, 1], c=timestamps, cmap='viridis', 
                          alpha=0.5, marker='x', label='Predicted')
    plt.colorbar(scatter1, label='Time')
    plt.title('t-SNE Projection')
    plt.legend()
    
    # 3. Error Distribution
    plt.subplot(1, 2, 2)
    error_per_dim = np.mean(errors, axis=0)
    plt.hist(error_per_dim, bins=30)
    plt.title('Error Distribution Across Dimensions')
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # 4. Interactive Time Series Plot using Plotly
    fig = go.Figure()
    dim_with_max_error = np.argmax(error_per_dim)
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=true_data[:, dim_with_max_error],
        name='True Values',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=pred_data[:, dim_with_max_error],
        name='Predicted Values',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f'Time Series for Dimension {dim_with_max_error} (Highest Error)',
        xaxis_title='Time',
        yaxis_title='Value'
    )
    fig.show()
    
    # 5. Rolling RMSE
    window_size = 10
    rmse = np.sqrt(np.mean((true_data - pred_data)**2, axis=1))
    rolling_rmse = pd.Series(rmse).rolling(window_size).mean()
    
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, rolling_rmse)
    plt.title(f'Rolling RMSE (Window Size: {window_size})')
    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.show()

if __name__ == "__main__":
    n_timepoints = 100
    n_dimensions = 50

    # Generate sample data
    timestamps = np.linspace(0, 10, n_timepoints)
    true_data = np.zeros((n_timepoints, n_dimensions))
    pred_data = np.zeros((n_timepoints, n_dimensions))

    for d in range(n_dimensions):
        # True values follow a sine wave with different frequencies and phases
        true_data[:, d] = np.sin(2 * np.pi * (d + 1) * timestamps / 10) + np.random.normal(0, 0.1, n_timepoints)
        # Predicted values have some error
        pred_data[:, d] = true_data[:, d] + np.random.normal(0, 0.2, n_timepoints)

    # Visualize the data
    visualize_timeseries_comparison(true_data, pred_data, timestamps)