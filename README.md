# CUR Dynamic Mode Decomposition

## Summary

- An implementation of the columns-submatrix-row (CUR)-based dynamic mode decomposition (DMD).

- The CUR DMD provides a fast and interpretable framework for generating and compressing timeseries dynamics.

- Originally developed by K. Allen and S. De Pascuale in 2021 as part of the SCGSR research fellowship with results published in [[1]](#references).

## Installation

### Clone the Repository:

```
git clone https://github.com/KennethJAllen/dynamic-mode-decomposition
cd dynamic-mode-decomposition
```

### Install Dependencies with UV:

*   [Install UV](https://docs.astral.sh/uv/#highlights) if not already installed.
*   Run the following command in the project directory:

```
uv sync
```
## Using the Code
To forecast timeseries data, use the `forecast` function in `dmd.py`.

`forecast` takes the following parameters:

- `data`: A numpy array of data to forecast.
- `rank`: The rank of the low rank approximation to use the in DMD.
- `num_forecasts`: The number of time steps to forecast out.
- `cur_or_svd`: A string `cur` for the CUR based DMD or `svd` for the SVD based DMD.

## Dynamic Mode Decomposition
The dynamic mode decomposition, originally developed for simulating fluid dynamics, was designed to extract features from high dimensional data.

Given a collection of data vectors $\{z_0, \dots, z_n\}$, suppose the dynamics evolve linearly. That is, there exists an $n \times n$ linear operator $A$ such that $z_i = A z_{i-1}$.

Letting

$$X = [z_0, \dots, z_{n-1}]$$

$$Y = [z_1, \dots, z_n]$$

We can define

$$A = YX^\dagger$$

where $X^\dagger$ is the pusedoinverse of $X$.

For large $n$, calculating and storing $A$ can be prohibitive. The DMD outputs the leading eigenvectors (known as the DMD modes) and eigenvalues of $A$ without explicitly calculating $A$, and thus allows us to approximate the dynamics of $A$.

Instead of calculating the full $A$, we can calculate a smaller matrix $\tilde{A}$ that can approximate the dynamics of $A$ via dimensionality reduction.

### SVD Based Dynamic Mode Decomposition

The traditional way of computing $\tilde{A}$ is with the SVD.

Given a desired rank $r$, calculate $X_r = U \Sigma V^\*$, where $\Sigma$ consists of the top $r$ singular values and $U$ and $V^\*$ are the corresponding singular vectors.

Then the $r \times r$ matrix $\tilde{A}$ is given by

$$\tilde{A} = U^\*YV\Sigma^{-1}.$$

To forecast the data vectors $\{z_i\}$, we have the formula

$$\hat{z}_{n+N} = U \tilde{A}^N U^\* z_n.$$

For theoretical results, see [[3]](#references).

### CUR Based Dynamic Mode Decomposition

In practice, the SVD may be computationally prohibitive for very high dimensional data sets. While the SVD gives the best low-rank approximation, we can trade off error in the low rank approximation for computational speed.

We use the CUR decomposition for a low rank approxmation.

Let $X_r = C U^{-1} R$, where $C, U, R$ can be chosen via a maximum volume algorithm. Analogous to the SVD case, we can define

$$\tilde{A} = C^\dagger Y R^\dagger U$$

To forecast the data vectors $\{z_i\}$, we have the formula

$$\hat{z}_{n+N} = C \tilde{A}^N C^\dagger z_n.$$

For theoretical results, see [[2]](#references).

## Examples

### M5 Competition
The M5 competition was a times series forecasting competition held in 2020, with the objective of advancing the theory and practice of time series forecasting methods. The M5 data, made available by Walmart, consists of the unit sales of 3049 products sold across 10 stores over 1941 days. Thus, our data matrix is $30490 \times 1941$. The goal of the competition is to predict the unit sales of products at each store 28 days ahead.

## Sources

- [1] De Pascuale, Sebastian, et al. "Compression of tokamak boundary plasma simulation data using a maximum volume algorithm for matrix skeleton decomposition." Journal of Computational Physics 484 (2023): 112089.

- [2] Allen, Kenneth. A geometric approach to low-rank matrix and tensor completion. Diss. University of Georgia, 2021.

- [3] Tu, Jonathan H. Dynamic mode decomposition: Theory and applications. Diss. Princeton University, 2013.
