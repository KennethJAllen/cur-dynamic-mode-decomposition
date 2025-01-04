# Dynamic Mode Decomposition

## Summary

- An implementation of the columns-submatrix-row (CUR)-based dynamic mode decomposition (DMD).

- The CUR DMD provides a fast and interpretable framework for generating and compressing timeseries dynamics.

- Originally developed by Kenneth Allen and Sebastian De Pascuale in 2021 as part of the SCGSR research fellowship with results published in [[1]](#references).

## 🔧 Installation

### Clone the Repository:

```
git clone https://github.com/KennethJAllen/dynamic-mode-decomposition
cd dynamic-mode-decomposition
```
### Install Dependencies with UV:

*   Install UV if not already installed.
*   Run the following command in the project directory:

```
uv sync
```
## Dynamic Mode Decomposition
The dynamic mode decomposition, originally developed for simulating fluid dynamics, was designed to extract features from high dimensional data.

Given a collection of $n$ dimensional data vectors $x_i$, suppose the dynamics evolve linearly. That is, there exists an $n \times n$ linear operator $A$ such that $x_i = A x_{i-1}$. For large $n$, calculating and storing $A$ can be prohibitive. The DMD outputs the leading eigenvectors (known as the DMD modes) and eigenvalues of $A$ without explicitly calculating $A$, and thus allows us to approximate the dynamics of $A$.

Demonstrates the dynamic mode decomposition by forecasting time series data from the M5 forecasting competition. The dynamic mode decomposition is a dimensionality reduction algorithm developed by Peter Schmid in 2008.

### SVD Based Dynamic Mode Decomposition

### CUR Based Dynamic Mode Decomposition

## Examples

### M5 Competition
The M5 competition was a times series forecasting competition held in 2020, with the objective of advancing the theory and practice of time series forecasting methods. The M5 data, made available by Walmart, consists of the unit sales of $3049$ products sold across $10$ stores over $1941$ days. Thus, our data matrix is $30490 \times 1941$. The goal of the competition is to predict the unit sales of products at each store $28$ days ahead.

## Sources

- [1] De Pascuale, Sebastian, et al. "Compression of tokamak boundary plasma simulation data using a maximum volume algorithm for matrix skeleton decomposition." Journal of Computational Physics 484 (2023): 112089.

- [2] Allen, Kenneth. A geometric approach to low-rank matrix and tensor completion. Diss. University of Georgia, 2021.

- [3] Tu, Jonathan H. Dynamic mode decomposition: Theory and applications. Diss. Princeton University, 2013.
