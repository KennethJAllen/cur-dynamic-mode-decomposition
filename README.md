# Dynamic Mode Decomposition

## Introduction
The goal of this notebook is to demonstrate the dynamic mode decomposition by forecasting time series data from the M5 forecasting competition. The dynamic mode decomposition is a dimensionality reduction algorithm developed by Peter Schmid in 2008.

## 🔧 Installation

### Clone the Repository:

```
git clone https://github.com/KennethJAllen/dynamic-mode-decomposition
cd dynamic-mode-decomposition
```
### Install Dependencies with Poetry:

*   Install Poetry if not already installed.
*   Run the following command in the project directory:

```
poetry install
```
### Activate the Virtual Environment:
```
poetry shell
```
You can now run the project's scripts within the poetry shell.

## Dynamic Mode Decomposition
The dynamic mode decomposition, originally developed for fluid dynamics simulation, was designed to extract features from high dimensional data.

Given a collection of $n$ dimensional data vectors $x_i$, suppose there exists an $n \times n$ linear operator $A$ such that $x_i = A x_{i-1}$. For large $n$, calculating and storing $A$ can be prohibitive. The DMD outputs the leading eigenvectors (known as the DMD modes) and eigenvalues of $A$ without explicitly calculating $A$, and thus allows us to approximate the dynamics of $A$.

## M5 Competition
The M5 competition was a times series forecasting competition held in 2020, with the objective of advancing the theory and practice of time series forecasting methods. The M5 data, made available by Walmart, consists of the unit sales of $3049$ products sold across $10$ stores over $1941$ days. Thus, our data matrix is $30490 \times 1941$. The goal of the competition is to predict the unit sales of products at each store $28$ days ahead.
