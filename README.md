# m5-forecasting
The goal of this notebook is to predict time series data from the M5 forecasting competition using the dynamic mode decomposition

## M5 Competition
The M5 competition was a times eries forecasting competition held in 2020, with the objective of advancing the theory and practice of time series forecasting methods. The M5 data, made available by Walmart, consists of the unit sales of $3049$ products sold across $10$ stores over $1941$ days. Thus, our data matrix is $30490 \times 1941$. The goal of the competition is to predict the unit sales of products at each store $28$ days ahead.

## Dynamic Mode Decomposition
The dynamic mode decomposition, or DMD, is a recently developed dimensionality reduction algorithm. Originally developed for fluid dynamics simulation, the DMD was designed to extract features from high dimensional data.

Given a collection of $n$ dimensional data vectors $x_i$, suppose there exists an $n \times n$ linear operator $A$ such that $x_i = A x_{i-1}$. For large $n$, calculating and storing $A$ can be prohibitive. The DMD outputs the leading eigenvectors (known as the DMD modes) and eigenvalues of $A$ without explicitly calculating $A$, and thus allows us to approximate the dynamics of $A$.
