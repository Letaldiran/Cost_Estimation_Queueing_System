# Multi-Level Efficiency Queueing Systems: Cost Estimation

This repository contains the mathematical framework and Python implementations for estimating the expected operating costs per time unit in queueing systems with multiple efficiency levels. The project bridges the gap between discrete-time Markov chains and continuous-time stochastic differential equations (SDEs) with reflection.

## 📌 Project Overview

The core objective of this research is to derive and verify analytical solutions for cost functions in complex queueing environments. The project is divided into two main theoretical frameworks:

1.  **Discrete Domain:** Analysis via an induced Markov chain and a specialized modification of the Ergodic Theorem.
2.  **Continuous Domain:** Analysis via SDEs with reflection and diffusion processes with drift.

Numerical simulations are implemented in Python to compare the accuracy and computational efficiency of these analytical derivations against empirical methods like Monte Carlo simulations and transition matrix power iterations.

---

## 🏗 System Architecture & Case Studies

The codebase covers four distinct scenarios to provide a comprehensive comparison of the system's behavior:

| Case | Domain | Methodology | Reference |
| :--- | :--- | :--- | :--- |
| **1** | Discrete | Analytical Solution | Equation (1.20) |
| **2** | Discrete | Matrix Convergence | Stationary distribution via $P^n$ convergence |
| **3** | Continuous | Analytical Solution | Equation (2.19) |
| **4** | Continuous | Monte Carlo | Time-$T$ simulation and cost distribution tracking |

## 💻 Implementation Details

The simulations are implemented in **Python**, focusing on:
* **Transition Matrix Multiplications:** Efficiently finding the stationary distribution for discrete cases.
* **Monte Carlo Techniques:** Simulating continuous paths over time $T$ to track the distribution of the cost function.
* **Performance Benchmarking:** A side-by-side comparison of analytical vs. empirical results in terms of convergence speed and error margins.

### Prerequisites
* Python 3.8+
* NumPy
* SciPy
  
---

## 📈 Results summary
The implementation demonstrates that the analytical solutions (1.20 and 2.19) align closely with the empirical results. While the Monte Carlo method is highly flexible for general distributions, the analytical approach offers significant gains in computational efficiency for real-time cost estimation.

---
*This project was developed as part of a Master's Diploma in Mathematics.*
