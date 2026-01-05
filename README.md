# BiFLICKER: Federated Community Detection in Bipartite Networks

This repository contains MATLAB code and data accompanying the paper on FLICKER / BiFLICKER for federated spectral community detection in bipartite networks.

The repo is organized into two main parts:

- `Data/` — real movie-rating bipartite network and covariates.
- `Simulation/` — synthetic experiments and convergence studies.

---

## Folder: `Data/`

This folder contains the real-data objects used in the movie analysis:

- **Movie–user adjacency matrix**  
  A binary matrix \( B \) where  
  - rows correspond to users,  
  - columns correspond to movies,  
  - `B(i,j) = 1` if user *i* rated movie *j*, and `0` otherwise.

- **Movie information matrix**  
  A matrix collecting covariates for each movie, such as:
  - year of production,
  - region,
  - type (e.g., movie vs. TV series),
  - genre (e.g., drama, horror, animation).

- **User data matrix**  
  A matrix with user-level covariates or summary statistics (e.g., number of ratings).

### Main script (real data analysis)

- **`Users.m`**  
  This is the main driver script for the movie data analysis:
  - loads the movie–user adjacency matrix and covariates,
  - runs BiFLICKER on the federated user–movie data,
  - performs clustering on users and movies,
  - generates the summary statistics and plots used in the paper.

To reproduce the real-data analysis, open MATLAB in the repo root and run:

```matlab
Users
```
---

## Folder: `Simulation/`

This folder contains all scripts used for the simulation studies in the paper.

### Overview

The simulations are organized around two main experiments:

1. **Experiment 1 (`Exp1.m`)**  
   Accuracy of singular vector estimation and community detection for different
   combinations of:
   - number of users \(n\),
   - number of servers \(K\),
   - network density (via the parameter \(\rho\)).

2. **Experiment 2 (`Exp2.m`)**  
   Convergence of BiFLICKER as a function of the number of iterations \(T\), for
   different \((n, K)\) settings.

Two plotting scripts generate the figures reported in the paper:

- `Figure1.m` – figures for Experiment 1.  
- `Figure2.m` – figures for the convergence experiment.

### Main scripts

#### `Exp1.m` — Experiment 1 (accuracy comparison)

**Purpose**

- Simulate bipartite SBM data under multiple \((n, K, \rho)\) configurations.
- Run BiFLICKER and competing federated PCA methods.
- Compute:
  - left/right singular subspace errors,
  - misclustering errors for left and right node sets.

**Typical workflow inside `Exp1.m`**

- Define grids for:
  - \(n \in \{2000, 4000, 8000\}\),
  - \(K \in \{5, 10, 20\}\),
  - density parameter \(\rho\).
- For each configuration and for each replication:
  - generate bipartite SBM data,
  - partition rows into \(K\) servers,
  - run:
    - BiFLICKER,
    - DisPCA,
    - DistPCA,
    - FastPCA,
    - oracle (full-data) method,
  - store summary errors (subspace error and misclustering error).

**Output**

- A `.mat` file (or multiple `.mat` files, depending on implementation) containing
  the aggregated errors for all methods and settings.  
  These files are then used by `Figure1.m` to generate plots.

To run:

```matlab
cd Simulation
Exp1
