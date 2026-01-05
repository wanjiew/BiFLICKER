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
