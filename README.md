# BiFLICKER: Federated Community Detection in Bipartite Networks

This repository contains code and aggregate reproducible results accompanying
the BiFLICKER project on federated spectral community detection in bipartite
networks. Row-level third-party data are not redistributed.

## Algorithm variants

The repository contains two related estimators:

- **BiFLICKER** estimates the singular directions associated with different
  target singular values independently. The directions can therefore be
  computed in parallel when the targets are sufficiently separated. This is
  the version used in the simulation study and the movie-rating application.
- **Sequential BiFLICKER** estimates target directions in descending order and
  projects each iterate onto the orthogonal complement of the previously
  recovered directions. This extension is intended for spectra with small
  eigengaps, where nearby targets may otherwise recover the same direction. It
  is the version used in the Last.fm 1K application.

## Repository structure

```text
.
├── Data/                         movie-rating application
├── Simulation/                   synthetic experiments
└── Application/
    └── LastFM1K/
        └── Final/                Last.fm 1K application
```

## Movie-rating application (`Data/`)

`Data/` contains the MATLAB code used for the movie-rating analysis. The
processed user–movie network and user/movie covariates are not redistributed;
users must obtain or construct authorized copies separately.

Main files:

- `Users.m`: main user-community analysis;
- `Movie.m`: movie-side analysis;
- `BiFLICKER.m`: BiFLICKER implementation used by the application.

`Users.m` expects locally supplied `adj_matrix_combined.csv`,
`movie_info_combined.csv`, and `user_metadata.csv`. These filenames are ignored
by Git to prevent accidental redistribution. Data provenance and permission
must be documented before any public data release.

Run the main analysis from MATLAB:

```matlab
cd Data
Users
```

## Simulation study (`Simulation/`)

`Simulation/` contains the MATLAB implementation and scripts for two synthetic
experiments:

- `Exp1.m`: singular-vector and community-detection accuracy over grids of
  network size, number of servers, and density;
- `Exp2.m`: convergence as the number of BiFLICKER iterations changes;
- `Figure1.m` and `Figure2.m`: construct the corresponding paper figures;
- `BiFLICKER.m` and `BiFLICKER_conv.m`: main estimators; and
- `DistPCA.m`, `disPCA.m`, and `FastPCA.m`: comparison methods. The
  `disPCA.m` implementation is based on Liang et al. (2014), as cited below.

Run from MATLAB:

```matlab
cd Simulation
Exp1
Exp2
Figure1
Figure2
```

The large generated `.mat` files are intentionally not stored in Git. They can
be regenerated from `Exp1.m` and `Exp2.m`.

## Last.fm 1K application (`Application/LastFM1K/Final/`)

This application constructs a cleaned binary user–artist bipartite network,
splits users into country servers, and compares local, centralized, and
Sequential BiFLICKER community detection. It also contains the final
demographic, listening-behavior, representative-artist, and community-genre
analyses.

The directory includes its own detailed documentation:

- [Last.fm 1K application README](Application/LastFM1K/Final/README.md)

Neither the raw Last.fm dataset nor the processed row-level network and
embeddings are included. Users must download the raw files separately and
follow the attribution and non-commercial-use terms in the application README.

To reproduce the application:

```bash
cd Application/LastFM1K/Final
python3 scripts/01_prepare_lastfm1k.py \
  --raw-dir /absolute/path/to/lastfm-dataset-1K
Rscript scripts/02_spectral_community_analysis.R
Rscript scripts/03_country_label_ari_figure.R
Rscript scripts/04_user_community_interpretation.R
```

## Software requirements

- MATLAB for `Data/` and `Simulation/`;
- Python 3 for Last.fm preprocessing;
- R and the R package `Matrix` for the Last.fm spectral analyses and figures.

## License and data terms

The repository's original software is released under the [MIT License](LICENSE).
That license applies to the code, not to third-party datasets. Dataset users
remain responsible for the terms and attribution requirements of the original
data providers. Row-level movie and Last.fm data are not redistributed here.

## Method reference

The `disPCA` simulation baseline is an independent implementation of the
distributed PCA algorithm described in:

> Maria-Florina F. Balcan, Vandana Kanchanapally, Yingyu Liang, and David P.
> Woodruff. “Improved Distributed Principal Component Analysis.” *Advances in
> Neural Information Processing Systems 27*, 2014.
> [Paper](https://proceedings.neurips.cc/paper_files/paper/2014/hash/e968f1646c1c6c35422b64c0934772a4-Abstract.html) ·
> [arXiv](https://arxiv.org/abs/1408.5823)
