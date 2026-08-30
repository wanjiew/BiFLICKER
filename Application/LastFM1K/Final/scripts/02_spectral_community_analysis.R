#!/usr/bin/env Rscript

# Final spectral/community analysis for the cleaned Last.fm 1K network.
#
# MAIN PARAMETERS (edit here, or pass key=value arguments on the command line):
#   rank = 7                       number of spectral components
#   communities = 7                number of user communities
#   kmeans_nstart = 1000            random starts for user clustering
#   biflicker_iterations = 20000    maximum iterations for each component
#   biflicker_log_interval = 100    convergence checkpoint interval
#   biflicker_min_iterations = 500  minimum iterations before early stopping
#   residual_tolerance = 0.05       maximum left/right relative residual
#   estimate_change_tolerance=1e-4  maximum relative Rayleigh-value change
#   convergence_patience = 3        consecutive successful checkpoints
#   upper_multiplier = 2            Sequential BiFLICKER upper spectral bound multiplier
#   step_safety = 1                 multiplier in the fixed step size
#   seed = 42                       reproducibility seed
#
# Outputs:
#   preparation/data/country_splits/  country-specific binary matrices/maps
#   results/local/                     local SVD and user communities
#   results/central/                   centralized SVD and user communities
#   results/biflicker/                 Sequential BiFLICKER outputs

suppressPackageStartupMessages(library(Matrix))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Cannot resolve script path")
script_path <- normalizePath(sub("^--file=", "", script_arg))
final_dir <- normalizePath(file.path(dirname(script_path), ".."))

settings <- list(
  data_dir = file.path(final_dir, "preparation", "data"),
  results_dir = file.path(final_dir, "results"),
  rank = 7L,
  communities = 7L,
  kmeans_nstart = 1000L,
  biflicker_iterations = 20000L,
  biflicker_log_interval = 100L,
  biflicker_min_iterations = 500L,
  residual_tolerance = 0.05,
  estimate_change_tolerance = 1e-4,
  convergence_patience = 3L,
  upper_multiplier = 2,
  step_safety = 1,
  seed = 42L
)

coerce_setting <- function(old, value) {
  if (is.integer(old)) return(as.integer(value))
  if (is.numeric(old)) return(as.numeric(value))
  value
}
for (argument in commandArgs(trailingOnly = TRUE)) {
  pieces <- strsplit(argument, "=", fixed = TRUE)[[1]]
  if (length(pieces) != 2L || !pieces[1] %in% names(settings)) {
    stop("Unknown key=value setting: ", argument)
  }
  settings[[pieces[1]]] <- coerce_setting(settings[[pieces[1]]], pieces[2])
}
if (settings$rank < 1L || settings$communities < 2L) stop("Invalid rank/community count")

split_dir <- file.path(settings$data_dir, "country_splits")
local_dir <- file.path(settings$results_dir, "local")
central_dir <- file.path(settings$results_dir, "central")
biflicker_dir <- file.path(settings$results_dir, "biflicker")
for (directory in c(split_dir, local_dir, central_dir, biflicker_dir)) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}

message("Reading final prepared network...")
users <- read.csv(file.path(settings$data_dir, "users.csv"), stringsAsFactors = FALSE)
artists <- read.csv(file.path(settings$data_dir, "artists.csv"), stringsAsFactors = FALSE)
edges <- read.csv(gzfile(file.path(settings$data_dir, "user_artist_edges.csv.gz")))
n <- nrow(users)
m <- nrow(artists)
B <- sparseMatrix(
  i = edges$user_index,
  j = edges$artist_index,
  x = edges$binary_weight,
  dims = c(n, m),
  giveCsparse = TRUE
)
B@x[] <- 1
B <- drop0(B)
if (nrow(edges) != length(B@x)) stop("Duplicate or invalid edge rows detected")

server_ids <- sort(unique(users$server_id))
server_rows <- lapply(server_ids, function(server) which(users$server_id == server))
names(server_rows) <- as.character(server_ids)
blocks <- lapply(server_rows, function(rows) B[rows, , drop = FALSE])
block_sizes <- vapply(blocks, nrow, integer(1))
if (any(block_sizes < settings$communities)) {
  stop("At least one country has fewer users than requested communities")
}

slugify <- function(value) {
  value <- tolower(gsub("[^A-Za-z0-9]+", "_", value))
  gsub("(^_+|_+$)", "", value)
}

write_mtx_gz <- function(value, path, comment) {
  triplet <- summary(value)
  connection <- gzfile(path, "wt")
  on.exit(close(connection))
  writeLines("%%MatrixMarket matrix coordinate integer general", connection)
  writeLines(paste0("% ", comment), connection)
  writeLines(sprintf("%d %d %d", nrow(value), ncol(value), nrow(triplet)), connection)
  write.table(
    data.frame(i = triplet$i, j = triplet$j, x = as.integer(triplet$x)),
    connection, row.names = FALSE, col.names = FALSE, quote = FALSE
  )
}

write_matrix_csv_gz <- function(metadata, matrix_value, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection))
  write.csv(cbind(metadata, as.data.frame(matrix_value)), connection, row.names = FALSE)
}

relabel_by_size <- function(labels) {
  ordering <- as.integer(names(sort(table(labels), decreasing = TRUE)))
  mapping <- integer(max(labels))
  mapping[ordering] <- seq_along(ordering)
  mapping[labels]
}

cluster_users <- function(embedding, seed) {
  set.seed(seed)
  result <- kmeans(
    embedding[, seq_len(settings$rank), drop = FALSE],
    centers = settings$communities,
    nstart = settings$kmeans_nstart,
    iter.max = 500
  )
  relabel_by_size(result$cluster)
}

top_svd <- function(value, rank) {
  gram <- as.matrix(tcrossprod(value))
  decomposition <- eigen(gram, symmetric = TRUE)
  eigenvalues <- pmax(decomposition$values[seq_len(rank)], 0)
  singular_values <- sqrt(eigenvalues)
  left <- decomposition$vectors[, seq_len(rank), drop = FALSE]
  right <- sweep(as.matrix(crossprod(value, left)), 2, singular_values, "/")
  list(
    eigenvalues = eigenvalues,
    singular_values = singular_values,
    left = left,
    right = right
  )
}

write_svd_outputs <- function(
    output_dir, spectral, left_metadata, right_metadata,
    communities, extra_values = NULL) {
  values <- data.frame(
    component = seq_len(settings$rank),
    eigenvalue = spectral$eigenvalues,
    singular_value = spectral$singular_values
  )
  if (!is.null(extra_values)) values <- cbind(values, extra_values)
  write.csv(values, file.path(output_dir, "eigenvalues.csv"), row.names = FALSE)
  colnames(spectral$left) <- paste0("component_", seq_len(settings$rank))
  colnames(spectral$right) <- paste0("component_", seq_len(settings$rank))
  write_matrix_csv_gz(
    left_metadata, spectral$left,
    file.path(output_dir, "left_eigenvectors_users.csv.gz")
  )
  write_matrix_csv_gz(
    right_metadata, spectral$right,
    file.path(output_dir, "right_eigenvectors_artists.csv.gz")
  )
  write.csv(
    cbind(left_metadata, community = communities),
    file.path(output_dir, "user_communities.csv"), row.names = FALSE
  )
}

# ---------------------------------------------------------------------------
# Country splits and local analyses
# ---------------------------------------------------------------------------
message("Writing country splits and computing local SVD/community results...")
split_membership <- vector("list", length(server_ids))
local_manifest <- vector("list", length(server_ids))
local_svd <- vector("list", length(server_ids))

for (index in seq_along(server_ids)) {
  server <- server_ids[index]
  rows <- server_rows[[index]]
  country <- unique(users$country[rows])
  if (length(country) != 1L) stop("Server maps to multiple countries")
  slug <- sprintf("server_%02d_%s", server, slugify(country))
  matrix_path <- file.path(split_dir, paste0(slug, ".mtx.gz"))
  write_mtx_gz(
    blocks[[index]], matrix_path,
    sprintf("Binary user-by-artist matrix for %s; columns use global artist_index", country)
  )
  split_membership[[index]] <- data.frame(
    server_id = server,
    country = country,
    local_user_index = seq_along(rows),
    user_index = users$user_index[rows],
    user_id = users$user_id[rows]
  )

  spectral <- top_svd(blocks[[index]], settings$rank)
  local_svd[[index]] <- spectral
  communities <- cluster_users(spectral$left, settings$seed + server)
  output_dir <- file.path(local_dir, slug)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  local_user_metadata <- data.frame(
    local_user_index = seq_along(rows),
    user_index = users$user_index[rows],
    user_id = users$user_id[rows],
    country = country
  )
  artist_metadata <- artists[, c("artist_index", "mbid", "artist_name")]
  write_svd_outputs(
    output_dir, spectral, local_user_metadata, artist_metadata, communities
  )
  local_manifest[[index]] <- data.frame(
    server_id = server,
    country = country,
    users = length(rows),
    artists = m,
    binary_edges = length(blocks[[index]]@x),
    matrix_file = basename(matrix_path),
    result_directory = slug
  )
  message(sprintf("  %s: %d users, %d edges", country, length(rows), length(blocks[[index]]@x)))
}
write.csv(
  do.call(rbind, split_membership),
  file.path(split_dir, "country_split_users.csv"), row.names = FALSE
)
write.csv(
  do.call(rbind, local_manifest),
  file.path(split_dir, "country_split_manifest.csv"), row.names = FALSE
)

# ---------------------------------------------------------------------------
# Centralized SVD and user communities
# ---------------------------------------------------------------------------
message("Computing centralized SVD and user communities...")
central_svd <- top_svd(B, settings$rank)
central_communities <- cluster_users(central_svd$left, settings$seed)
user_metadata <- users[, c("user_index", "user_id", "server_id", "country")]
artist_metadata <- artists[, c("artist_index", "mbid", "artist_name")]
write_svd_outputs(
  central_dir, central_svd, user_metadata, artist_metadata,
  central_communities
)

# ---------------------------------------------------------------------------
# Sequential BiFLICKER
# ---------------------------------------------------------------------------
message("Running Sequential BiFLICKER...")
reference_server <- which.max(block_sizes)
reference_svd <- local_svd[[reference_server]]
targets <- sqrt(n / block_sizes[reference_server]) * reference_svd$singular_values

normalize_left <- function(values) {
  norm2 <- Reduce(`+`, lapply(values, function(value) colSums(value * value)))
  norms <- sqrt(pmax(norm2, .Machine$double.eps))
  lapply(values, function(value) sweep(value, 2, norms, "/"))
}
normalize_right <- function(value) {
  sweep(value, 2, sqrt(pmax(colSums(value * value), .Machine$double.eps)), "/")
}
left_gram <- function(values) {
  shared <- matrix(0, nrow = m, ncol = ncol(values[[1]]))
  for (index in seq_along(blocks)) {
    shared <- shared + as.matrix(crossprod(blocks[[index]], values[[index]]))
  }
  lapply(blocks, function(block) as.matrix(block %*% shared))
}
right_gram <- function(value) {
  result <- matrix(0, nrow = m, ncol = ncol(value))
  for (block in blocks) {
    result <- result + as.matrix(crossprod(block, block %*% value))
  }
  result
}
project_fixed <- function(xi, psi, fixed_xi, fixed_psi) {
  if (ncol(fixed_psi) == 0L) {
    return(list(xi = normalize_left(xi), psi = normalize_right(psi)))
  }
  psi <- psi - fixed_psi %*% crossprod(fixed_psi, psi)
  coefficients <- Reduce(`+`, Map(crossprod, fixed_xi, xi))
  for (index in seq_along(xi)) {
    xi[[index]] <- xi[[index]] - fixed_xi[[index]] %*% coefficients
  }
  list(xi = normalize_left(xi), psi = normalize_right(psi))
}
diagnose <- function(xi, psi) {
  left_once <- left_gram(xi)
  left_rho <- Reduce(`+`, Map(function(x, ax) colSums(x * ax), xi, left_once))
  left_residual_sq <- Reduce(`+`, Map(function(x, ax) {
    value <- ax - sweep(x, 2, left_rho, "*")
    colSums(value * value)
  }, xi, left_once))
  left_residual <- sqrt(left_residual_sq) /
    pmax(abs(left_rho), .Machine$double.eps)
  right_once <- right_gram(psi)
  right_rho <- colSums(psi * right_once)
  right_residual <- sqrt(colSums(
    (right_once - sweep(psi, 2, right_rho, "*"))^2
  )) / pmax(abs(right_rho), .Machine$double.eps)
  cross_singular <- Reduce(`+`, Map(function(x, block) {
    colSums(x * as.matrix(block %*% psi))
  }, xi, blocks))
  list(
    cross_singular = cross_singular,
    left_rayleigh_singular = sqrt(pmax(left_rho, 0)),
    right_rayleigh_singular = sqrt(pmax(right_rho, 0)),
    left_residual = left_residual,
    right_residual = right_residual,
    max_residual = pmax(left_residual, right_residual)
  )
}

fixed_xi <- lapply(block_sizes, function(size) {
  matrix(numeric(0), nrow = size, ncol = 0L)
})
fixed_psi <- matrix(numeric(0), nrow = m, ncol = 0L)
trace <- vector("list", settings$rank)
step_sizes <- numeric(settings$rank)
stopping_status <- vector("list", settings$rank)

for (component in seq_len(settings$rank)) {
  message(sprintf(
    "  component %d/%d: target %.3f", component, settings$rank, targets[component]
  ))
  local_left <- reference_svd$left[, component, drop = FALSE]
  psi <- as.matrix(crossprod(blocks[[reference_server]], local_left))
  psi <- normalize_right(psi / reference_svd$singular_values[component])
  xi <- lapply(blocks, function(block) as.matrix(block %*% psi))
  projected <- project_fixed(xi, psi, fixed_xi, fixed_psi)
  xi <- projected$xi
  psi <- projected$psi
  eta <- settings$step_safety / max(
    targets[component]^4,
    ((settings$upper_multiplier * targets[1])^2 - targets[component]^2)^2
  )
  step_sizes[component] <- eta

  component_trace <- vector(
    "list",
    ceiling(settings$biflicker_iterations / settings$biflicker_log_interval) + 1L
  )
  trace_index <- 0L
  previous_left_estimate <- NA_real_
  previous_right_estimate <- NA_real_
  consecutive_converged <- 0L
  component_converged <- FALSE
  stopping_reason <- "maximum_iterations"
  for (iteration in seq_len(settings$biflicker_iterations)) {
    left_once <- left_gram(xi)
    # Revised FLICKER: update the current left singular-value estimate at
    # every iteration.  The fixed input target is still used in the gradient.
    left_rayleigh <- Reduce(`+`, Map(function(x, ax) {
      colSums(x * ax)
    }, xi, left_once))
    sigma_u <- sqrt(pmax(left_rayleigh, 0))
    left_twice <- left_gram(left_once)
    xi_new <- vector("list", length(xi))
    for (index in seq_along(xi)) {
      gradient <- xi[[index]] * targets[component]^4 -
        2 * left_once[[index]] * targets[component]^2 + left_twice[[index]]
      xi_new[[index]] <- xi[[index]] - eta * gradient
    }
    right_once <- right_gram(psi)
    # Revised FLICKER line: sigma_v = sqrt(max((B'B v)' v, 0)).
    # It is updated every iteration but does not replace the fixed target.
    right_rayleigh <- colSums(psi * right_once)
    sigma_v <- sqrt(pmax(right_rayleigh, 0))
    right_twice <- right_gram(right_once)
    gradient <- psi * targets[component]^4 -
      2 * right_once * targets[component]^2 + right_twice
    psi_new <- psi - eta * gradient
    projected <- project_fixed(xi_new, psi_new, fixed_xi, fixed_psi)
    xi <- projected$xi
    psi <- projected$psi

    if (iteration == 1L || iteration %% settings$biflicker_log_interval == 0L) {
      diagnostic <- diagnose(xi, psi)
      left_change <- if (is.finite(previous_left_estimate)) {
        abs(diagnostic$left_rayleigh_singular - previous_left_estimate) /
          max(abs(previous_left_estimate), .Machine$double.eps)
      } else NA_real_
      right_change <- if (is.finite(previous_right_estimate)) {
        abs(diagnostic$right_rayleigh_singular - previous_right_estimate) /
          max(abs(previous_right_estimate), .Machine$double.eps)
      } else NA_real_
      convergence_conditions_met <-
        iteration >= settings$biflicker_min_iterations &&
        diagnostic$max_residual <= settings$residual_tolerance &&
        is.finite(left_change) && is.finite(right_change) &&
        left_change <= settings$estimate_change_tolerance &&
        right_change <= settings$estimate_change_tolerance
      if (convergence_conditions_met) {
        consecutive_converged <- consecutive_converged + 1L
      } else {
        consecutive_converged <- 0L
      }
      trace_index <- trace_index + 1L
      component_trace[[trace_index]] <- data.frame(
        component = component,
        iteration = iteration,
        target_singular_value = targets[component],
        left_rayleigh_singular_value = diagnostic$left_rayleigh_singular,
        right_rayleigh_singular_value = diagnostic$right_rayleigh_singular,
        cross_singular_value = abs(diagnostic$cross_singular),
        left_relative_estimate_change = left_change,
        right_relative_estimate_change = right_change,
        left_relative_residual = diagnostic$left_residual,
        right_relative_residual = diagnostic$right_residual,
        max_relative_residual = diagnostic$max_residual,
        convergence_conditions_met = convergence_conditions_met,
        consecutive_converged_checkpoints = consecutive_converged
      )
      previous_left_estimate <- diagnostic$left_rayleigh_singular
      previous_right_estimate <- diagnostic$right_rayleigh_singular
      if (iteration %% 1000L == 0L || iteration == 1L) {
        message(sprintf(
          "    iteration %d: left %.3f, right %.3f, cross %.3f, residual %.3e",
          iteration, diagnostic$left_rayleigh_singular,
          diagnostic$right_rayleigh_singular, abs(diagnostic$cross_singular),
          diagnostic$max_residual
        ))
      }
      if (consecutive_converged >= settings$convergence_patience) {
        component_converged <- TRUE
        stopping_reason <- "convergence_criterion"
        message(sprintf(
          "    converged at iteration %d after %d consecutive checkpoints",
          iteration, consecutive_converged
        ))
        break
      }
    }
  }
  trace[[component]] <- do.call(rbind, component_trace[seq_len(trace_index)])
  stopping_status[[component]] <- data.frame(
    component = component,
    iterations_used = iteration,
    converged = component_converged,
    stopping_reason = stopping_reason
  )
  for (index in seq_along(fixed_xi)) {
    fixed_xi[[index]] <- cbind(fixed_xi[[index]], xi[[index]])
  }
  fixed_psi <- cbind(fixed_psi, psi)
}

trace <- do.call(rbind, trace)
write.csv(trace, file.path(biflicker_dir, "convergence_trace.csv"), row.names = FALSE)

Xi <- matrix(NA_real_, nrow = n, ncol = settings$rank)
for (index in seq_along(server_rows)) Xi[server_rows[[index]], ] <- fixed_xi[[index]]
Psi <- fixed_psi
final_trace <- do.call(rbind, lapply(split(trace, trace$component), function(value) {
  value[nrow(value), , drop = FALSE]
}))
stopping_status <- do.call(rbind, stopping_status)
biflicker_svd <- list(
  eigenvalues = final_trace$right_rayleigh_singular_value^2,
  singular_values = final_trace$right_rayleigh_singular_value,
  left = Xi,
  right = Psi
)
biflicker_communities <- cluster_users(Xi, settings$seed)
write_svd_outputs(
  biflicker_dir, biflicker_svd, user_metadata, artist_metadata,
  biflicker_communities,
  extra_values = data.frame(
    target_singular_value = targets,
    left_rayleigh_singular_value = final_trace$left_rayleigh_singular_value,
    right_rayleigh_singular_value = final_trace$right_rayleigh_singular_value,
    cross_singular_value = final_trace$cross_singular_value,
    step_size = step_sizes,
    iterations_used = stopping_status$iterations_used,
    converged = stopping_status$converged,
    stopping_reason = stopping_status$stopping_reason,
    left_relative_residual = final_trace$left_relative_residual,
    right_relative_residual = final_trace$right_relative_residual,
    max_relative_residual = final_trace$max_relative_residual
  )
)

settings_for_output <- settings
if (normalizePath(settings$data_dir, mustWork = FALSE) ==
    normalizePath(file.path(final_dir, "preparation", "data"), mustWork = FALSE)) {
  settings_for_output$data_dir <- "preparation/data"
}
if (normalizePath(settings$results_dir, mustWork = FALSE) ==
    normalizePath(file.path(final_dir, "results"), mustWork = FALSE)) {
  settings_for_output$results_dir <- "results"
}
write.csv(
  data.frame(
    setting = names(settings_for_output),
    value = vapply(settings_for_output, as.character, character(1))
  ),
  file.path(biflicker_dir, "run_settings.csv"), row.names = FALSE
)

message("Final spectral/community analysis complete: ", settings$results_dir)
