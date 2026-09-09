#!/usr/bin/env Rscript

# Final interpretation analysis for the L = 7 Sequential BiFLICKER user communities.
#
# Part I: demographic and country-composition summary table
#
# Part II: listening-behavior and binary-profile-similarity figure
#   A. listening breadth (distinct retained artists per user),
#   B. activity (total retained listening events per user), and
#   C. average within-community binary cosine similarity.
# Breadth and activity use log10 axes with original-value tick labels.
# Similarity is not rescaled by community size. Its uncertainty is a
# leave-one-user-out jackknife 95% confidence interval.
#
# Part III: representative artist lists and community-level genre
# interpretation. Genres summarize each community's representative set; they
# are not assigned separately to every artist. The full interpretation remains
# in this single reproducible analysis file.

suppressPackageStartupMessages(library(Matrix))

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Cannot resolve script path")
script_path <- normalizePath(sub("^--file=", "", script_arg))
final_dir <- normalizePath(file.path(dirname(script_path), ".."))

edge_path <- file.path(
  final_dir, "preparation", "data", "user_artist_edges.csv.gz"
)
community_path <- file.path(
  final_dir, "results", "biflicker", "user_communities.csv"
)
artist_path <- file.path(final_dir, "preparation", "data", "artists.csv")
user_path <- file.path(final_dir, "preparation", "data", "users.csv")
output_dir <- file.path(
  final_dir, "results", "interpretation"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

edges <- read.csv(gzfile(edge_path), stringsAsFactors = FALSE)
communities <- read.csv(community_path, stringsAsFactors = FALSE)
artists <- read.csv(artist_path, stringsAsFactors = FALSE)
users <- read.csv(user_path, stringsAsFactors = FALSE)
communities <- communities[order(communities$user_index), ]
community_ids <- sort(unique(communities$community))

if (anyDuplicated(communities$user_index)) stop("Duplicate user_index")
if (anyNA(edges$event_count) || any(edges$event_count < 0)) {
  stop("event_count must be nonnegative and nonmissing")
}

# Part I: one-row-per-community summary to appear before the three-panel
# listening-behavior figure. Female and Male are reported as count (percentage
# of all users in the community), so users with unreported gender remain in the
# denominator. Median age uses available ages. Main countries are the three
# largest country groups, with count and community percentage.
user_demographics <- merge(
  communities[, c("user_index", "community", "country")],
  users[, c("user_index", "gender", "age")],
  by = "user_index",
  all.x = TRUE,
  sort = FALSE
)
if (nrow(user_demographics) != nrow(communities)) {
  stop("Incomplete or duplicated user demographics")
}

format_count_percentage <- function(count, denominator) {
  sprintf("%d (%.1f%%)", count, 100 * count / denominator)
}

format_main_countries <- function(country_values, top_n = 3L) {
  counts <- sort(table(country_values), decreasing = TRUE)
  # Make tied country counts deterministic without changing the count order.
  country_table <- data.frame(
    country = names(counts), count = as.integer(counts),
    stringsAsFactors = FALSE
  )
  country_table <- country_table[order(
    -country_table$count, country_table$country
  ), ]
  country_table <- head(country_table, top_n)
  denominator <- length(country_values)
  paste(sprintf(
    "%s: %d (%.1f%%)",
    country_table$country,
    country_table$count,
    100 * country_table$count / denominator
  ), collapse = "; ")
}

demographic_rows <- lapply(community_ids, function(community_id) {
  rows <- user_demographics[user_demographics$community == community_id, ]
  n_c <- nrow(rows)
  gender <- tolower(trimws(rows$gender))
  observed_age <- rows$age[!is.na(rows$age)]
  data.frame(
    Community = paste0("C", community_id),
    n_c = n_c,
    Female = format_count_percentage(sum(gender == "f"), n_c),
    Male = format_count_percentage(sum(gender == "m"), n_c),
    `Median age` = if (length(observed_age) > 0) median(observed_age) else NA_real_,
    `Main countries` = format_main_countries(rows$country),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
})
demographic_summary <- do.call(rbind, demographic_rows)
write.csv(
  demographic_summary,
  file.path(output_dir, "biflicker_community_demographic_summary.csv"),
  row.names = FALSE,
  na = ""
)

# Part II: user-level behavior.
breadth <- aggregate(artist_index ~ user_index, edges, function(x) length(unique(x)))
names(breadth)[2] <- "listening_breadth"
activity <- aggregate(event_count ~ user_index, edges, sum)
names(activity)[2] <- "activity"
user_metrics <- merge(communities, breadth, by = "user_index")
user_metrics <- merge(user_metrics, activity, by = "user_index")
if (nrow(user_metrics) != nrow(communities)) stop("Incomplete user metrics")

# Sparse binary artist-incidence matrix, with each row divided by the square
# root of its binary degree. Row inner products are binary cosine similarities.
user_ids <- communities$user_index
artist_ids <- sort(unique(edges$artist_index))
binary_network <- sparseMatrix(
  i = match(edges$user_index, user_ids),
  j = match(edges$artist_index, artist_ids),
  x = edges$binary_weight,
  dims = c(length(user_ids), length(artist_ids))
)
user_degrees <- as.numeric(rowSums(binary_network != 0))
if (any(user_degrees <= 0)) stop("Every user must have positive binary degree")
profiles <- Diagonal(x = 1 / sqrt(user_degrees)) %*% binary_network

estimate_similarity <- function(community_id) {
  rows <- which(communities$community == community_id)
  profile_c <- profiles[rows, , drop = FALSE]
  n_c <- length(rows)
  if (n_c < 3) stop("At least three users are required for jackknife uncertainty")

  profile_sum <- as.numeric(colSums(profile_c))
  squared_norms <- as.numeric(rowSums(profile_c ^ 2))
  pairwise_numerator <- sum(profile_sum ^ 2) - sum(squared_norms)
  estimate <- pairwise_numerator / (n_c * (n_c - 1))

  overlap_with_sum <- as.numeric(profile_c %*% profile_sum)
  leave_one_estimates <- (
    pairwise_numerator - 2 * overlap_with_sum + 2 * squared_norms
  ) / ((n_c - 1) * (n_c - 2))
  leave_one_mean <- mean(leave_one_estimates)
  standard_error <- sqrt(
    (n_c - 1) / n_c * sum((leave_one_estimates - leave_one_mean) ^ 2)
  )

  data.frame(
    community = community_id,
    community_size = n_c,
    average_binary_cosine_similarity = estimate,
    jackknife_se = standard_error,
    ci_lower_95 = max(0, estimate - qnorm(0.975) * standard_error),
    ci_upper_95 = min(1, estimate + qnorm(0.975) * standard_error)
  )
}

similarity <- do.call(rbind, lapply(community_ids, estimate_similarity))

summarize_user_metric <- function(values, community_id, prefix) {
  x <- values[user_metrics$community == community_id]
  q <- quantile(x, c(0.25, 0.5, 0.75), names = FALSE)
  answer <- c(mean(x), q)
  names(answer) <- paste0(prefix, c("_mean", "_q25", "_median", "_q75"))
  answer
}

summary_rows <- lapply(community_ids, function(community_id) {
  c(
    community = community_id,
    community_size = sum(user_metrics$community == community_id),
    summarize_user_metric(user_metrics$listening_breadth, community_id, "breadth"),
    summarize_user_metric(user_metrics$activity, community_id, "activity")
  )
})
summary_table <- as.data.frame(do.call(rbind, summary_rows))
summary_table <- merge(summary_table, similarity, by = c("community", "community_size"))
summary_table <- summary_table[order(summary_table$community), ]
write.csv(
  summary_table,
  file.path(output_dir, "biflicker_three_panel_interpretation_summary.csv"),
  row.names = FALSE
)

community_colors <- c(
  "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
  "#66A61E", "#E6AB02", "#A6761D"
)[seq_along(community_ids)]
community_labels <- paste0("C", community_ids)
community_sizes <- vapply(
  community_ids,
  function(x) sum(communities$community == x),
  integer(1)
)

format_tick <- function(value) {
  format(value, big.mark = ",", scientific = FALSE, trim = TRUE)
}

draw_box_panel <- function(
  values, ticks, y_label, panel_title, annotate_sizes = FALSE,
  title_adjustment = 0.5, tick_labels = NULL,
  colors = community_colors
) {
  groups <- lapply(community_ids, function(community_id) {
    log10(values[user_metrics$community == community_id])
  })
  plot_limits <- range(unlist(groups), finite = TRUE)
  if (annotate_sizes) plot_limits[2] <- plot_limits[2] + 0.48
  box_result <- boxplot(
    groups,
    names = community_labels,
    col = adjustcolor(colors, alpha.f = 0.68),
    border = colors,
    medcol = "#202020",
    medlwd = 1.8,
    whisklty = 1,
    staplewex = 0.55,
    lwd = 1.15,
    outline = TRUE,
    outpch = 21,
    outcex = 0.42,
    outbg = "grey72",
    outcol = "grey55",
    xaxt = "n", yaxt = "n",
    xlab = "", ylab = y_label,
    ylim = plot_limits
  )
  if (is.null(tick_labels)) tick_labels <- format_tick(ticks)
  axis(2, at = log10(ticks), labels = tick_labels, cex.axis = 1)
  axis(
    1, at = seq_along(community_ids), labels = community_labels,
    tick = FALSE, line = -0.10, cex.axis = 1,
    gap.axis = -1, las = 2
  )
  if (annotate_sizes) {
    text(
      seq_along(community_ids), box_result$stats[5, ] + 0.035,
      labels = sprintf("n=%d", community_sizes),
      cex = 1, srt = 90, adj = c(0, 0.5), xpd = FALSE
    )
  }
  mtext(
    panel_title, side = 3, line = 0.75,
    font = 2, adj = title_adjustment
  )
}

draw_similarity_panel <- function(colors = community_colors) {
  upper_limit <- max(similarity$ci_upper_95) * 1.12
  positions <- barplot(
    similarity$average_binary_cosine_similarity,
    names.arg = community_labels,
    col = adjustcolor(colors, alpha.f = 0.68),
    border = colors,
    lwd = 1.15,
    ylim = c(0, upper_limit),
    xaxt = "n",
    xlab = "",
    ylab = "Pairwise similarity between users",
    space = 0.28
  )
  axis(
    1, at = positions, labels = community_labels,
    tick = FALSE, line = -0.10, cex.axis = 1,
    gap.axis = -1, las = 2
  )
  arrows(
    positions, similarity$ci_lower_95,
    positions, similarity$ci_upper_95,
    angle = 90, code = 3, length = 0.038,
    lwd = 1.2, col = "#333333"
  )
  mtext(
    "Within-community similarity",
    side = 3, line = 0.75, font = 2, adj = 1
  )
}

draw_figure <- function() {
  old <- par(no.readonly = TRUE)
  on.exit(par(old))
  par(
    mfrow = c(1, 3),
    mar = c(3.15, 4.4, 2.0, 0.3),
    oma = c(1.55, 0, 0, 0),
    mgp = c(2.45, 0.55, 0),
    tcl = -0.25, las = 1
  )
  # par(mfrow = c(1, 3)) otherwise applies an automatic text shrink.
  par(cex = 1)
  draw_box_panel(
    user_metrics$listening_breadth,
    ticks = c(50, 100, 500, 1000, 4000),
    tick_labels = c("50", "100", "500", "1K", "4K"),
    y_label = "Number of artists",
    panel_title = "Listening breadth",
    annotate_sizes = TRUE
  )
  draw_box_panel(
    user_metrics$activity,
    ticks = c(100, 1000, 10000, 100000),
    tick_labels = c("100", "1k", "10k", "100k"),
    y_label = "Number of listening activities",
    panel_title = "Activity",
    annotate_sizes = FALSE
  )
  draw_similarity_panel()
  mtext(
    "BiFLICKER community",
    side = 1, outer = TRUE, line = -0.45
  )
}

manuscript_text_width_in <- 345 / 72.27 + 1
figure_width_in <- 1.10 * manuscript_text_width_in
figure_height_in <- 3.05
figure_pointsize <- 10.5

pdf(
  file.path(output_dir, "biflicker_three_panel_interpretation.pdf"),
  width = figure_width_in,
  height = figure_height_in,
  pointsize = figure_pointsize,
  useDingbats = FALSE
)
draw_figure()
dev.off()

png(
  file.path(output_dir, "biflicker_three_panel_interpretation.png"),
  width = figure_width_in,
  height = figure_height_in,
  units = "in", res = 300,
  pointsize = figure_pointsize
)
draw_figure()
dev.off()

# -----------------------------------------------------------------------------
# Part II. Representative artists for each BiFLICKER user community
# -----------------------------------------------------------------------------

# An artist must be sufficiently common within a community and more prevalent
# inside than outside it. Eligible artists are ranked first by the prevalence
# difference (within minus outside) and then by a Jeffreys-smoothed log-odds
# ratio. This balances representativeness and distinctiveness without allowing
# artists heard by only one or two users to dominate a small community.
representative_min_users <- 5L
representative_min_within_prevalence <- 0.10
representative_top_n <- 5L
coverage_top_n <- 5L

artist_metadata <- artists[match(artist_ids, artists$artist_index), ]
if (anyNA(artist_metadata$artist_index)) stop("Artist metadata are incomplete")
if (anyDuplicated(edges[, c("user_index", "artist_index")])) {
  stop("Duplicate user-artist edges are not expected")
}

event_network <- sparseMatrix(
  i = match(edges$user_index, user_ids),
  j = match(edges$artist_index, artist_ids),
  x = edges$event_count,
  dims = c(length(user_ids), length(artist_ids))
)
total_users <- nrow(communities)
global_artist_users <- as.numeric(colSums(binary_network != 0))
global_artist_events <- as.numeric(colSums(event_network))

artist_statistics <- vector("list", length(community_ids))
representative_lists <- vector("list", length(community_ids))

for (index in seq_along(community_ids)) {
  community_id <- community_ids[index]
  inside_rows <- which(communities$community == community_id)
  outside_rows <- which(communities$community != community_id)
  n_inside <- length(inside_rows)
  n_outside <- length(outside_rows)

  inside_users <- as.numeric(colSums(binary_network[inside_rows, , drop = FALSE] != 0))
  outside_users <- as.numeric(colSums(binary_network[outside_rows, , drop = FALSE] != 0))
  inside_events <- as.numeric(colSums(event_network[inside_rows, , drop = FALSE]))

  within_prevalence <- inside_users / n_inside
  outside_prevalence <- outside_users / n_outside
  global_prevalence <- global_artist_users / total_users
  prevalence_difference <- within_prevalence - outside_prevalence
  smoothed_log_odds_ratio <-
    log((inside_users + 0.5) / (n_inside - inside_users + 0.5)) -
    log((outside_users + 0.5) / (n_outside - outside_users + 0.5))
  smoothed_prevalence_ratio <-
    ((inside_users + 0.5) / (n_inside + 1)) /
    ((outside_users + 0.5) / (n_outside + 1))

  minimum_users_for_community <- max(
    representative_min_users,
    ceiling(representative_min_within_prevalence * n_inside)
  )
  eligible <-
    inside_users >= minimum_users_for_community &
    within_prevalence >= representative_min_within_prevalence &
    prevalence_difference > 0
  ordered <- which(eligible)
  ordered <- ordered[order(
    -prevalence_difference[ordered],
    -smoothed_log_odds_ratio[ordered],
    -inside_users[ordered],
    artist_metadata$artist_name[ordered]
  )]
  distinctive_rank <- rep(NA_integer_, length(artist_ids))
  distinctive_rank[ordered] <- seq_along(ordered)

  if (length(ordered) >= representative_top_n) {
    selected <- head(ordered, representative_top_n)
    selection_basis_value <- "distinctive_prevalence"
  } else {
    # If a community has no sufficiently prevalent enriched artist, use its
    # most prevalent artists as a transparent fallback. This avoids lowering
    # the threshold and selecting artists heard by only a few users.
    fallback <- which(
      inside_users >= minimum_users_for_community &
        within_prevalence >= representative_min_within_prevalence
    )
    fallback <- fallback[order(
      -within_prevalence[fallback],
      -inside_users[fallback],
      -prevalence_difference[fallback],
      artist_metadata$artist_name[fallback]
    )]
    selected <- head(fallback, representative_top_n)
    selection_basis_value <- "most_prevalent_fallback"
  }
  representative_rank <- rep(NA_integer_, length(artist_ids))
  representative_rank[selected] <- seq_along(selected)
  representative_selection_basis <- rep(NA_character_, length(artist_ids))
  representative_selection_basis[selected] <- selection_basis_value

  statistics_c <- data.frame(
    community = community_id,
    community_name = paste0("C", community_id),
    community_size = n_inside,
    outside_community_size = n_outside,
    artist_index = artist_ids,
    mbid = artist_metadata$mbid,
    artist_name = artist_metadata$artist_name,
    musicbrainz_url = paste0(
      "https://musicbrainz.org/artist/", artist_metadata$mbid
    ),
    users_in_community = inside_users,
    within_community_prevalence = within_prevalence,
    users_outside_community = outside_users,
    outside_community_prevalence = outside_prevalence,
    global_user_degree = global_artist_users,
    global_prevalence = global_prevalence,
    prevalence_difference = prevalence_difference,
    smoothed_prevalence_ratio = smoothed_prevalence_ratio,
    smoothed_log_odds_ratio = smoothed_log_odds_ratio,
    community_event_count = inside_events,
    community_event_share = inside_events / sum(inside_events),
    share_of_artist_users_from_community = ifelse(
      global_artist_users > 0, inside_users / global_artist_users, NA_real_
    ),
    share_of_artist_events_from_community = ifelse(
      global_artist_events > 0, inside_events / global_artist_events, NA_real_
    ),
    eligible_as_distinctive = eligible,
    distinctive_rank = distinctive_rank,
    representative_rank = representative_rank,
    representative_selection_basis = representative_selection_basis,
    stringsAsFactors = FALSE
  )
  artist_statistics[[index]] <- statistics_c
  representative_lists[[index]] <- statistics_c[
    !is.na(statistics_c$representative_rank) &
      statistics_c$representative_rank <= representative_top_n,
  ]
}

artist_statistics <- do.call(rbind, artist_statistics)
representative_artists <- do.call(rbind, representative_lists)
representative_artists <- representative_artists[order(
  representative_artists$community,
  representative_artists$representative_rank
), ]

write.csv(
  representative_artists,
  file.path(output_dir, "biflicker_representative_artists_by_community.csv"),
  row.names = FALSE
)

statistics_connection <- gzfile(
  file.path(output_dir, "biflicker_artist_community_statistics.csv.gz"),
  open = "wt"
)
write.csv(artist_statistics, statistics_connection, row.names = FALSE)
close(statistics_connection)

representative_coverage <- do.call(rbind, lapply(community_ids, function(community_id) {
  top_rows <- representative_artists[
    representative_artists$community == community_id &
      representative_artists$representative_rank <= coverage_top_n,
  ]
  inside_rows <- which(communities$community == community_id)
  top_columns <- match(top_rows$artist_index, artist_ids)
  artists_per_user <- as.numeric(rowSums(
    binary_network[inside_rows, top_columns, drop = FALSE] != 0
  ))
  data.frame(
    community = community_id,
    community_size = length(inside_rows),
    representative_artists_used = nrow(top_rows),
    representative_artist_names = paste(top_rows$artist_name, collapse = "; "),
    musicbrainz_links = paste(top_rows$musicbrainz_url, collapse = "; "),
    users_hearing_at_least_one = sum(artists_per_user > 0),
    fraction_hearing_at_least_one = mean(artists_per_user > 0),
    median_representative_artists_per_user = median(artists_per_user),
    mean_representative_artists_per_user = mean(artists_per_user),
    stringsAsFactors = FALSE
  )
}))
write.csv(
  representative_coverage,
  file.path(output_dir, "biflicker_representative_artist_coverage.csv"),
  row.names = FALSE
)

# Article-facing genre labels are community-level syntheses of the five
# representative artists. They deliberately do not assign a genre to each
# individual artist. The linked MusicBrainz artist records provide transparent
# citation points for the synthesis; C1 is described as mixed because it has no
# high-coverage artist with positive community enrichment.
community_genres <- data.frame(
  community = community_ids,
  community_genre = c(
    "Mixed indie/alternative and mainstream rock",
    "Indie rock and post-punk revival",
    "Heavy metal, groove metal, and alternative metal",
    "Hip hop and contemporary R&B",
    "Classic pop rock, soul, and new wave",
    "Downtempo, trip hop, and nu jazz/electronica",
    "Roots-oriented country, blues, folk, and psychedelic rock"
  ),
  community_genre_interpretation = c(
    "A heterogeneous general-audience community without a single distinctive genre core.",
    "A concentrated UK-oriented indie and post-punk-revival profile.",
    "A metal-oriented profile spanning heavier, groove, and alternative styles.",
    "A hip-hop and R&B profile with some crossover material.",
    "An older pop-rock and soul profile with a substantial new-wave component.",
    "An electronic listening profile centered on downtempo, trip hop, and jazz-influenced production.",
    "An eclectic roots profile connecting country, blues, folk, and psychedelic rock."
  ),
  stringsAsFactors = FALSE
)

community_article_table <- merge(
  representative_coverage,
  community_genres,
  by = "community",
  sort = FALSE
)
community_article_table <- community_article_table[
  match(community_ids, community_article_table$community),
]
community_article_table$top5_user_coverage <- sprintf(
  "%d/%d (%.1f%%)",
  community_article_table$users_hearing_at_least_one,
  community_article_table$community_size,
  100 * community_article_table$fraction_hearing_at_least_one
)
community_article_table$representative_artist_citation_links <-
  community_article_table$musicbrainz_links
community_article_table$genre_classification_note <-
  "Community-level synthesis from the representative-artist set; not a per-artist genre assignment."
community_article_table <- community_article_table[, c(
  "community", "community_size", "representative_artist_names",
  "representative_artist_citation_links", "community_genre",
  "community_genre_interpretation",
  "top5_user_coverage", "median_representative_artists_per_user",
  "genre_classification_note"
)]
write.csv(
  community_article_table,
  file.path(output_dir, "biflicker_community_artist_genre_table.csv"),
  row.names = FALSE
)

cat("\nPart I: community demographic summary\n")
print(demographic_summary, row.names = FALSE)
cat("\nPart II: user-community interpretation summary\n")
print(summary_table, digits = 8)
cat("\nPart III: top representative artists\n")
print(
  representative_artists[, c(
    "community", "representative_rank", "representative_selection_basis",
    "artist_name",
    "users_in_community", "within_community_prevalence",
    "outside_community_prevalence", "prevalence_difference"
  )],
  row.names = FALSE,
  digits = 5
)
cat(sprintf("\nRepresentative-artist coverage (top %d)\n", coverage_top_n))
print(representative_coverage[, c(
  "community", "community_size", "fraction_hearing_at_least_one",
  "median_representative_artists_per_user",
  "mean_representative_artists_per_user"
)], row.names = FALSE, digits = 5)
cat("\nCommunity-level genre interpretation\n")
print(
  community_article_table[, c(
    "community", "representative_artist_names", "community_genre",
    "top5_user_coverage"
  )],
  row.names = FALSE
)
