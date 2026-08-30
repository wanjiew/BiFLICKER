#!/usr/bin/env Rscript

# Country-level comparison of user-community labels.
# Produces grouped ARI bars for:
#   A. Centralized vs Local
#   B. BiFLICKER vs Local
#   C. Centralized vs BiFLICKER

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1L) stop("Cannot resolve script path")
script_path <- normalizePath(sub("^--file=", "", script_arg))
final_dir <- normalizePath(file.path(dirname(script_path), ".."))

results_dir <- file.path(final_dir, "results")
output_dir <- file.path(results_dir, "comparisons")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

manifest <- read.csv(
  file.path(final_dir, "preparation", "data", "country_splits",
            "country_split_manifest.csv"),
  stringsAsFactors = FALSE
)
central <- read.csv(
  file.path(results_dir, "central", "user_communities.csv"),
  stringsAsFactors = FALSE
)
biflicker <- read.csv(
  file.path(results_dir, "biflicker", "user_communities.csv"),
  stringsAsFactors = FALSE
)

choose2 <- function(value) value * (value - 1) / 2
adjusted_rand <- function(left, right) {
  tab <- table(left, right)
  same_pairs <- sum(choose2(tab))
  left_pairs <- sum(choose2(rowSums(tab)))
  right_pairs <- sum(choose2(colSums(tab)))
  total_pairs <- choose2(sum(tab))
  expected <- left_pairs * right_pairs / total_pairs
  maximum <- (left_pairs + right_pairs) / 2
  if (maximum == expected) return(1)
  (same_pairs - expected) / (maximum - expected)
}

comparison <- do.call(rbind, lapply(seq_len(nrow(manifest)), function(index) {
  server <- manifest$server_id[index]
  local_dir <- file.path(results_dir, "local", manifest$result_directory[index])
  local <- read.csv(
    file.path(local_dir, "user_communities.csv"), stringsAsFactors = FALSE
  )
  joined <- merge(
    local[, c("user_index", "community")],
    central[, c("user_index", "community")],
    by = "user_index", suffixes = c("_local", "_central")
  )
  joined <- merge(
    joined,
    biflicker[, c("user_index", "community")],
    by = "user_index"
  )
  names(joined)[names(joined) == "community"] <- "community_biflicker"
  if (nrow(joined) != manifest$users[index]) {
    stop("User-count mismatch for server ", server)
  }
  data.frame(
    server_id = server,
    country = manifest$country[index],
    users = manifest$users[index],
    ari_central_vs_local = adjusted_rand(
      joined$community_central, joined$community_local
    ),
    ari_biflicker_vs_local = adjusted_rand(
      joined$community_biflicker, joined$community_local
    ),
    ari_central_vs_biflicker = adjusted_rand(
      joined$community_central, joined$community_biflicker
    )
  )
}))
comparison <- comparison[order(comparison$server_id), ]
write.csv(
  comparison,
  file.path(output_dir, "country_label_ari.csv"),
  row.names = FALSE
)

values <- rbind(
  comparison$ari_central_vs_local,
  comparison$ari_biflicker_vs_local,
  comparison$ari_central_vs_biflicker
)
labels <- sprintf("%s (%d)", comparison$country, comparison$users)
colors <- c("#9ECAE1", "#3F7CAC", "#4A4A4A")
bar_density <- c(NA, NA, 18)
bar_angle <- c(45, 45, 45)
legend_labels <- c(
  "Centralized vs Local",
  "BiFLICKER vs Local",
  "Centralized vs BiFLICKER"
)
y_min <- min(0, floor(10 * (min(values) - 0.05)) / 10)

draw_figure <- function() {
  old <- par(no.readonly = TRUE)
  on.exit(par(old))
  par(mar = c(7.2, 4.5, 3.0, 1), mgp = c(2.6, 0.8, 0), las = 1)
  positions <- barplot(
    values,
    beside = TRUE,
    col = colors,
    border = c("#6BAED6", "#24557A", "#111111"),
    density = bar_density,
    angle = bar_angle,
    names.arg = labels,
    cex.names = 0.78,
    ylim = c(y_min, 1.23),
    yaxp = c(0, 1, 5),
    ylab = "Adjusted Rand index (ARI)",
    xlab = "Country (number of users)",
    main = "Agreement of user-community assignments by country"
  )
  abline(h = seq(0, 1, by = 0.2), col = "#E6E6E6", lwd = 0.8)
  abline(h = 0, col = "#777777", lwd = 0.9)
  barplot(
    values,
    beside = TRUE,
    col = colors,
    border = c("#6BAED6", "#24557A", "#111111"),
    density = bar_density,
    angle = bar_angle,
    names.arg = labels,
    cex.names = 0.78,
    ylim = c(y_min, 1.23),
    yaxp = c(0, 1, 5),
    ylab = "Adjusted Rand index (ARI)",
    xlab = "Country (number of users)",
    main = "Agreement of user-community assignments by country",
    add = TRUE, axes = FALSE
  )
  label_y <- ifelse(values >= 0, values + 0.025, values - 0.025)
  text(
    positions, label_y,
    labels = sprintf("%.2f", values),
    cex = 0.67,
    pos = ifelse(values >= 0, 3, 1),
    xpd = TRUE
  )
  legend(
    "top", legend = legend_labels, fill = colors,
    border = c("#6BAED6", "#24557A", "#111111"),
    density = bar_density, angle = bar_angle,
    bty = "n", horiz = TRUE, cex = 0.82, inset = c(0, 0.015)
  )
}

pdf(
  file.path(output_dir, "country_label_ari.pdf"),
  width = 11, height = 6.5, useDingbats = FALSE
)
draw_figure()
dev.off()

png(
  file.path(output_dir, "country_label_ari.png"),
  width = 2200, height = 1300, res = 200
)
draw_figure()
dev.off()

print(comparison)
