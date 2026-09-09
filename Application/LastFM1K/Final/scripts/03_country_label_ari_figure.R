#!/usr/bin/env Rscript

# Country-level comparison of user-community labels.
# Produces grouped ARI bars for:
#   A. Centralized vs Local
#   B. Sequential BiFLICKER vs Local
#   C. Centralized vs Sequential BiFLICKER

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
country_labels <- sprintf("%s (%d)", comparison$country, comparison$users)
country_labels[comparison$country == "United States"] <- sprintf(
  "US (%d)", comparison$users[comparison$country == "United States"]
)
country_labels[comparison$country == "United Kingdom"] <- sprintf(
  "UK (%d)", comparison$users[comparison$country == "United Kingdom"]
)
fills <- c("#9ECAE1", "#3F7CAC", "#4A4A4A")
borders <- c("#6BAED6", "#24557A", "#111111")
hatch_density <- c(NA, NA, 18)
hatch_angle <- c(45, 45, 45)
legend_labels <- c(
  "Centralized vs Local",
  "BiFLICKER vs Local",
  "Centralized vs BiFLICKER"
)
y_min <- min(0, floor(10 * (min(values) - 0.05)) / 10)
figure_height_in <- 3.20
figure_width_in <- 2 * figure_height_in
figure_pointsize <- 13.6

draw_figure <- function() {
  old <- par(no.readonly = TRUE)
  on.exit(par(old))
  layout(matrix(c(1, 2), nrow = 2), heights = c(0.78, 2.57))
  par(mar = c(0, 0, 0, 0), xpd = NA)
  plot.new()
  text(
    0.5, 0.82,
    "Agreement of user-community assignments by country",
    font = 2, cex = 1.1
  )
  legend_x <- c(0.155, 0.480, 0.840)
  for (legend_index in seq_along(legend_labels)) {
    legend(
      x = legend_x[legend_index], y = 0.34,
      xjust = 0.5, yjust = 0.5,
      legend = legend_labels[legend_index],
      fill = fills[legend_index], border = borders[legend_index],
      density = hatch_density[legend_index],
      angle = hatch_angle[legend_index],
      bty = "n", cex = 1, x.intersp = 0.20
    )
  }
  par(
    mar = c(4.5, 4.2, 0.3, 0.5),
    mgp = c(2.5, 0.62, 0), tcl = -0.25, las = 1
  )
  positions <- barplot(
    values,
    beside = TRUE,
    col = fills,
    border = borders,
    density = hatch_density,
    angle = hatch_angle,
    names.arg = rep("", ncol(values)),
    ylim = c(y_min, 1.38),
    yaxp = c(0, 1, 5),
    ylab = "ARI", xlab = "", main = ""
  )
  abline(h = 0, col = "#777777", lwd = 0.9)
  barplot(
    values,
    beside = TRUE,
    col = fills,
    border = borders,
    density = hatch_density,
    angle = hatch_angle,
    names.arg = rep("", ncol(values)),
    ylim = c(y_min, 1.38),
    yaxp = c(0, 1, 5),
    ylab = "ARI", xlab = "", main = "",
    add = TRUE,
    axes = FALSE
  )
  axis(1, at = colMeans(positions), labels = FALSE, tick = FALSE)
  text(
    colMeans(positions), par("usr")[3] - 0.14,
    labels = country_labels, srt = 20, adj = 1, xpd = NA
  )
  mtext("Country (number of users)", side = 1, line = 3.20)
  label_y <- ifelse(values >= 0, values + 0.025, values - 0.025)
  text(
    positions, label_y,
    labels = sprintf("%.2f", values),
    cex = 1, srt = 90, adj = c(0, 0.5),
    xpd = TRUE
  )
}

pdf(
  file.path(output_dir, "country_label_ari.pdf"),
  width = figure_width_in,
  height = figure_height_in,
  pointsize = figure_pointsize,
  useDingbats = FALSE
)
draw_figure()
dev.off()

png(
  file.path(output_dir, "country_label_ari.png"),
  width = figure_width_in,
  height = figure_height_in,
  units = "in", res = 300,
  pointsize = figure_pointsize
)
draw_figure()
dev.off()

print(comparison)
