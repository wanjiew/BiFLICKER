#!/usr/bin/env python3
"""Prepare the final Last.fm 1K user--artist bipartite network.

FILTERING PARAMETERS (edit here, or override with the command-line options
shown by ``python3 01_prepare_lastfm1k.py --help``):

    MIN_COUNTRY_USERS = 30
        Keep countries represented by at least 30 users in the profile file.
    MIN_ARTIST_USERS = 10
        Keep artists heard by at least 10 retained users.
    MAX_ARTIST_USER_PROPORTION = 0.40
        Keep artists with degree strictly below 0.40 * n, where n is the
        current retained-user count.  At the final n=511, degree >=205 is
        excluded.
    MIN_USER_ARTISTS = 50
        Keep users connected to at least 50 retained artists.
    MAX_TIMESTAMP_EXCLUSIVE = "2009-05-05T00:00:00Z"
        Match the observation window documented with the dataset.
    REQUIRE_VALID_MBID = True
        Artists without an MBID are excluded; there is no name-based fallback.

The user and artist degree filters are alternated until the bipartite core is
stable.  Raw files are streamed and never modified.

Runtime requirements are Python 3 (standard library), R, and the R package
``Matrix``.  The Python program invokes R internally for the spectrum and
figures, so this remains a one-file preparation workflow.

Default outputs are written below ``Final/preparation``:
    data/       users, artists, weighted edge list, binary matrix
    summary/    one consolidated data summary and the numerical spectrum
    figures/    degree-distribution plots and singular-value elbow plots

The BiFLICKER adjacency is binary.  The edge list additionally retains the
number of listening events as ``event_count`` for post-hoc interpretation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Filtering parameters: this is the single place to edit the default design.
# ---------------------------------------------------------------------------
MIN_COUNTRY_USERS = 30
MIN_ARTIST_USERS = 10
MAX_ARTIST_USER_PROPORTION = 0.40
MIN_USER_ARTISTS = 50
MAX_TIMESTAMP_EXCLUSIVE = "2009-05-05T00:00:00Z"
REQUIRE_VALID_MBID = True
SPECTRUM_COMPONENTS_TO_PLOT = 20

DATASET_PAGE = "http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html"
DATASET_DOI = "https://doi.org/10.5281/zenodo.6090214"
BOOK_DOI = "https://doi.org/10.1007/978-3-642-13287-2"
EXPECTED_MD5 = {
    "userid-profile.tsv": "c53608b6b445db201098c1489ea497df",
    "userid-timestamp-artid-artname-traid-traname.tsv":
        "64747b21563e3d2aa95751e0ddc46b68",
}


@dataclass(frozen=True)
class Profile:
    user_id: str
    gender: str
    age: str
    country: str
    registered: str


@dataclass(frozen=True)
class Artist:
    mbid: str
    name: str


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # nosec: file-integrity check, not cryptography
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "q25": quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q75": quantile(values, 0.75),
        "maximum": max(values),
    }


def load_profiles(path: Path) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[:4] != ["#id", "gender", "age", "country"]:
            raise ValueError(f"Unexpected profile header: {header}")
        for row in reader:
            if len(row) < 5:
                continue
            profiles[row[0].strip()] = Profile(
                row[0].strip(), row[1].strip(), row[2].strip(),
                row[3].strip(), row[4].strip()
            )
    return profiles


def choose_countries(
    profiles: dict[str, Profile], minimum: int
) -> tuple[list[str], Counter[str]]:
    counts = Counter(profile.country for profile in profiles.values() if profile.country)
    countries = [
        country for country, count in counts.most_common() if count >= minimum
    ]
    if not countries:
        raise ValueError("Country filter removed every country")
    return countries, counts


def read_selected_events(
    event_path: Path,
    selected_users: set[str],
    cutoff: str,
) -> tuple[dict[str, Counter[str]], dict[str, Artist], Counter[str]]:
    pairs: dict[str, Counter[str]] = {}
    artists: dict[str, Artist] = {}
    audit: Counter[str] = Counter()
    current_user: str | None = None
    current_counts: Counter[str] = Counter()

    def finish_user() -> None:
        if current_user in selected_users:
            pairs[current_user] = current_counts.copy()

    with event_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            audit["raw_event_rows"] += 1
            row = line.rstrip("\r\n").split("\t", 5)
            if len(row) < 4:
                audit["malformed_event_rows"] += 1
                continue
            user_id, timestamp, mbid, artist_name = row[:4]
            if user_id != current_user:
                if current_user is not None:
                    finish_user()
                current_user = user_id
                current_counts = Counter()
            if user_id not in selected_users:
                continue
            if cutoff and timestamp >= cutoff:
                audit["selected_events_after_cutoff"] += 1
                continue
            mbid = mbid.strip().lower()
            if not mbid:
                audit["selected_events_without_mbid"] += 1
                continue
            current_counts[mbid] += 1
            audit["selected_valid_mbid_events"] += 1
            artists.setdefault(mbid, Artist(mbid=mbid, name=artist_name.strip()))
    if current_user is not None:
        finish_user()
    return pairs, artists, audit


def bipartite_core(
    pairs: dict[str, Counter[str]],
    min_artist_users: int,
    max_artist_proportion: float,
    min_user_artists: int,
) -> tuple[list[str], set[str], list[dict[str, float]]]:
    if not 0 < max_artist_proportion <= 1:
        raise ValueError("max_artist_user_proportion must be in (0, 1]")
    active_users = set(pairs)
    active_artists: set[str] = set()
    history: list[dict[str, float]] = []
    while True:
        degrees: Counter[str] = Counter()
        for user_id in active_users:
            degrees.update(pairs[user_id].keys())
        new_artists = {
            mbid for mbid, degree in degrees.items()
            if degree >= min_artist_users
            and degree / len(active_users) < max_artist_proportion
        }
        new_users = {
            user_id for user_id in active_users
            if sum(mbid in new_artists for mbid in pairs[user_id]) >= min_user_artists
        }
        history.append({
            "iteration": len(history) + 1,
            "input_users": len(active_users),
            "candidate_artists": len(degrees),
            "artists_below_lower_bound": sum(
                degree < min_artist_users for degree in degrees.values()
            ),
            "artists_at_or_above_upper_bound": sum(
                degree / len(active_users) >= max_artist_proportion
                for degree in degrees.values()
            ),
            "eligible_artists": len(new_artists),
            "eligible_users": len(new_users),
            "users_below_lower_bound": len(active_users) - len(new_users),
            "artist_upper_degree_exclusive": max_artist_proportion * len(active_users),
        })
        if not new_users or not new_artists:
            raise ValueError("Core filtering removed every user or artist")
        if new_users == active_users and new_artists == active_artists:
            return sorted(new_users), new_artists, history
        active_users, active_artists = new_users, new_artists


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_r_spectrum_and_figures(data_dir: Path, summary_dir: Path, figures_dir: Path) -> None:
    r_code = r'''
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]; summary_dir <- args[2]; figures_dir <- args[3]
users <- read.csv(file.path(data_dir, "users.csv"), stringsAsFactors = FALSE)
artists <- read.csv(file.path(data_dir, "artists.csv"), stringsAsFactors = FALSE)
edges <- read.csv(gzfile(file.path(data_dir, "user_artist_edges.csv.gz")))
B <- sparseMatrix(i=edges$user_index, j=edges$artist_index,
                  x=edges$binary_weight,
                  dims=c(nrow(users), nrow(artists)), giveCsparse=TRUE)
values <- pmax(eigen(as.matrix(tcrossprod(B)), symmetric=TRUE,
                     only.values=TRUE)$values, 0)
singular <- sqrt(values)
energy <- values / sum(values)
spectrum <- data.frame(
  component=seq_along(singular), singular_value=singular,
  squared_singular_value=values, explained_energy=energy,
  cumulative_energy=cumsum(energy),
  next_gap=c(singular[-length(singular)]-singular[-1], NA),
  next_gap_ratio=c(singular[-length(singular)] /
                   pmax(singular[-1], .Machine$double.eps), NA)
)
write.csv(spectrum, file.path(summary_dir, "singular_values.csv"), row.names=FALSE)

draw_elbow <- function() {
  keep <- seq_len(min(20, nrow(spectrum)))
  plot(keep, spectrum$singular_value[keep], type="b", pch=19,
       xlab="Component", ylab="Singular value",
       main="Singular-value spectrum of the cleaned bipartite network")
}
pdf(file.path(figures_dir, "singular_values_elbow.pdf"), width=7, height=5)
draw_elbow(); dev.off()
png(file.path(figures_dir, "singular_values_elbow.png"), width=1400,
    height=1000, res=180)
draw_elbow(); dev.off()

draw_degrees <- function() {
  par(mfrow=c(1,2), mar=c(4,4,3,1))
  hist(users$network_degree, breaks=30, col="#4C78A8", border="white",
       xlab="Number of distinct retained artists", ylab="Number of users",
       main="User degree histogram")
  abline(v=median(users$network_degree), col="#B22222", lty=2, lwd=2)
  hist(artists$user_degree, breaks=30, col="#59A14F", border="white",
       xlab="Number of retained users", ylab="Number of artists",
       main="Artist degree histogram")
  abline(v=median(artists$user_degree), col="#B22222", lty=2, lwd=2)
}
pdf(file.path(figures_dir, "degree_distributions.pdf"), width=10, height=4.5)
draw_degrees(); dev.off()
png(file.path(figures_dir, "degree_distributions.png"), width=1800,
    height=800, res=180)
draw_degrees(); dev.off()
'''
    command = [
        "Rscript", "-e", r_code,
        str(data_dir.resolve()), str(summary_dir.resolve()), str(figures_dir.resolve()),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    lastfm_dir = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path,
        default=lastfm_dir / "data" / "raw" / "lastfm-dataset-1K"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=script.parents[1] / "preparation"
    )
    parser.add_argument("--min-country-users", type=int, default=MIN_COUNTRY_USERS)
    parser.add_argument("--min-artist-users", type=int, default=MIN_ARTIST_USERS)
    parser.add_argument(
        "--max-artist-user-proportion", type=float,
        default=MAX_ARTIST_USER_PROPORTION
    )
    parser.add_argument("--min-user-artists", type=int, default=MIN_USER_ARTISTS)
    parser.add_argument("--max-timestamp-exclusive", default=MAX_TIMESTAMP_EXCLUSIVE)
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_country_users < 1 or args.min_artist_users < 1 or args.min_user_artists < 1:
        raise ValueError("All lower bounds must be positive")

    profile_path = args.raw_dir / "userid-profile.tsv"
    event_path = args.raw_dir / "userid-timestamp-artid-artname-traid-traname.tsv"
    for path in (profile_path, event_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir
    data_dir = output_dir / "data"
    summary_dir = output_dir / "summary"
    figures_dir = output_dir / "figures"
    for directory in (data_dir, summary_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checksums = {}
    if not args.skip_checksums:
        for path in (profile_path, event_path):
            checksums[path.name] = md5sum(path)
            if checksums[path.name] != EXPECTED_MD5[path.name]:
                raise ValueError(f"MD5 mismatch for {path.name}")

    profiles = load_profiles(profile_path)
    countries, profile_country_counts = choose_countries(
        profiles, args.min_country_users
    )
    selected_users = {
        user_id for user_id, profile in profiles.items()
        if profile.country in countries
    }
    pairs, artist_info, event_audit = read_selected_events(
        event_path, selected_users, args.max_timestamp_exclusive
    )
    active_users, active_artists, core_history = bipartite_core(
        pairs, args.min_artist_users, args.max_artist_user_proportion,
        args.min_user_artists
    )

    country_order = sorted(
        countries,
        key=lambda country: (-sum(profiles[u].country == country for u in active_users), country)
    )
    country_id = {country: index for index, country in enumerate(country_order, 1)}
    active_users.sort(key=lambda user: (country_id[profiles[user].country], user))

    artist_degree: Counter[str] = Counter()
    artist_events: Counter[str] = Counter()
    for user_id in active_users:
        for mbid, count in pairs[user_id].items():
            if mbid in active_artists:
                artist_degree[mbid] += 1
                artist_events[mbid] += count
    artist_order = sorted(
        active_artists,
        key=lambda mbid: (-artist_degree[mbid], -artist_events[mbid], mbid)
    )
    user_index = {user: index for index, user in enumerate(active_users, 1)}
    artist_index = {mbid: index for index, mbid in enumerate(artist_order, 1)}

    user_rows = []
    user_degrees = []
    for user_id in active_users:
        profile = profiles[user_id]
        kept = {mbid: count for mbid, count in pairs[user_id].items() if mbid in active_artists}
        full = pairs[user_id]
        full_activity = sum(full.values())
        proportions = [count / full_activity for count in full.values()]
        entropy = 0.0 if len(full) <= 1 else (
            -sum(p * math.log(p) for p in proportions) / math.log(len(full))
        )
        user_degrees.append(len(kept))
        user_rows.append((
            user_index[user_id], user_id, country_id[profile.country], profile.country,
            profile.gender, profile.age, profile.registered, len(kept), sum(kept.values()),
            len(full), full_activity, entropy,
        ))
    write_csv(
        data_dir / "users.csv",
        ["user_index", "user_id", "server_id", "country", "gender", "age",
         "registered", "network_degree", "retained_event_count",
         "full_history_breadth", "full_history_activity",
         "full_history_normalized_shannon_entropy"],
        user_rows,
    )

    write_csv(
        data_dir / "artists.csv",
        ["artist_index", "mbid", "artist_name", "user_degree", "event_count"],
        ((artist_index[mbid], mbid, artist_info[mbid].name,
          artist_degree[mbid], artist_events[mbid]) for mbid in artist_order),
    )

    edge_count = sum(artist_degree.values())
    total_events = sum(artist_events.values())
    country_edges: Counter[int] = Counter()
    country_events: Counter[int] = Counter()
    edge_weights: list[int] = []
    node_count = len(active_users) + len(artist_order)
    parent = list(range(node_count))
    component_size = [1] * node_count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    with gzip.open(
        data_dir / "user_artist_edges.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "user_index", "artist_index", "binary_weight", "event_count"
        ])
        for user_id in active_users:
            server = country_id[profiles[user_id].country]
            for mbid, count in pairs[user_id].items():
                if mbid in active_artists:
                    writer.writerow([user_index[user_id], artist_index[mbid], 1, count])
                    country_edges[server] += 1
                    country_events[server] += count
                    edge_weights.append(count)
                    union(
                        user_index[user_id] - 1,
                        len(active_users) + artist_index[mbid] - 1,
                    )

    with gzip.open(
        data_dir / "bipartite_network.mtx.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        handle.write("%%MatrixMarket matrix coordinate integer general\n")
        handle.write("% Binary user-by-artist adjacency used by BiFLICKER\n")
        handle.write(f"{len(active_users)} {len(artist_order)} {edge_count}\n")
        for user_id in active_users:
            for mbid in pairs[user_id]:
                if mbid in active_artists:
                    handle.write(f"{user_index[user_id]} {artist_index[mbid]} 1\n")

    country_rows = []
    for country in country_order:
        server = country_id[country]
        members = [u for u in active_users if profiles[u].country == country]
        degrees = [len([a for a in pairs[u] if a in active_artists]) for u in members]
        description = describe(degrees)
        country_rows.append((
            server, country, profile_country_counts[country],
            sum(profiles[u].country == country for u in selected_users),
            len(members), country_edges[server], country_events[server],
            description["mean"], description["median"], description["minimum"],
            description["q25"], description["q75"], description["maximum"],
        ))
    artist_degrees = [artist_degree[mbid] for mbid in artist_order]

    user_description = describe(user_degrees)
    artist_description = describe(artist_degrees)
    edge_weight_description = describe(edge_weights)
    roots = Counter(find(node) for node in range(node_count))

    summary = {
        "dataset": "Last.fm 1K",
        "citations": {
            "dataset_page": DATASET_PAGE,
            "archive_doi": DATASET_DOI,
            "book_doi": BOOK_DOI,
        },
        "parameters": {
            "min_country_users": args.min_country_users,
            "country_rule": "profile user count >= min_country_users",
            "require_valid_mbid": REQUIRE_VALID_MBID,
            "min_artist_users": args.min_artist_users,
            "artist_lower_rule": "degree >= min_artist_users",
            "max_artist_user_proportion": args.max_artist_user_proportion,
            "artist_upper_rule": "degree / current retained users < max proportion",
            "min_user_artists": args.min_user_artists,
            "user_rule": "retained artist degree >= min_user_artists",
            "max_timestamp_exclusive": args.max_timestamp_exclusive,
        },
        "raw_checksums_md5": checksums,
        "counts": {
            "profile_users": len(profiles),
            "eligible_countries": len(countries),
            "selected_country_users_before_core": len(selected_users),
            "users_with_selected_event_record": len(pairs),
            "valid_mbid_artists_before_core": len(artist_info),
            "retained_users": len(active_users),
            "retained_artists": len(active_artists),
            "retained_binary_edges": edge_count,
            "retained_listening_events": total_events,
            "first_excluded_artist_degree_at_final_n": math.ceil(
                args.max_artist_user_proportion * len(active_users)
            ),
        },
        "network_summary": {
            "density": edge_count / (len(active_users) * len(artist_order)),
            "user_degree": user_description,
            "artist_degree": artist_description,
            "edge_event_count": edge_weight_description,
            "connected_components": len(roots),
            "largest_component_nodes": max(roots.values()),
            "largest_component_node_fraction": max(roots.values()) / node_count,
        },
        "profile_missingness": {
            "users_with_missing_age": sum(not profiles[u].age for u in active_users),
            "users_with_missing_gender": sum(not profiles[u].gender for u in active_users),
            "artists_with_missing_mbid": sum(not mbid for mbid in artist_order),
        },
        "country_summary": [
            {
                "server_id": row[0],
                "country": row[1],
                "profile_users": row[2],
                "users_before_core": row[3],
                "retained_users": row[4],
                "binary_edges": row[5],
                "retained_events": row[6],
                "average_user_degree": row[7],
                "median_user_degree": row[8],
                "minimum_user_degree": row[9],
                "user_degree_q25": row[10],
                "user_degree_q75": row[11],
                "maximum_user_degree": row[12],
            }
            for row in country_rows
        ],
        "selected_country_profile_counts": {
            country: profile_country_counts[country] for country in countries
        },
        "core_filter_history": core_history,
        "event_stream_audit": dict(event_audit),
    }
    with (summary_dir / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    run_r_spectrum_and_figures(data_dir, summary_dir, figures_dir)

    print("Preparation complete")
    print(f"  countries: {len(country_order)}")
    print(f"  users: {len(active_users)}")
    print(f"  artists: {len(active_artists)}")
    print(f"  binary edges: {edge_count}")
    print(f"  listening events on retained edges: {total_events}")
    print(f"  output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
