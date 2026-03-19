from __future__ import annotations

import argparse
import logging
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import ncaab_ranker as nr

try:
    import fitz
except Exception:  # pragma: no cover - dependency failure is handled at runtime
    fitz = None


REGION_LAYOUTS = {
    "EAST": {"x_min": 0.0, "x_max": 200.0, "y_min": 100.0, "y_max": 330.0},
    "WEST": {"x_min": 680.0, "x_max": 820.0, "y_min": 100.0, "y_max": 330.0},
    "SOUTH": {"x_min": 0.0, "x_max": 200.0, "y_min": 330.0, "y_max": 560.0},
    "MIDWEST": {"x_min": 680.0, "x_max": 820.0, "y_min": 330.0, "y_max": 560.0},
}
FIRST_ROUND_SEED_PAIRS = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]
REGION_ORDER = ["EAST", "WEST", "SOUTH", "MIDWEST"]
ROUND_ORDER = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite 8",
    "Final Four Left",
    "Final Four Right",
    "Championship",
]
POOL_BRACKET_MODES = {"most_likely", "upset_weighted", "pool_ev"}
BLUE_BLOOD_TEAM_CLEAN = {
    "duke",
    "kansas",
    "kentucky",
    "north carolina",
    "uconn",
    "ucla",
    "michigan state",
    "villanova",
    "louisville",
    "indiana",
}
ROUND_PICK_SCORE_COLUMNS = {
    "Round of 64": "pick_score_r32",
    "Round of 32": "pick_score_sweet16",
    "Sweet 16": "pick_score_elite8",
    "Elite 8": "pick_score_final4",
    "Final Four Left": "pick_score_title",
    "Final Four Right": "pick_score_title",
    "Championship": "pick_score_title",
}
ROUND_POOL_EV_WEIGHT = {
    "Round of 64": 0.35,
    "Round of 32": 0.50,
    "Sweet 16": 0.68,
    "Elite 8": 0.82,
    "Final Four Left": 0.92,
    "Final Four Right": 0.92,
    "Championship": 1.00,
}
ROUND_UPSET_BONUS = {
    (8, 9): 0.025,
    (7, 10): 0.045,
    (6, 11): 0.055,
    (5, 12): 0.040,
}
SEED_LINE_PATTERN = re.compile(r"^(?P<seed>\d{1,2})\s*(?P<team>.+?)\s*\((?P<record>\d+-\d+)\)$")
BRACKET_TEAM_ALIASES = {
    "miami fl": "miami hurricanes",
    "queens nc": "queens university royals",
    "cal baptist": "california baptist lancers",
    "penn": "pennsylvania quakers",
}
SCHOOL_QUALIFIER_TOKENS = {
    "a",
    "and",
    "am",
    "aandm",
    "aandt",
    "asheville",
    "atlantic",
    "baptist",
    "central",
    "christian",
    "city",
    "coast",
    "commonwealth",
    "corpus",
    "delta",
    "eastern",
    "fl",
    "fort",
    "gulf",
    "hilo",
    "illinois",
    "international",
    "maryland",
    "n",
    "north",
    "northern",
    "oh",
    "ohio",
    "poly",
    "pacific",
    "state",
    "southern",
    "tech",
    "texas",
    "the",
    "valley",
    "west",
    "western",
    "wilmington",
}


def build_logger() -> logging.Logger:
    logger = logging.getLogger("bracket_simulator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _quiet_model_logger() -> logging.Logger:
    logger = logging.getLogger("bracket_simulator.model")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    return logger


def parse_bracket_pdf(pdf_path: str | Path) -> pd.DataFrame:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required. Install pymupdf to parse the bracket PDF.")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Bracket PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if doc.page_count == 0:
        raise RuntimeError(f"Bracket PDF has no pages: {pdf_path}")

    page = doc[0]
    region_entries: dict[str, list[dict[str, object]]] = {}

    for region, bounds in REGION_LAYOUTS.items():
        lines: list[tuple[float, float, int, str]] = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            if not (bounds["x_min"] <= x0 <= bounds["x_max"]):
                continue
            if not (bounds["y_min"] <= y0 <= bounds["y_max"]):
                continue
            for raw_line in str(text or "").splitlines():
                line = raw_line.strip()
                match = SEED_LINE_PATTERN.match(line)
                if not match:
                    continue
                seed = int(match.group("seed"))
                if not (1 <= seed <= 16):
                    continue
                team = match.group("team").strip()
                lines.append((y0, x0, seed, team))

        lines.sort(key=lambda item: (item[0], item[1]))
        seed_map: dict[int, str] = {}
        for _, _, seed, team in lines:
            if seed not in seed_map:
                seed_map[seed] = team

        if len(seed_map) != 16:
            raise RuntimeError(
                f"Failed to parse 16 seeds for {region}. Found {len(seed_map)} seeds: {sorted(seed_map)}"
            )

        region_entries[region] = [
            {"region": region, "seed": seed, "team": seed_map[seed]}
            for seed in range(1, 17)
        ]

    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        seed_map = {row["seed"]: row["team"] for row in region_entries[region]}
        for seed_a, seed_b in FIRST_ROUND_SEED_PAIRS:
            rows.append(
                {
                    "region": region,
                    "seed": seed_a,
                    "team": seed_map[seed_a],
                    "opponent": seed_map[seed_b],
                    "seed_opponent": seed_b,
                }
            )
            rows.append(
                {
                    "region": region,
                    "seed": seed_b,
                    "team": seed_map[seed_b],
                    "opponent": seed_map[seed_a],
                    "seed_opponent": seed_a,
                }
            )

    bracket_df = pd.DataFrame(rows).sort_values(["region", "seed"]).reset_index(drop=True)
    return bracket_df


def load_ratings_df(project_root: str | Path) -> pd.DataFrame:
    project_root = Path(project_root)
    ratings_path = project_root / "outputs" / "teams_power_full.xlsx"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    ratings_df = pd.read_excel(ratings_path)
    if ratings_df.empty:
        raise RuntimeError(f"Ratings file is empty: {ratings_path}")
    return ratings_df


def rebuild_ratings_df(project_root: str | Path, logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = build_logger() if logger is None else logger
    project_root = Path(project_root)
    games_path = project_root / "outputs" / "games_history.csv"
    cfg_path = project_root / "config.json"

    if not games_path.exists():
        raise FileNotFoundError(f"games_history.csv not found: {games_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {cfg_path}")

    games_df = pd.read_csv(games_path)
    if games_df.empty:
        raise RuntimeError(f"games_history.csv is empty: {games_path}")

    # Historical game IDs are not reliable enough for bracket use; name keys avoid
    # false merges like Nebraska collapsing into unrelated teams via a bad team_id.
    if "team_id" in games_df.columns:
        games_df["team_id"] = ""
    if "opponent_team_id" in games_df.columns:
        games_df["opponent_team_id"] = ""

    cfg = nr.load_config(str(cfg_path))
    asof_day = nr.datetime.now(tz=nr.NY).date()
    logger.info(
        "BRACKET rebuilding_ratings "
        f"source={games_path} rows={len(games_df)} asof={asof_day}"
    )
    ratings_df = nr.compute_ratings(
        games=games_df,
        cfg=cfg,
        logger=logger,
        asof=asof_day,
    )
    if ratings_df.empty:
        raise RuntimeError("compute_ratings returned an empty dataframe for bracket simulation")
    logger.info(
        "BRACKET ratings_rebuilt "
        f"rows={len(ratings_df)} source=games_history"
    )
    return ratings_df


def _build_ratings_lookup(ratings_df: pd.DataFrame) -> pd.DataFrame:
    team_col = "team" if "team" in ratings_df.columns else "Team"
    team_id_col = "team_id" if "team_id" in ratings_df.columns else "teamId"
    lookup = ratings_df.copy()
    lookup["rating_team"] = lookup[team_col].fillna("").astype(str).str.strip()
    lookup["rating_team_id"] = lookup[team_id_col].fillna("").astype(str).apply(nr._safe_team_id)
    lookup["team_clean"] = lookup["rating_team"].apply(nr.clean_team_name)
    lookup["team_key"] = lookup["rating_team"].apply(nr._prediction_name_key)
    lookup["rating_team_lower"] = lookup["rating_team"].str.lower()
    lookup["team_tokens"] = lookup["team_clean"].apply(lambda value: tuple(str(value).split()))
    lookup["rating_games"] = pd.to_numeric(lookup.get("Games", 0), errors="coerce").fillna(0.0)
    lookup = lookup[(lookup["rating_team"] != "") & (lookup["rating_team_id"] != "")].copy()
    return lookup


def _resolve_bracket_team(team_name: str, ratings_lookup: pd.DataFrame) -> pd.Series:
    clean_name = nr.clean_team_name(team_name)
    team_key = nr._prediction_name_key(team_name)
    if not clean_name:
        raise KeyError(f"Unable to resolve blank bracket team name from {team_name!r}")

    raw_name = str(team_name).strip().lower()
    exact_raw = ratings_lookup.loc[ratings_lookup["rating_team_lower"] == raw_name]
    if len(exact_raw) == 1:
        return exact_raw.iloc[0]

    exact_clean = ratings_lookup.loc[ratings_lookup["team_clean"] == clean_name]
    if len(exact_clean) == 1:
        return exact_clean.iloc[0]

    exact_key = ratings_lookup.loc[ratings_lookup["team_key"] == team_key]
    if len(exact_key) == 1:
        return exact_key.iloc[0]

    alias_target = BRACKET_TEAM_ALIASES.get(clean_name, "")
    if alias_target:
        alias_clean = nr.clean_team_name(alias_target)
        alias_key = nr._prediction_name_key(alias_target)
        alias_match = ratings_lookup.loc[
            (ratings_lookup["team_clean"] == alias_clean)
            | (ratings_lookup["team_key"] == alias_key)
            | (ratings_lookup["rating_team_lower"] == alias_target.lower())
        ]
        if len(alias_match) == 1:
            return alias_match.iloc[0]

    query_tokens = tuple(token for token in clean_name.split() if token)
    prefix_candidates: list[tuple[int, int, float, int, str, int]] = []
    if query_tokens:
        query_len = len(query_tokens)
        for idx, row in ratings_lookup.iterrows():
            candidate_tokens = tuple(row["team_tokens"])
            if len(candidate_tokens) < query_len:
                continue
            if candidate_tokens[:query_len] != query_tokens:
                continue
            extra_tokens = candidate_tokens[query_len:]
            next_token = extra_tokens[0] if extra_tokens else ""
            qualifier_penalty = 1 if next_token in SCHOOL_QUALIFIER_TOKENS else 0
            prefix_candidates.append(
                (
                    qualifier_penalty,
                    len(extra_tokens),
                    -float(row["rating_games"]),
                    len(candidate_tokens),
                    str(row["rating_team"]),
                    int(idx),
                )
            )

    if prefix_candidates:
        prefix_candidates.sort()
        if len(prefix_candidates) == 1 or prefix_candidates[0][:4] < prefix_candidates[1][:4]:
            return ratings_lookup.loc[prefix_candidates[0][5]]

    containment_matches, _ = nr._resolve_containment_name_matches(
        [clean_name],
        ratings_lookup["team_clean"].tolist(),
    )
    containment_target = containment_matches.get(clean_name)
    if containment_target:
        containment_df = ratings_lookup.loc[ratings_lookup["team_clean"] == containment_target]
        if len(containment_df) == 1:
            return containment_df.iloc[0]

    raise KeyError(f"Unable to resolve bracket team {team_name!r} (clean={clean_name!r})")


def map_bracket_teams_to_ratings(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = build_logger() if logger is None else logger
    ratings_lookup = _build_ratings_lookup(ratings_df)

    unique_teams = sorted({str(team).strip() for team in bracket_df["team"].dropna().astype(str)})
    resolved_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for team in unique_teams:
        try:
            match = _resolve_bracket_team(team, ratings_lookup)
        except Exception:
            failures.append(team)
            continue
        resolved_rows.append(
            {
                "team": team,
                "rating_team": str(match["rating_team"]),
                "rating_team_id": str(match["rating_team_id"]),
                "rating_team_clean": str(match["team_clean"]),
                "rating_team_key": str(match["team_key"]),
            }
        )

    if failures:
        raise RuntimeError(f"Failed to map bracket teams to ratings: {sorted(failures)}")

    mapping_df = pd.DataFrame(resolved_rows)
    mapped = bracket_df.merge(mapping_df, on="team", how="left")
    mapped = mapped.rename(
        columns={
            "rating_team": "team_rating_team",
            "rating_team_id": "team_rating_team_id",
            "rating_team_clean": "team_rating_team_clean",
            "rating_team_key": "team_rating_team_key",
        }
    )
    mapped = mapped.merge(mapping_df, left_on="opponent", right_on="team", how="left", suffixes=("", "_opp"))
    mapped = mapped.rename(
        columns={
            "rating_team": "opponent_rating_team",
            "rating_team_id": "opponent_rating_team_id",
            "rating_team_clean": "opponent_rating_team_clean",
            "rating_team_key": "opponent_rating_team_key",
        }
    ).drop(columns=["team_opp"])

    missing_team_map = mapped["team_rating_team"].isna().sum()
    missing_opp_map = mapped["opponent_rating_team"].isna().sum()
    if missing_team_map or missing_opp_map:
        raise RuntimeError(
            f"Bracket mapping incomplete. team_missing={missing_team_map} opponent_missing={missing_opp_map}"
        )

    logger.info(
        "BRACKET mapped "
        f"teams={mapped['team'].nunique()} rows={len(mapped)}"
    )
    return mapped


def get_matchup_win_prob(
    team: str,
    opponent: str,
    ratings_df: pd.DataFrame,
    cache: dict | None = None,
) -> float:
    if cache is None:
        cache = {}

    lookup = cache.get("_ratings_lookup")
    if lookup is None:
        ratings_lookup = _build_ratings_lookup(ratings_df)
        lookup = {
            row["rating_team"]: {
                "team": row["rating_team"],
                "team_id": row["rating_team_id"],
                "team_clean": row["team_clean"],
                "team_key": row["team_key"],
            }
            for _, row in ratings_lookup.iterrows()
        }
        cache["_ratings_lookup"] = lookup

    key = (team, opponent)
    if key in cache:
        return float(cache[key])

    if team not in lookup or opponent not in lookup:
        raise KeyError(f"Missing matchup team in ratings lookup: {team!r} vs {opponent!r}")

    team_entry = lookup[team]
    opp_entry = lookup[opponent]
    schedule_df = pd.DataFrame(
        [
            {
                "date": nr.datetime.now(tz=nr.NY).date().isoformat(),
                "team_id": team_entry["team_id"],
                "opponent_team_id": opp_entry["team_id"],
                "team": team_entry["team"],
                "opponent": opp_entry["team"],
                "team_name_clean": team_entry["team_clean"],
                "opponent_name_clean": opp_entry["team_clean"],
                "team_name_norm": team_entry["team_key"],
                "opponent_name_norm": opp_entry["team_key"],
                "neutral_site": True,
            }
        ]
    )

    pred_df = nr.build_game_predictions(
        ratings_df,
        season=int(nr.date.today().year),
        logger=_quiet_model_logger(),
        day=nr.datetime.now(tz=nr.NY).date(),
        use_external_blend=False,
        use_vegas_blend=False,
        schedule_df=schedule_df,
        hca_points_per_100=0.0,
        apply_calibration=True,
    )
    if pred_df.empty:
        raise RuntimeError(f"Unable to build synthetic matchup prediction for {team} vs {opponent}")

    win_prob = float(pred_df.iloc[0]["win_prob"])
    cache[(team, opponent)] = win_prob
    cache[(opponent, team)] = 1.0 - win_prob
    return win_prob


def build_bracket_matchup_cache(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> dict:
    logger = build_logger() if logger is None else logger
    teams_df = (
        bracket_df[
            [
                "team_rating_team",
                "team_rating_team_id",
                "team_rating_team_clean",
                "team_rating_team_key",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "team_rating_team": "team",
                "team_rating_team_id": "team_id",
                "team_rating_team_clean": "team_name_clean",
                "team_rating_team_key": "team_name_norm",
            }
        )
        .sort_values("team")
        .reset_index(drop=True)
    )

    cache = {
        "_ratings_lookup": {
            str(row["team"]): {
                "team": str(row["team"]),
                "team_id": str(row["team_id"]),
                "team_clean": str(row["team_name_clean"]),
                "team_key": str(row["team_name_norm"]),
            }
            for _, row in teams_df.iterrows()
        }
    }

    schedule_rows: list[dict[str, object]] = []
    today_str = nr.datetime.now(tz=nr.NY).date().isoformat()
    for team_a, team_b in combinations(teams_df.to_dict(orient="records"), 2):
        schedule_rows.append(
            {
                "date": today_str,
                "team_id": team_a["team_id"],
                "opponent_team_id": team_b["team_id"],
                "team": team_a["team"],
                "opponent": team_b["team"],
                "team_name_clean": team_a["team_name_clean"],
                "opponent_name_clean": team_b["team_name_clean"],
                "team_name_norm": team_a["team_name_norm"],
                "opponent_name_norm": team_b["team_name_norm"],
                "neutral_site": True,
            }
        )

    pred_df = nr.build_game_predictions(
        ratings_df,
        season=int(nr.date.today().year),
        logger=_quiet_model_logger(),
        day=nr.datetime.now(tz=nr.NY).date(),
        use_external_blend=False,
        use_vegas_blend=False,
        schedule_df=pd.DataFrame(schedule_rows),
        hca_points_per_100=0.0,
        apply_calibration=True,
    )
    if pred_df.empty:
        raise RuntimeError("Failed to build tournament matchup cache")

    for _, row in pred_df.iterrows():
        team = str(row["team"])
        opponent = str(row["opponent"])
        win_prob = float(row["win_prob"])
        cache[(team, opponent)] = win_prob
        cache[(opponent, team)] = 1.0 - win_prob

    logger.info(
        "BRACKET matchup_cache_built "
        f"teams={len(teams_df)} matchups={len(pred_df)}"
    )
    return cache


def _build_team_probability_lookup(sim_df: pd.DataFrame | None) -> dict[str, dict[str, float | int | str]]:
    if sim_df is None or sim_df.empty:
        return {}

    ranked = (
        sim_df.copy()
        .sort_values(
            ["title_prob", "final4_prob", "elite8_prob", "sweet16_prob", "seed"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    ranked["title_rank"] = np.arange(1, len(ranked) + 1)
    lookup: dict[str, dict[str, float | int | str]] = {}
    for _, row in ranked.iterrows():
        lookup[str(row["team"])] = {
            "seed": int(row["seed"]),
            "region": str(row["region"]),
            "r32_prob": float(row["r32_prob"]),
            "sweet16_prob": float(row["sweet16_prob"]),
            "elite8_prob": float(row["elite8_prob"]),
            "final4_prob": float(row["final4_prob"]),
            "title_prob": float(row["title_prob"]),
            "title_rank": int(row["title_rank"]),
        }
    return lookup


def estimate_pick_rates(bracket_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "team",
        "seed",
        "region",
        "title_rank",
        "brand_score",
        "pick_score_r32",
        "pick_score_sweet16",
        "pick_score_elite8",
        "pick_score_final4",
        "pick_score_title",
    ]
    if bracket_predictions is None or bracket_predictions.empty:
        return pd.DataFrame(columns=columns)

    work = bracket_predictions.copy()
    work["seed"] = pd.to_numeric(work["seed"], errors="coerce")
    for col in ["r32_prob", "sweet16_prob", "elite8_prob", "final4_prob", "title_prob"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["team", "seed", "r32_prob", "sweet16_prob", "elite8_prob", "final4_prob", "title_prob"]).copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work = work.sort_values(
        ["title_prob", "final4_prob", "elite8_prob", "sweet16_prob", "seed"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    work["title_rank"] = np.arange(1, len(work) + 1)
    max_rank = max(len(work) - 1, 1)

    work["seed_strength"] = ((17.0 - work["seed"]) / 16.0).clip(lower=0.0, upper=1.0)
    work["rank_strength"] = (1.0 - ((work["title_rank"] - 1) / max_rank)).clip(lower=0.0, upper=1.0)
    work["title_strength"] = (work["title_prob"] / max(float(work["title_prob"].max()), 1e-6)).clip(0.0, 1.0)
    work["final4_strength"] = (work["final4_prob"] / max(float(work["final4_prob"].max()), 1e-6)).clip(0.0, 1.0)
    work["elite8_strength"] = (work["elite8_prob"] / max(float(work["elite8_prob"].max()), 1e-6)).clip(0.0, 1.0)
    work["sweet16_strength"] = (work["sweet16_prob"] / max(float(work["sweet16_prob"].max()), 1e-6)).clip(0.0, 1.0)
    work["blue_blood_bonus"] = work["team"].fillna("").astype(str).apply(
        lambda name: 1.0 if nr.clean_team_name(name) in BLUE_BLOOD_TEAM_CLEAN else 0.0
    )
    work["one_seed_bonus"] = (work["seed"] == 1).astype(float)
    work["brand_score"] = (
        0.40 * work["seed_strength"]
        + 0.25 * work["rank_strength"]
        + 0.20 * work["title_strength"]
        + 0.15 * work["blue_blood_bonus"]
        + 0.10 * work["one_seed_bonus"]
    ).clip(lower=0.0)

    work["pick_score_r32"] = (
        0.52 * work["seed_strength"]
        + 0.18 * work["rank_strength"]
        + 0.12 * work["r32_prob"]
        + 0.08 * work["title_strength"]
        + 0.10 * work["brand_score"]
    )
    work["pick_score_sweet16"] = (
        0.40 * work["seed_strength"]
        + 0.12 * work["rank_strength"]
        + 0.20 * work["sweet16_strength"]
        + 0.15 * work["final4_strength"]
        + 0.13 * work["brand_score"]
    )
    work["pick_score_elite8"] = (
        0.26 * work["seed_strength"]
        + 0.12 * work["rank_strength"]
        + 0.24 * work["elite8_strength"]
        + 0.20 * work["final4_strength"]
        + 0.18 * work["brand_score"]
    )
    work["pick_score_final4"] = (
        0.16 * work["seed_strength"]
        + 0.10 * work["rank_strength"]
        + 0.24 * work["final4_strength"]
        + 0.28 * work["title_strength"]
        + 0.22 * work["brand_score"]
    )
    work["pick_score_title"] = (
        0.10 * work["seed_strength"]
        + 0.08 * work["rank_strength"]
        + 0.18 * work["final4_strength"]
        + 0.42 * work["title_strength"]
        + 0.22 * work["brand_score"]
    )

    for col in [
        "pick_score_r32",
        "pick_score_sweet16",
        "pick_score_elite8",
        "pick_score_final4",
        "pick_score_title",
    ]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=1e-6)

    return work[columns].copy()


def _pool_size_profile(pool_size: int) -> dict[str, float]:
    size = max(int(pool_size), 1)
    pick_rate_exponent = 1.0 + (float(np.log(size)) / 3.0)
    if size < 20:
        leverage_scale = 0.65
        late_round_bonus = 0.00
    elif size < 100:
        leverage_scale = 1.00
        late_round_bonus = 0.04
    else:
        leverage_scale = 1.25
        late_round_bonus = 0.10
    return {
        "pool_size": float(size),
        "pick_rate_exponent": pick_rate_exponent,
        "leverage_scale": leverage_scale,
        "late_round_bonus": late_round_bonus,
    }


def _matchup_pick_rates(
    team_a: dict[str, object],
    team_b: dict[str, object],
    round_name: str,
    pick_rate_lookup: dict[str, dict[str, float | int | str]],
    pool_size: int = 50,
) -> tuple[float, float]:
    round_col = ROUND_PICK_SCORE_COLUMNS.get(round_name, "pick_score_title")
    score_a = float(pick_rate_lookup.get(str(team_a["team"]), {}).get(round_col, 1.0))
    score_b = float(pick_rate_lookup.get(str(team_b["team"]), {}).get(round_col, 1.0))
    score_a = max(score_a, 1e-6)
    score_b = max(score_b, 1e-6)
    total = score_a + score_b
    if total <= 0:
        return 0.5, 0.5
    base_a = float(score_a / total)
    base_b = float(score_b / total)

    exponent = _pool_size_profile(pool_size)["pick_rate_exponent"]
    adjusted_a = max(base_a, 1e-6) ** exponent
    adjusted_b = max(base_b, 1e-6) ** exponent
    adjusted_total = adjusted_a + adjusted_b
    if adjusted_total <= 0:
        return base_a, base_b
    return float(adjusted_a / adjusted_total), float(adjusted_b / adjusted_total)


def _pick_game_for_mode(
    team_a: dict[str, object],
    team_b: dict[str, object],
    round_name: str,
    ratings_df: pd.DataFrame,
    cache: dict,
    mode: str,
    team_prob_lookup: dict[str, dict[str, float | int | str]],
    pick_rate_lookup: dict[str, dict[str, float | int | str]],
    pool_size: int = 50,
) -> tuple[dict[str, object], float, float, float, float]:
    win_prob = get_matchup_win_prob(
        str(team_a["rating_team"]),
        str(team_b["rating_team"]),
        ratings_df,
        cache=cache,
    )
    team_a_pick_rate, team_b_pick_rate = _matchup_pick_rates(
        team_a,
        team_b,
        round_name,
        pick_rate_lookup,
        pool_size=pool_size,
    )
    if mode == "most_likely":
        winner = team_a if win_prob >= 0.5 else team_b
        winner_pick_rate = team_a_pick_rate if winner is team_a else team_b_pick_rate
        winner_win_prob = float(win_prob) if winner is team_a else float(1.0 - win_prob)
        return winner, float(win_prob), winner_pick_rate, winner_win_prob - winner_pick_rate, winner_win_prob

    team_a_seed = int(team_a["seed"])
    team_b_seed = int(team_b["seed"])
    pair = tuple(sorted((team_a_seed, team_b_seed)))
    pool_profile = _pool_size_profile(pool_size)

    if mode == "upset_weighted":
        bonus_a = 0.0
        bonus_b = 0.0
        if team_a_seed > team_b_seed and float(win_prob) >= 0.34:
            bonus_a = ROUND_UPSET_BONUS.get(pair, 0.0)
        if team_b_seed > team_a_seed and float(1.0 - win_prob) >= 0.34:
            bonus_b = ROUND_UPSET_BONUS.get(pair, 0.0)
        future_col = {
            "Round of 64": "sweet16_prob",
            "Round of 32": "elite8_prob",
            "Sweet 16": "final4_prob",
            "Elite 8": "title_prob",
            "Final Four Left": "title_prob",
            "Final Four Right": "title_prob",
            "Championship": "title_prob",
        }.get(round_name, "title_prob")
        future_a = 0.15 * float(team_prob_lookup.get(str(team_a["team"]), {}).get(future_col, 0.0))
        future_b = 0.15 * float(team_prob_lookup.get(str(team_b["team"]), {}).get(future_col, 0.0))
        score_a = float(win_prob) + bonus_a + future_a
        score_b = float(1.0 - win_prob) + bonus_b + future_b
    else:
        ev_weight = float(ROUND_POOL_EV_WEIGHT.get(round_name, 1.0)) * float(pool_profile["leverage_scale"])
        ev_weight = min(max(ev_weight, 0.0), 1.0)
        pure_a = float(win_prob)
        pure_b = float(1.0 - win_prob)
        ev_a = pure_a * (1.0 - team_a_pick_rate)
        ev_b = pure_b * (1.0 - team_b_pick_rate)
        if round_name in {"Elite 8", "Final Four Left", "Final Four Right", "Championship"}:
            late_round_bonus = float(pool_profile["late_round_bonus"])
            if team_a_seed > team_b_seed:
                ev_a += late_round_bonus * max(pure_a - team_a_pick_rate, 0.0)
            if team_b_seed > team_a_seed:
                ev_b += late_round_bonus * max(pure_b - team_b_pick_rate, 0.0)
        score_a = (1.0 - ev_weight) * pure_a + ev_weight * ev_a
        score_b = (1.0 - ev_weight) * pure_b + ev_weight * ev_b

    winner = team_a if score_a >= score_b else team_b
    winner_pick_rate = team_a_pick_rate if winner is team_a else team_b_pick_rate
    winner_win_prob = float(win_prob) if winner is team_a else float(1.0 - win_prob)
    return winner, float(win_prob), winner_pick_rate, winner_win_prob - winner_pick_rate, winner_win_prob


def _build_bracket_game_nodes(bracket_df: pd.DataFrame) -> list[dict[str, object]]:
    nodes: list[dict[str, object]] = []
    regional_winner_nodes: dict[str, dict[str, object]] = {}

    for region in REGION_ORDER:
        seeds = _region_seed_records(bracket_df, region)
        round64 = [
            {
                "round": "Round of 64",
                "region": region,
                "slot": slot_idx + 1,
                "left": seeds[seed_a],
                "right": seeds[seed_b],
            }
            for slot_idx, (seed_a, seed_b) in enumerate(FIRST_ROUND_SEED_PAIRS)
        ]
        round32 = [
            {
                "round": "Round of 32",
                "region": region,
                "slot": slot_idx + 1,
                "left": round64[idx],
                "right": round64[idx + 1],
            }
            for slot_idx, idx in enumerate(range(0, len(round64), 2))
        ]
        sweet16 = [
            {
                "round": "Sweet 16",
                "region": region,
                "slot": slot_idx + 1,
                "left": round32[idx],
                "right": round32[idx + 1],
            }
            for slot_idx, idx in enumerate(range(0, len(round32), 2))
        ]
        elite8 = {
            "round": "Elite 8",
            "region": region,
            "slot": 1,
            "left": sweet16[0],
            "right": sweet16[1],
        }
        nodes.extend(round64)
        nodes.extend(round32)
        nodes.extend(sweet16)
        nodes.append(elite8)
        regional_winner_nodes[region] = elite8

    semifinal_left = {
        "round": "Final Four Left",
        "region": "EAST vs SOUTH",
        "slot": 1,
        "left": regional_winner_nodes["EAST"],
        "right": regional_winner_nodes["SOUTH"],
    }
    semifinal_right = {
        "round": "Final Four Right",
        "region": "WEST vs MIDWEST",
        "slot": 2,
        "left": regional_winner_nodes["WEST"],
        "right": regional_winner_nodes["MIDWEST"],
    }
    championship = {
        "round": "Championship",
        "region": "National Championship",
        "slot": 1,
        "left": semifinal_left,
        "right": semifinal_right,
    }
    nodes.extend([semifinal_left, semifinal_right, championship])
    return nodes


def _possible_teams_from_entry(entry: dict[str, object]) -> list[dict[str, object]]:
    if "left" not in entry or "right" not in entry:
        return [entry]

    left_teams = _possible_teams_from_entry(entry["left"])
    right_teams = _possible_teams_from_entry(entry["right"])
    deduped: dict[str, dict[str, object]] = {}
    for team in left_teams + right_teams:
        deduped[str(team["team"])] = team
    return list(deduped.values())


def build_bracket_matchup_probs(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    cache: dict | None = None,
) -> pd.DataFrame:
    if cache is None:
        cache = {}

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    round_order = {name: idx for idx, name in enumerate(ROUND_ORDER)}
    for node in _build_bracket_game_nodes(bracket_df):
        left_teams = _possible_teams_from_entry(node["left"])
        right_teams = _possible_teams_from_entry(node["right"])
        for team_a in left_teams:
            for team_b in right_teams:
                win_prob = get_matchup_win_prob(
                    str(team_a["rating_team"]),
                    str(team_b["rating_team"]),
                    ratings_df,
                    cache=cache,
                )
                for current_team, current_opponent, current_prob in (
                    (team_a, team_b, win_prob),
                    (team_b, team_a, 1.0 - win_prob),
                ):
                    dedupe_key = (
                        str(node["round"]),
                        str(current_team["team"]),
                        str(current_opponent["team"]),
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    rows.append(
                        {
                            "round": str(node["round"]),
                            "region": str(node["region"]),
                            "team": str(current_team["team"]),
                            "seed": int(current_team["seed"]),
                            "opponent": str(current_opponent["team"]),
                            "seed_opponent": int(current_opponent["seed"]),
                            "win_prob": round(float(current_prob), 4),
                        }
                    )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["round", "region", "team", "opponent"],
            key=lambda s: s.map(round_order).fillna(999) if s.name == "round" else s,
        )
        .reset_index(drop=True)
    )


def _play_game(
    team_a: dict[str, object],
    team_b: dict[str, object],
    ratings_df: pd.DataFrame,
    cache: dict,
    rng: np.random.Generator | None = None,
    deterministic: bool = False,
) -> tuple[dict[str, object], float]:
    win_prob = get_matchup_win_prob(
        str(team_a["rating_team"]),
        str(team_b["rating_team"]),
        ratings_df,
        cache=cache,
    )
    if deterministic:
        winner = team_a if win_prob >= 0.5 else team_b
    else:
        if rng is None:
            raise ValueError("rng is required for stochastic simulation")
        winner = team_a if float(rng.random()) < win_prob else team_b
    return winner, win_prob


def _region_seed_records(bracket_df: pd.DataFrame, region: str) -> dict[int, dict[str, object]]:
    region_df = (
        bracket_df.loc[bracket_df["region"] == region]
        .sort_values("seed")
        .drop_duplicates(subset=["region", "seed"])
    )
    return {
        int(row["seed"]): {
            "region": region,
            "seed": int(row["seed"]),
            "team": str(row["team"]),
            "rating_team": str(row["team_rating_team"]),
            "team_id": str(row["team_rating_team_id"]),
        }
        for _, row in region_df.iterrows()
    }


def simulate_single_bracket(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    cache: dict | None = None,
    deterministic: bool = False,
) -> tuple[dict[str, dict[str, int]], list[dict[str, object]]]:
    if cache is None:
        cache = {}
    if rng is None and not deterministic:
        rng = np.random.default_rng()

    advancement: dict[str, dict[str, int]] = {}
    for _, row in bracket_df.drop_duplicates(subset=["team"]).iterrows():
        advancement[str(row["team"])] = {
            "r32": 0,
            "sweet16": 0,
            "elite8": 0,
            "final4": 0,
            "title": 0,
        }

    bracket_rows: list[dict[str, object]] = []
    regional_winners: dict[str, dict[str, object]] = {}

    for region in REGION_ORDER:
        seeds = _region_seed_records(bracket_df, region)

        round64_winners: list[dict[str, object]] = []
        for seed_a, seed_b in FIRST_ROUND_SEED_PAIRS:
            winner, win_prob = _play_game(seeds[seed_a], seeds[seed_b], ratings_df, cache, rng, deterministic)
            round64_winners.append(winner)
            advancement[str(winner["team"])]["r32"] = 1
            bracket_rows.append(
                {
                    "round": "Round of 64",
                    "region": region,
                    "team": seeds[seed_a]["team"],
                    "seed": seeds[seed_a]["seed"],
                    "opponent": seeds[seed_b]["team"],
                    "seed_opponent": seeds[seed_b]["seed"],
                    "winner": winner["team"],
                    "winner_seed": winner["seed"],
                    "win_prob": round(win_prob, 4),
                }
            )

        round32_winners: list[dict[str, object]] = []
        for idx in range(0, len(round64_winners), 2):
            team_a = round64_winners[idx]
            team_b = round64_winners[idx + 1]
            winner, win_prob = _play_game(team_a, team_b, ratings_df, cache, rng, deterministic)
            round32_winners.append(winner)
            advancement[str(winner["team"])]["sweet16"] = 1
            bracket_rows.append(
                {
                    "round": "Round of 32",
                    "region": region,
                    "team": team_a["team"],
                    "seed": team_a["seed"],
                    "opponent": team_b["team"],
                    "seed_opponent": team_b["seed"],
                    "winner": winner["team"],
                    "winner_seed": winner["seed"],
                    "win_prob": round(win_prob, 4),
                }
            )

        sweet16_winners: list[dict[str, object]] = []
        for idx in range(0, len(round32_winners), 2):
            team_a = round32_winners[idx]
            team_b = round32_winners[idx + 1]
            winner, win_prob = _play_game(team_a, team_b, ratings_df, cache, rng, deterministic)
            sweet16_winners.append(winner)
            advancement[str(winner["team"])]["elite8"] = 1
            bracket_rows.append(
                {
                    "round": "Sweet 16",
                    "region": region,
                    "team": team_a["team"],
                    "seed": team_a["seed"],
                    "opponent": team_b["team"],
                    "seed_opponent": team_b["seed"],
                    "winner": winner["team"],
                    "winner_seed": winner["seed"],
                    "win_prob": round(win_prob, 4),
                }
            )

        regional_winner, win_prob = _play_game(
            sweet16_winners[0],
            sweet16_winners[1],
            ratings_df,
            cache,
            rng,
            deterministic,
        )
        regional_winners[region] = regional_winner
        advancement[str(regional_winner["team"])]["final4"] = 1
        bracket_rows.append(
            {
                "round": "Elite 8",
                "region": region,
                "team": sweet16_winners[0]["team"],
                "seed": sweet16_winners[0]["seed"],
                "opponent": sweet16_winners[1]["team"],
                "seed_opponent": sweet16_winners[1]["seed"],
                "winner": regional_winner["team"],
                "winner_seed": regional_winner["seed"],
                "win_prob": round(win_prob, 4),
            }
        )

    semifinal_pairs = [
        ("EAST", "SOUTH", "Final Four Left"),
        ("WEST", "MIDWEST", "Final Four Right"),
    ]
    semifinal_winners: list[dict[str, object]] = []
    for region_a, region_b, round_name in semifinal_pairs:
        winner, win_prob = _play_game(
            regional_winners[region_a],
            regional_winners[region_b],
            ratings_df,
            cache,
            rng,
            deterministic,
        )
        semifinal_winners.append(winner)
        bracket_rows.append(
            {
                "round": round_name,
                "region": f"{region_a} vs {region_b}",
                "team": regional_winners[region_a]["team"],
                "seed": regional_winners[region_a]["seed"],
                "opponent": regional_winners[region_b]["team"],
                "seed_opponent": regional_winners[region_b]["seed"],
                "winner": winner["team"],
                "winner_seed": winner["seed"],
                "win_prob": round(win_prob, 4),
            }
        )

    champion, win_prob = _play_game(
        semifinal_winners[0],
        semifinal_winners[1],
        ratings_df,
        cache,
        rng,
        deterministic,
    )
    advancement[str(champion["team"])]["title"] = 1
    bracket_rows.append(
        {
            "round": "Championship",
            "region": "National Championship",
            "team": semifinal_winners[0]["team"],
            "seed": semifinal_winners[0]["seed"],
            "opponent": semifinal_winners[1]["team"],
            "seed_opponent": semifinal_winners[1]["seed"],
            "winner": champion["team"],
            "winner_seed": champion["seed"],
            "win_prob": round(win_prob, 4),
        }
    )

    return advancement, bracket_rows


def simulate_bracket(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n_simulations: int = 20000,
    seed: int = 20260319,
    cache: dict | None = None,
) -> pd.DataFrame:
    teams_df = (
        bracket_df[["team", "seed", "region"]]
        .drop_duplicates(subset=["team"])
        .sort_values(["region", "seed", "team"])
        .reset_index(drop=True)
    )
    counts = {
        str(row["team"]): {
            "r32": 0,
            "sweet16": 0,
            "elite8": 0,
            "final4": 0,
            "title": 0,
            "seed": int(row["seed"]),
            "region": str(row["region"]),
        }
        for _, row in teams_df.iterrows()
    }

    rng = np.random.default_rng(seed)
    if cache is None:
        cache = {}
    for _ in range(int(n_simulations)):
        advancement, _ = simulate_single_bracket(
            bracket_df=bracket_df,
            ratings_df=ratings_df,
            rng=rng,
            cache=cache,
            deterministic=False,
        )
        for team, rounds in advancement.items():
            for round_name, value in rounds.items():
                counts[team][round_name] += int(value)

    rows: list[dict[str, object]] = []
    for team, info in counts.items():
        rows.append(
            {
                "team": team,
                "seed": info["seed"],
                "region": info["region"],
                "r32_prob": round(info["r32"] / n_simulations, 4),
                "sweet16_prob": round(info["sweet16"] / n_simulations, 4),
                "elite8_prob": round(info["elite8"] / n_simulations, 4),
                "final4_prob": round(info["final4"] / n_simulations, 4),
                "title_prob": round(info["title"] / n_simulations, 4),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["title_prob", "final4_prob", "elite8_prob", "region", "seed"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def build_most_likely_bracket(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    cache: dict | None = None,
    sim_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return build_pool_bracket(
        bracket_df=bracket_df,
        ratings_df=ratings_df,
        mode="most_likely",
        cache=cache,
        sim_df=sim_df,
    )


def build_pool_bracket(
    bracket_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    mode: str = "pool_ev",
    cache: dict | None = None,
    sim_df: pd.DataFrame | None = None,
    pool_size: int = 50,
) -> pd.DataFrame:
    if mode not in POOL_BRACKET_MODES:
        raise ValueError(f"Unsupported pool bracket mode: {mode}")

    if cache is None:
        cache = {}
    team_prob_lookup = _build_team_probability_lookup(sim_df)
    pick_rate_df = estimate_pick_rates(sim_df if sim_df is not None else pd.DataFrame())
    pick_rate_lookup = {
        str(row["team"]): row.to_dict()
        for _, row in pick_rate_df.iterrows()
    }
    bracket_rows: list[dict[str, object]] = []
    regional_winners: dict[str, dict[str, object]] = {}

    for region in REGION_ORDER:
        seeds = _region_seed_records(bracket_df, region)

        round64_winners: list[dict[str, object]] = []
        for seed_a, seed_b in FIRST_ROUND_SEED_PAIRS:
            winner, raw_win_prob, pick_rate, leverage_score, winner_win_prob = _pick_game_for_mode(
                seeds[seed_a],
                seeds[seed_b],
                "Round of 64",
                ratings_df,
                cache,
                mode,
                team_prob_lookup,
                pick_rate_lookup,
                pool_size=pool_size,
            )
            round64_winners.append(winner)
            bracket_rows.append(
                {
                    "mode": mode,
                    "round": "Round of 64",
                    "region": region,
                    "team": seeds[seed_a]["team"],
                    "seed": seeds[seed_a]["seed"],
                    "opponent": seeds[seed_b]["team"],
                    "seed_opponent": seeds[seed_b]["seed"],
                    "winner": winner["team"],
                    "winner_seed": winner["seed"],
                    "win_prob": round(winner_win_prob, 4),
                    "pick_rate": round(pick_rate, 4),
                    "leverage_score": round(leverage_score, 4),
                    "pool_size": int(pool_size),
                }
            )

        round32_winners: list[dict[str, object]] = []
        for idx in range(0, len(round64_winners), 2):
            team_a = round64_winners[idx]
            team_b = round64_winners[idx + 1]
            winner, raw_win_prob, pick_rate, leverage_score, winner_win_prob = _pick_game_for_mode(
                team_a,
                team_b,
                "Round of 32",
                ratings_df,
                cache,
                mode,
                team_prob_lookup,
                pick_rate_lookup,
                pool_size=pool_size,
            )
            round32_winners.append(winner)
            bracket_rows.append(
                {
                    "mode": mode,
                    "round": "Round of 32",
                    "region": region,
                    "team": team_a["team"],
                    "seed": team_a["seed"],
                    "opponent": team_b["team"],
                    "seed_opponent": team_b["seed"],
                    "winner": winner["team"],
                    "winner_seed": winner["seed"],
                    "win_prob": round(winner_win_prob, 4),
                    "pick_rate": round(pick_rate, 4),
                    "leverage_score": round(leverage_score, 4),
                    "pool_size": int(pool_size),
                }
            )

        sweet16_winners: list[dict[str, object]] = []
        for idx in range(0, len(round32_winners), 2):
            team_a = round32_winners[idx]
            team_b = round32_winners[idx + 1]
            winner, raw_win_prob, pick_rate, leverage_score, winner_win_prob = _pick_game_for_mode(
                team_a,
                team_b,
                "Sweet 16",
                ratings_df,
                cache,
                mode,
                team_prob_lookup,
                pick_rate_lookup,
                pool_size=pool_size,
            )
            sweet16_winners.append(winner)
            bracket_rows.append(
                {
                    "mode": mode,
                    "round": "Sweet 16",
                    "region": region,
                    "team": team_a["team"],
                    "seed": team_a["seed"],
                    "opponent": team_b["team"],
                    "seed_opponent": team_b["seed"],
                    "winner": winner["team"],
                    "winner_seed": winner["seed"],
                    "win_prob": round(winner_win_prob, 4),
                    "pick_rate": round(pick_rate, 4),
                    "leverage_score": round(leverage_score, 4),
                    "pool_size": int(pool_size),
                }
            )

        regional_winner, raw_win_prob, pick_rate, leverage_score, winner_win_prob = _pick_game_for_mode(
            sweet16_winners[0],
            sweet16_winners[1],
            "Elite 8",
            ratings_df,
            cache,
            mode,
            team_prob_lookup,
            pick_rate_lookup,
            pool_size=pool_size,
        )
        regional_winners[region] = regional_winner
        bracket_rows.append(
            {
                "mode": mode,
                "round": "Elite 8",
                "region": region,
                "team": sweet16_winners[0]["team"],
                "seed": sweet16_winners[0]["seed"],
                "opponent": sweet16_winners[1]["team"],
                "seed_opponent": sweet16_winners[1]["seed"],
                "winner": regional_winner["team"],
                "winner_seed": regional_winner["seed"],
                "win_prob": round(winner_win_prob, 4),
                "pick_rate": round(pick_rate, 4),
                "leverage_score": round(leverage_score, 4),
                "pool_size": int(pool_size),
            }
        )

    semifinal_pairs = [
        ("EAST", "SOUTH", "Final Four Left"),
        ("WEST", "MIDWEST", "Final Four Right"),
    ]
    semifinal_winners: list[dict[str, object]] = []
    for region_a, region_b, round_name in semifinal_pairs:
        winner, raw_win_prob, pick_rate, leverage_score, winner_win_prob = _pick_game_for_mode(
            regional_winners[region_a],
            regional_winners[region_b],
            round_name,
            ratings_df,
            cache,
            mode,
            team_prob_lookup,
            pick_rate_lookup,
            pool_size=pool_size,
        )
        semifinal_winners.append(winner)
        bracket_rows.append(
            {
                "mode": mode,
                "round": round_name,
                "region": f"{region_a} vs {region_b}",
                "team": regional_winners[region_a]["team"],
                "seed": regional_winners[region_a]["seed"],
                "opponent": regional_winners[region_b]["team"],
                "seed_opponent": regional_winners[region_b]["seed"],
                "winner": winner["team"],
                "winner_seed": winner["seed"],
                "win_prob": round(winner_win_prob, 4),
                "pick_rate": round(pick_rate, 4),
                "leverage_score": round(leverage_score, 4),
                "pool_size": int(pool_size),
            }
        )

    champion, raw_win_prob, pick_rate, leverage_score, winner_win_prob = _pick_game_for_mode(
        semifinal_winners[0],
        semifinal_winners[1],
        "Championship",
        ratings_df,
        cache,
        mode,
        team_prob_lookup,
        pick_rate_lookup,
        pool_size=pool_size,
    )
    bracket_rows.append(
        {
            "mode": mode,
            "round": "Championship",
            "region": "National Championship",
            "team": semifinal_winners[0]["team"],
            "seed": semifinal_winners[0]["seed"],
            "opponent": semifinal_winners[1]["team"],
            "seed_opponent": semifinal_winners[1]["seed"],
            "winner": champion["team"],
            "winner_seed": champion["seed"],
            "win_prob": round(winner_win_prob, 4),
            "pick_rate": round(pick_rate, 4),
            "leverage_score": round(leverage_score, 4),
            "pool_size": int(pool_size),
        }
    )
    return pd.DataFrame(bracket_rows)


def validate_bracket(
    bracket_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> None:
    logger = build_logger() if logger is None else logger
    unique_teams = bracket_df["team"].nunique()
    teams_per_region = bracket_df.groupby("region")["team"].nunique().to_dict()
    matchups_per_region = {region: count // 2 for region, count in teams_per_region.items()}
    duplicates = (
        bracket_df["team"].value_counts().loc[lambda s: s > 1].index.tolist()
    )
    logger.info(f"BRACKET validate total_teams={unique_teams}")
    logger.info(f"BRACKET validate matchups_per_region={matchups_per_region}")
    logger.info(f"BRACKET validate duplicate_team_count={len(duplicates)}")
    if unique_teams != 64:
        raise RuntimeError(f"Expected 64 teams in bracket, found {unique_teams}")
    if any(count != 16 for count in teams_per_region.values()):
        raise RuntimeError(f"Expected 16 teams per region, found {teams_per_region}")
    if any(count != 8 for count in matchups_per_region.values()):
        raise RuntimeError(f"Expected 8 matchups per region, found {matchups_per_region}")
    if len(duplicates) != 0:
        raise RuntimeError(f"Duplicate teams found in bracket: {duplicates}")


def run_bracket_predictions(
    pdf_path: str | Path,
    project_root: str | Path,
    n_simulations: int = 20000,
    pool_size: int = 50,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger = build_logger() if logger is None else logger
    project_root = Path(project_root)
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    bracket_df = parse_bracket_pdf(pdf_path)
    validate_bracket(bracket_df, logger=logger)

    ratings_df = load_ratings_df(project_root)
    try:
        mapped_bracket_df = map_bracket_teams_to_ratings(bracket_df, ratings_df, logger=logger)
    except RuntimeError as ex:
        logger.warning(
            "BRACKET ratings_source_incomplete "
            f"source=teams_power_full reason={ex}"
        )
        ratings_df = rebuild_ratings_df(project_root, logger=logger)
        mapped_bracket_df = map_bracket_teams_to_ratings(bracket_df, ratings_df, logger=logger)
    logger.info(
        "BRACKET validate all_teams_mapped=true "
        f"mapped_teams={mapped_bracket_df['team'].nunique()}"
    )
    matchup_cache = build_bracket_matchup_cache(
        bracket_df=mapped_bracket_df,
        ratings_df=ratings_df,
        logger=logger,
    )

    sim_df = simulate_bracket(
        bracket_df=mapped_bracket_df,
        ratings_df=ratings_df,
        n_simulations=n_simulations,
        cache=matchup_cache,
    )
    matchup_probs_df = build_bracket_matchup_probs(
        bracket_df=mapped_bracket_df,
        ratings_df=ratings_df,
        cache=matchup_cache,
    )
    most_likely_df = build_most_likely_bracket(
        bracket_df=mapped_bracket_df,
        ratings_df=ratings_df,
        cache=matchup_cache,
        sim_df=sim_df,
    )
    pool_df = build_pool_bracket(
        bracket_df=mapped_bracket_df,
        ratings_df=ratings_df,
        mode="pool_ev",
        cache=matchup_cache,
        sim_df=sim_df,
        pool_size=pool_size,
    )

    sim_path = outputs_dir / "bracket_predictions.csv"
    matchup_probs_path = outputs_dir / "bracket_matchup_probs.csv"
    most_likely_path = outputs_dir / "most_likely_bracket.csv"
    pool_path = outputs_dir / "pool_bracket.csv"
    sim_df.to_csv(sim_path, index=False)
    matchup_probs_df.to_csv(matchup_probs_path, index=False)
    most_likely_df.to_csv(most_likely_path, index=False)
    pool_df.to_csv(pool_path, index=False)

    logger.info(
        "BRACKET outputs_written "
        f"predictions_path={sim_path} matchup_probs_path={matchup_probs_path} "
        f"most_likely_path={most_likely_path} pool_path={pool_path}"
    )
    return sim_df, matchup_probs_df, most_likely_df, pool_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default="/Users/breezy/Documents/2026.pdf",
        help="Path to the tournament bracket PDF.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
        help="Project root containing outputs/teams_power_full.xlsx.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=20000,
        help="Number of Monte Carlo simulations to run.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Bracket pool size used for pool EV optimization.",
    )
    args = parser.parse_args()

    logger = build_logger()
    run_bracket_predictions(
        pdf_path=args.pdf,
        project_root=args.project_root,
        n_simulations=args.simulations,
        pool_size=args.pool_size,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
