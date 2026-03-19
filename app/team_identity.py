from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.runtime import DATA_DIR, OUTPUTS_DIR
else:
    from .runtime import DATA_DIR, OUTPUTS_DIR


TEAM_ALIASES_PATH = DATA_DIR / "team_aliases.json"
TEAM_ID_MAP_PATH = OUTPUTS_DIR / "team_id_map.json"
FUZZY_MATCH_CUTOFF = 0.88
AUTO_ALIAS_CUTOFF = 0.95
INSTITUTION_PREFIXES = (
    "the ",
    "university of ",
    "college of ",
)
INSTITUTION_SUFFIXES = (
    " university",
    " college",
)


def _safe_team_id(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    if text.lower() in {"", "nan", "none"} or not text.isdigit():
        return ""
    return text


def normalize_team_name(name: str) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("&", " and ")
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_team_aliases(raw_aliases: dict) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in raw_aliases.items():
        key = normalize_team_name(raw_key)
        value = normalize_team_name(raw_value)
        if key and value:
            normalized[key] = value
    return normalized


def load_team_aliases(path: str | Path = TEAM_ALIASES_PATH) -> dict[str, str]:
    alias_path = Path(path)
    if not alias_path.exists():
        return {}
    try:
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return _normalize_team_aliases(payload)


def save_team_aliases(aliases: dict[str, str], path: str | Path = TEAM_ALIASES_PATH) -> Path:
    alias_path = Path(path)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    clean = dict(sorted(_normalize_team_aliases(aliases).items()))
    alias_path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return alias_path


def load_team_id_map(path: str | Path = TEAM_ID_MAP_PATH) -> dict[str, str]:
    team_map_path = Path(path)
    if not team_map_path.exists():
        return {}
    try:
        payload = json.loads(team_map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    clean: dict[str, str] = {}
    for raw_name, raw_team_id in payload.items():
        key = normalize_team_name(raw_name)
        team_id = _safe_team_id(raw_team_id)
        if key and team_id:
            clean[key] = team_id
    return clean


def save_team_id_map(team_id_map: dict[str, str], path: str | Path = TEAM_ID_MAP_PATH) -> Path:
    team_map_path = Path(path)
    team_map_path.parent.mkdir(parents=True, exist_ok=True)
    clean = dict(load_team_id_map(team_map_path))
    for raw_key, raw_team_id in team_id_map.items():
        key = normalize_team_name(raw_key)
        team_id = _safe_team_id(raw_team_id)
        if key and team_id:
            clean[key] = team_id
    clean = {key: value for key, value in sorted(clean.items()) if key and value}
    team_map_path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return team_map_path


def apply_team_alias(name: str, aliases: dict[str, str] | None = None) -> str:
    normalized = normalize_team_name(name)
    if not normalized:
        return ""
    alias_map = aliases if aliases is not None else {}
    return alias_map.get(normalized, normalized)


def expand_team_name_variants(name: str) -> list[str]:
    raw_name = str(name or "").strip()
    if not raw_name:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add(raw_value: str) -> None:
        normalized = normalize_team_name(raw_value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(normalized)

    base = normalize_team_name(raw_name)
    groups = [normalize_team_name(group) for group in re.findall(r"\(([^)]*)\)", raw_name)]

    for group in groups:
        if group:
            add(f"{base} {group}")
    add(raw_name)

    for variant in list(variants):
        for prefix in INSTITUTION_PREFIXES:
            if variant.startswith(prefix):
                add(variant[len(prefix):])
        for suffix in INSTITUTION_SUFFIXES:
            if variant.endswith(suffix):
                add(variant[: -len(suffix)])

    return variants


def build_team_id_map(
    *,
    aliases: dict[str, str] | None = None,
    existing_map: dict[str, str] | None = None,
    team_meta: dict | None = None,
    games_df: pd.DataFrame | None = None,
    ratings_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    alias_map = aliases if aliases is not None else {}
    key_to_ids: dict[str, set[str]] = {}

    def record(name: str, team_id) -> None:
        safe_team_id = _safe_team_id(team_id)
        if not safe_team_id:
            return
        for variant in expand_team_name_variants(name):
            key_to_ids.setdefault(variant, set()).add(safe_team_id)
            aliased = apply_team_alias(variant, alias_map)
            if aliased:
                key_to_ids.setdefault(aliased, set()).add(safe_team_id)

    for raw_name, raw_team_id in (existing_map or {}).items():
        record(str(raw_name), raw_team_id)

    for team_id, rec in (team_meta or {}).items():
        if not isinstance(rec, dict):
            continue
        for raw_name in (
            rec.get("nameShort", ""),
            rec.get("nameFull", ""),
            str(rec.get("seoname", "")).replace("-", " "),
            rec.get("name6Char", ""),
        ):
            record(str(raw_name), team_id)

    if games_df is not None and not games_df.empty:
        for id_col, name_col in (
            ("team_id", "team"),
            ("opponent_team_id", "opponent"),
        ):
            if id_col not in games_df.columns or name_col not in games_df.columns:
                continue
            for raw_team_id, raw_name in zip(games_df[id_col], games_df[name_col]):
                record(str(raw_name), raw_team_id)

    if ratings_df is not None and not ratings_df.empty:
        team_col = None
        team_id_col = None
        for candidate in ("team", "Team"):
            if candidate in ratings_df.columns:
                team_col = candidate
                break
        for candidate in ("team_id", "teamId"):
            if candidate in ratings_df.columns:
                team_id_col = candidate
                break
        if team_col and team_id_col:
            for raw_team_id, raw_name in zip(ratings_df[team_id_col], ratings_df[team_col]):
                record(str(raw_name), raw_team_id)

    return {
        key: next(iter(team_ids))
        for key, team_ids in sorted(key_to_ids.items())
        if len(team_ids) == 1
    }


@dataclass(frozen=True)
class TeamMatchResult:
    team_id: str = ""
    matched_key: str = ""
    input_key: str = ""
    method: str = ""
    score: float = 0.0
    learned_alias: bool = False
    learned_team_id: bool = False

    @property
    def matched(self) -> bool:
        return bool(self.team_id)


def _remember_match(
    *,
    input_key: str,
    matched_key: str,
    team_id: str,
    method: str,
    score: float,
    team_id_map: dict[str, str],
    aliases: dict[str, str],
    remember: bool,
) -> TeamMatchResult:
    learned_alias = False
    learned_team_id = False

    if remember and input_key and team_id and input_key not in team_id_map:
        team_id_map[input_key] = team_id
        learned_team_id = True

    if (
        remember
        and input_key
        and matched_key
        and input_key != matched_key
        and input_key not in aliases
        and score >= AUTO_ALIAS_CUTOFF
    ):
        aliases[input_key] = matched_key
        learned_alias = True

    return TeamMatchResult(
        team_id=team_id,
        matched_key=matched_key,
        input_key=input_key,
        method=method,
        score=score,
        learned_alias=learned_alias,
        learned_team_id=learned_team_id,
    )


def resolve_team_match(
    name: str,
    team_id_map: dict[str, str],
    aliases: dict[str, str] | None = None,
    *,
    logger: logging.Logger | None = None,
    source: str = "",
    remember: bool = True,
    allow_fuzzy: bool = True,
) -> TeamMatchResult:
    alias_map = aliases if aliases is not None else {}
    variants = expand_team_name_variants(name)
    if not variants:
        if logger is not None:
            logger.warning("WARNING TEAM_MATCH_FAILED name=%r source=%s", name, source or "unknown")
        return TeamMatchResult()

    for variant in variants:
        canonical = apply_team_alias(variant, alias_map)
        team_id = _safe_team_id(team_id_map.get(canonical, ""))
        if team_id:
            method = "alias" if canonical != variant else "direct"
            return _remember_match(
                input_key=variant,
                matched_key=canonical,
                team_id=team_id,
                method=method,
                score=1.0,
                team_id_map=team_id_map,
                aliases=alias_map,
                remember=remember,
            )

    if not allow_fuzzy:
        if logger is not None:
            logger.warning("WARNING TEAM_MATCH_FAILED name=%r source=%s", name, source or "unknown")
        return TeamMatchResult()

    keys = list(team_id_map.keys())
    best_result = TeamMatchResult()
    best_variant = ""
    candidate_count = 0
    ambiguous = False

    for variant in variants:
        canonical = apply_team_alias(variant, alias_map)
        matches = get_close_matches(canonical, keys, n=3, cutoff=FUZZY_MATCH_CUTOFF)
        if not matches:
            continue

        scored_matches = [
            (
                SequenceMatcher(None, canonical, match_key).ratio(),
                match_key,
                _safe_team_id(team_id_map.get(match_key, "")),
            )
            for match_key in matches
        ]
        scored_matches = [entry for entry in scored_matches if entry[2]]
        if not scored_matches:
            continue

        scored_matches.sort(reverse=True)
        top_score, top_key, top_team_id = scored_matches[0]
        if not top_team_id:
            continue
        candidate_count += 1

        if len(scored_matches) > 1:
            second_score, _, second_team_id = scored_matches[1]
            if second_team_id and second_team_id != top_team_id and abs(top_score - second_score) < 0.03:
                ambiguous = True
                continue

        if (top_score > best_result.score) or (
            top_score == best_result.score and len(variant) > len(best_variant)
        ):
            best_variant = variant
            best_result = _remember_match(
                input_key=variant,
                matched_key=top_key,
                team_id=top_team_id,
                method="fuzzy",
                score=top_score,
                team_id_map=team_id_map,
                aliases=alias_map,
                remember=remember,
            )

    if best_result.matched and not ambiguous:
        return best_result

    if logger is not None:
        logger.warning(
            "WARNING TEAM_MATCH_FAILED name=%r source=%s variants=%s fuzzy_candidates=%s",
            name,
            source or "unknown",
            variants,
            candidate_count,
        )
    return TeamMatchResult()


def log_team_match_coverage(
    logger: logging.Logger,
    *,
    scope: str,
    matched: int,
    total: int,
) -> None:
    coverage = 100.0 if total <= 0 else (100.0 * float(matched) / float(total))
    logger.info(
        "TEAM_MATCH scope=%s coverage=%.1f%% matched=%s/%s",
        scope,
        coverage,
        matched,
        total,
    )
