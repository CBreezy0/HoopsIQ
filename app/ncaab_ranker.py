#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import io
import json
import logging
import math
import os
import pickle
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import team_identity as ti
    from app.runtime import DEFAULT_CONFIG_PATH, REPO_ROOT, load_public_config, resolve_project_root
else:
    from . import team_identity as ti
    from .runtime import DEFAULT_CONFIG_PATH, REPO_ROOT, load_public_config, resolve_project_root


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


import numpy as np
import pandas as pd
try:
    import requests
except Exception:
    requests = None
try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # type: ignore[override]
        def __init__(self, total=0, **kwargs):
            self.total = total
            self.n = 0
            self.format_dict = {"rate": None}

        def update(self, n=1):
            self.n += int(n)

        def refresh(self):
            return None

        def close(self):
            return None

        def set_postfix_str(self, _):
            return None

if requests is not None:
    REQUESTS_HTTP_ERROR = requests.exceptions.HTTPError
    REQUESTS_TIMEOUT = requests.exceptions.Timeout
    REQUESTS_CONNECTION = requests.exceptions.ConnectionError
    REQUESTS_REQUEST = requests.exceptions.RequestException
else:
    class _RequestsUnavailable(Exception):
        pass

    REQUESTS_HTTP_ERROR = _RequestsUnavailable
    REQUESTS_TIMEOUT = _RequestsUnavailable
    REQUESTS_CONNECTION = _RequestsUnavailable
    REQUESTS_REQUEST = _RequestsUnavailable

NY = ZoneInfo("America/New_York")
HTTP_TIMEOUT_CONNECT_S = 5.0
HTTP_TIMEOUT_READ_S = 30.0
HTTP_MAX_ATTEMPTS = 4
HTTP_RETRY_STATUS_CODES = {502, 503, 504}
HEARTBEAT_INTERVAL_SECONDS = 15.0
HEARTBEAT_STALL_SECONDS = 60.0
SEASON_START = date(2025, 11, 3)
DEFAULT_RECENCY_DECAY_FACTOR_DAYS = 30.0
DEFAULT_D1_WHITELIST_RELPATH = "data/d1_whitelist_2026.json"

_PUBLIC_CONFIG = load_public_config()
_BETTING_CFG = (
    _PUBLIC_CONFIG.get("betting", {})
    if isinstance(_PUBLIC_CONFIG.get("betting", {}), dict)
    else {}
)


def _norm_text(v) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def to_bool_flag(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


PLAYER_DEBUG = to_bool_flag(os.environ.get("NCAAB_PLAYER_DEBUG", False))
PLAYER_DEBUG_DUMP = to_bool_flag(os.environ.get("NCAAB_PLAYER_DEBUG_DUMP", False))
_PLAYER_DUMPED_ONCE = False


def generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = uuid.uuid4().hex[:8]
    return f"{ts}_{rand}"


def hash_config(config_dict: dict) -> str:
    try:
        payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        payload = str(config_dict)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _default_project_root() -> str:
    return str(REPO_ROOT)


def _default_config_path() -> str:
    return str(DEFAULT_CONFIG_PATH)


def _resolve_project_root_from_cfg(cfg_path: str | None = None) -> str:
    return str(resolve_project_root(cfg_path))


def write_run_manifest(
    run_id: str,
    config_hash: str,
    games_used_df,
    ratings_df,
    artifact_paths: list[str],
    out_dir: str,
) -> str:
    now_utc = datetime.now(timezone.utc)
    timestamp_utc = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

    games_used_rows = 0 if games_used_df is None else int(len(games_used_df))
    min_date = None
    max_date = None
    if games_used_df is not None and (not games_used_df.empty) and "game_date" in games_used_df.columns:
        dates = pd.to_datetime(games_used_df["game_date"], errors="coerce")
        if dates.notna().any():
            min_date = dates.min().date().isoformat()
            max_date = dates.max().date().isoformat()

    ratings_rows = 0 if ratings_df is None else int(len(ratings_df))

    artifacts = {}
    for path in artifact_paths:
        key = os.path.basename(path)
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            artifacts[key] = {
                "mtime_utc": mtime.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "size_bytes": int(os.path.getsize(path)),
            }
        else:
            artifacts[key] = {"mtime_utc": None, "size_bytes": None}

    manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "config_hash": config_hash,
        "games_used": {"rows": games_used_rows, "min_date": min_date, "max_date": max_date},
        "ratings": {"rows": ratings_rows},
        "artifacts": artifacts,
    }

    project_root = os.path.dirname(out_dir)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if commit:
            manifest["git_commit"] = commit
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            manifest["git_dirty"] = bool(status)
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_player_backfill_state(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, "player_backfill_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_player_backfill_state(
    out_dir: str, season: int, through_date: date | str, run_id: str
) -> str:
    through_str = (
        through_date.isoformat()
        if isinstance(through_date, date)
        else str(through_date or "").strip()
    )
    state = {
        "season": int(season),
        "through_date": through_str,
        "run_id": str(run_id or "").strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "player_backfill_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return path


def is_d1_division_value(division, division_name=None) -> bool:
    vals = [division, division_name]
    for v in vals:
        if v is None:
            continue
        if isinstance(v, (int, float)) and int(v) == 1:
            return True
        s = _norm_text(v)
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        if s in {
            "1",
            "i",
            "d1",
            "di",
            "division 1",
            "division i",
            "ncaa division 1",
            "ncaa division i",
        }:
            return True
    return False


def normalize_team_name(name: str) -> str:
    return ti.normalize_team_name(name)


def normalize_name(s: str) -> str:
    return normalize_team_name(s)


def _normalize_whitelist_match_name(raw_name: str) -> str:
    text = str(raw_name or "").strip()
    if not text:
        return ""
    text = _norm_text(text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("&", " and ")
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.replace(".", "")
    text = re.sub(r"[^\w\s-]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return apply_name_alias(text)


TEAM_NAME_ALIASES = ti.load_team_aliases()


def apply_name_alias(norm_name: str) -> str:
    return ti.apply_team_alias(norm_name, aliases=TEAM_NAME_ALIASES)


def normalize_player_name(s: str) -> str:
    s = _norm_text(s)
    s = s.replace("&", " and ")
    s = s.replace("’", "")
    s = s.replace("'", "")
    s = s.replace(".", "")
    s = re.sub(r"[^\w\s-]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _team_name_key(raw_name) -> str:
    return apply_name_alias(normalize_name(raw_name))


PLAYER_STATUS_WEIGHTS = {
    "available": 0.0,
    "probable": 0.25,
    "questionable": 0.5,
    "doubtful": 0.75,
    "out": 1.0,
}

PLAYER_STATUS_PRIORITY = {
    "available": 0,
    "probable": 1,
    "questionable": 2,
    "doubtful": 3,
    "out": 4,
}

ROTOWIRE_CBB_INJURY_URL = "https://www.rotowire.com/cbasketball/tables/injury-report.php"
ESPN_CBB_INJURY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/injuries"
)
ESPN_CBB_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)


def _safe_team_id(x) -> str:
    if x is None or pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        if float(x).is_integer():
            return str(int(x))
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def canonical_team_id(raw) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    if not s.isdigit():
        return None
    return s


def _upsert_team_meta(team_meta: dict, team_id: str, team_obj: dict | None, source: str, division=None, division_name=None):
    tid = _safe_team_id(team_id)
    if not tid:
        return
    team_obj = team_obj if isinstance(team_obj, dict) else {}
    rec = team_meta.get(tid, {
        "teamId": tid,
        "nameShort": "",
        "nameFull": "",
        "seoname": "",
        "name6Char": "",
        "division": "",
        "divisionName": "",
        "source": source,
    })
    names = team_obj.get("names", {}) if isinstance(team_obj.get("names", {}), dict) else {}
    name_short = (
        team_obj.get("nameShort")
        or names.get("short")
        or team_obj.get("teamName")
        or team_obj.get("name6Char")
        or team_obj.get("name")
    )
    name_full = (
        team_obj.get("nameFull")
        or names.get("full")
        or team_obj.get("teamName")
        or team_obj.get("name")
    )
    seo = team_obj.get("seoname") or team_obj.get("seoName")
    name6 = team_obj.get("name6Char")
    if name_short and not rec.get("nameShort"):
        rec["nameShort"] = str(name_short)
    if name_full and not rec.get("nameFull"):
        rec["nameFull"] = str(name_full)
    if seo and not rec.get("seoname"):
        rec["seoname"] = str(seo)
    if name6 and not rec.get("name6Char"):
        rec["name6Char"] = str(name6)
    if division is not None and str(division).strip():
        rec["division"] = str(division)
    if division_name is not None and str(division_name).strip():
        rec["divisionName"] = str(division_name)
    if source:
        rec["source"] = source
    team_meta[tid] = rec


def extract_team_records(payload, source: str) -> list[dict]:
    out = []

    def walk(obj):
        if isinstance(obj, dict):
            has_team_shape = (
                ("teamId" in obj)
                or any(k in obj for k in ["nameShort", "nameFull", "seoname", "name6Char", "teamName"])
            )
            if has_team_shape:
                tid = _safe_team_id(obj.get("teamId") or obj.get("id"))
                if tid:
                    out.append({
                        "teamId": tid,
                        "nameShort": obj.get("nameShort") or obj.get("teamName") or obj.get("name6Char") or "",
                        "nameFull": obj.get("nameFull") or obj.get("teamName") or "",
                        "seoname": obj.get("seoname") or obj.get("seoName") or "",
                        "name6Char": obj.get("name6Char") or "",
                        "division": obj.get("division"),
                        "divisionName": obj.get("divisionName"),
                        "source": source,
                    })
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(payload)
    dedup = {}
    for r in out:
        tid = r["teamId"]
        if tid not in dedup:
            dedup[tid] = r
        else:
            cur = dedup[tid]
            for k in ["nameShort", "nameFull", "seoname", "name6Char", "division", "divisionName"]:
                if not cur.get(k) and r.get(k):
                    cur[k] = r[k]
    return list(dedup.values())


def extract_boxscore_team_records_strict(
    bs: dict, game_id: str, logger: logging.Logger, stats: dict | None = None
) -> list[dict]:
    teams = bs.get("teams", []) or []
    out = []
    for t in teams:
        raw_tid = t.get("teamId", None)
        canonical = canonical_team_id(raw_tid)
        if canonical is None:
            if stats is not None:
                stats["invalid_team_ids"] = stats.get("invalid_team_ids", 0) + 1
            raw_s = "" if raw_tid is None else str(raw_tid).strip()
            reason = "empty" if raw_s == "" else "non-numeric"
            logger.warning(
                f"TEAM_DIR skip {reason} teamId game_id={game_id} teamId={raw_s!r}"
            )
            continue
        out.append({
            "teamId": canonical,
            "nameShort": t.get("nameShort") or t.get("teamName") or t.get("name6Char") or "",
            "nameFull": t.get("nameFull") or t.get("teamName") or "",
            "seoname": t.get("seoname") or t.get("seoName") or "",
            "name6Char": t.get("name6Char") or "",
            "source": f"boxscore:{game_id}",
        })
    dedup = {}
    for r in out:
        tid = r["teamId"]
        if tid not in dedup:
            dedup[tid] = r
        else:
            cur = dedup[tid]
            for k in ["nameShort", "nameFull", "seoname", "name6Char"]:
                if not cur.get(k) and r.get(k):
                    cur[k] = r[k]
    return list(dedup.values())


def _safe_player_id(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        if float(v).is_integer():
            return str(int(v))
    s = str(v).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return "" if s.lower() in {"", "nan", "none"} else s


def _parse_minutes_value(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            x = float(v)
            return x if np.isfinite(x) and x >= 0 else 0.0
        except Exception:
            return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    if ":" in s:
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", s)
        if m:
            mm = float(m.group(1))
            ss = float(m.group(2))
            return mm + (ss / 60.0)
    try:
        x = float(s)
        return x if np.isfinite(x) and x >= 0 else 0.0
    except Exception:
        return 0.0


def _to_stat_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default
    s = str(v).strip()
    if not s:
        return default
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            x = float(m.group(0))
            return x if np.isfinite(x) else default
        except Exception:
            return default
    return default


def _find_first_stat_value(obj, keys: set[str]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in keys:
                return v
        for v in obj.values():
            out = _find_first_stat_value(v, keys)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for it in obj:
            out = _find_first_stat_value(it, keys)
            if out is not None:
                return out
    return None


def _extract_made_attempted(
    stat_obj: dict,
    made_keys: set[str],
    att_keys: set[str],
    combo_keys: set[str],
) -> tuple[float, float]:
    made = _find_first_stat_value(stat_obj, made_keys)
    att = _find_first_stat_value(stat_obj, att_keys)
    made_f = _to_stat_float(made, default=np.nan)
    att_f = _to_stat_float(att, default=np.nan)
    if np.isfinite(made_f) and np.isfinite(att_f):
        return float(made_f), float(att_f)

    combo = _find_first_stat_value(stat_obj, combo_keys)
    if combo is not None:
        s = str(combo)
        m = re.search(r"(\d+)\s*-\s*(\d+)", s)
        if m:
            return float(m.group(1)), float(m.group(2))
    return float(np.nan_to_num(made_f, nan=0.0)), float(np.nan_to_num(att_f, nan=0.0))


def parse_player_rows_from_boxscore(
    boxscore_json: dict,
    game_meta: dict,
    logger: logging.Logger | None = None,
) -> list[dict]:
    active_logger = logger or logging.getLogger("ncaab_ranker")
    if not isinstance(boxscore_json, dict):
        active_logger.info("PLAYER_EXTRACT_DONE game_id=unknown rows_out=0")
        return []

    game_id = str(
        game_meta.get("game_id")
        or boxscore_json.get("contestId")
        or boxscore_json.get("contestID")
        or boxscore_json.get("gameID")
        or ""
    ).strip()
    game_date = str(game_meta.get("game_date") or "").strip()

    teams_list = boxscore_json.get("teams", []) or []
    team_lookup: dict[str, dict] = {}
    team_ids_ordered: list[str] = []
    for t in teams_list:
        if not isinstance(t, dict):
            continue
        tid = canonical_team_id(t.get("teamId")) or _safe_team_id(t.get("teamId"))
        if not tid:
            continue
        tname = (
            t.get("nameShort")
            or t.get("teamName")
            or t.get("nameFull")
            or t.get("name6Char")
            or ""
        )
        team_lookup[tid] = {
            "name": str(tname or "").strip(),
            "is_home": bool(t.get("isHome")),
        }
        team_ids_ordered.append(tid)

    home_team_id = canonical_team_id(game_meta.get("home_team_id")) or _safe_team_id(
        game_meta.get("home_team_id")
    )
    away_team_id = canonical_team_id(game_meta.get("away_team_id")) or _safe_team_id(
        game_meta.get("away_team_id")
    )
    home_name = str(game_meta.get("home_name") or "").strip()
    away_name = str(game_meta.get("away_name") or "").strip()

    team_boxscores = boxscore_json.get("teamBoxscore", []) or []
    if isinstance(team_boxscores, dict):
        team_boxscores = [team_boxscores]

    total_team_boxes_seen = 0
    total_player_entries_seen = 0
    skipped_missing_player_name = 0
    skipped_missing_player_id_and_name = 0
    skipped_unknown_player_shape = 0
    out: list[dict] = []
    for team_box in team_boxscores:
        total_team_boxes_seen += 1
        if not isinstance(team_box, dict):
            continue
        tid = canonical_team_id(team_box.get("teamId")) or _safe_team_id(team_box.get("teamId"))
        if not tid:
            nested_team = team_box.get("team", {})
            if isinstance(nested_team, dict):
                tid = canonical_team_id(nested_team.get("teamId")) or _safe_team_id(
                    nested_team.get("teamId")
                )
        if not tid:
            continue

        if home_team_id and away_team_id:
            if tid == home_team_id:
                opp_tid = away_team_id
                team_name = home_name or team_lookup.get(tid, {}).get("name", "")
                opp_name = away_name or team_lookup.get(opp_tid, {}).get("name", "")
            elif tid == away_team_id:
                opp_tid = home_team_id
                team_name = away_name or team_lookup.get(tid, {}).get("name", "")
                opp_name = home_name or team_lookup.get(opp_tid, {}).get("name", "")
            else:
                others = [x for x in team_ids_ordered if x != tid]
                opp_tid = others[0] if others else ""
                team_name = team_lookup.get(tid, {}).get("name", "")
                opp_name = team_lookup.get(opp_tid, {}).get("name", "")
        else:
            others = [x for x in team_ids_ordered if x != tid]
            opp_tid = others[0] if others else ""
            team_name = team_lookup.get(tid, {}).get("name", "")
            opp_name = team_lookup.get(opp_tid, {}).get("name", "")

        player_stats = (
            team_box.get("playerStats")
            or team_box.get("players")
            or team_box.get("playerBoxscore")
            or []
        )
        if isinstance(player_stats, dict):
            player_stats = player_stats.get("players", []) or []
        active_logger.info(
            f"PLAYER_PARSE_TEAM game_id={game_id or 'unknown'} team_id={tid} "
            f"player_stats_type={type(player_stats).__name__} "
            f"player_stats_len={len(player_stats) if isinstance(player_stats, list) else 'n/a'}"
        )
        if not isinstance(player_stats, list):
            continue
        total_player_entries_seen += len(player_stats)

        for p in player_stats:
            if not isinstance(p, dict):
                skipped_unknown_player_shape += 1
                continue
            p_nested = p.get("player", {})
            if not isinstance(p_nested, dict):
                p_nested = {}

            player_name = (
                p.get("playerName")
                or p.get("name")
                or p.get("nameShort")
                or p.get("nameFull")
                or p_nested.get("name")
                or p_nested.get("nameShort")
                or p_nested.get("nameFull")
                or p_nested.get("fullName")
                or ""
            )
            player_name = str(player_name or "").strip()
            if not player_name:
                first_name = p.get("firstName") or p_nested.get("firstName") or ""
                last_name = p.get("lastName") or p_nested.get("lastName") or ""
                first_name = str(first_name or "").strip()
                last_name = str(last_name or "").strip()
                if first_name and last_name:
                    player_name = f"{first_name} {last_name}"
                elif last_name:
                    player_name = last_name
                elif first_name:
                    player_name = first_name
                player_name = str(player_name or "").strip()
            player_id_candidate = _safe_player_id(
                p.get("playerId")
                or p.get("athleteId")
                or p.get("personId")
                or p.get("id")
                or p_nested.get("playerId")
                or p_nested.get("athleteId")
                or p_nested.get("personId")
                or p_nested.get("id")
            )
            if not player_name:
                skipped_missing_player_name += 1
                if not player_id_candidate:
                    skipped_missing_player_id_and_name += 1
                active_logger.info(
                    f"PLAYER_SKIP_MISSING_NAME game_id={game_id or 'unknown'} "
                    f"team_id={tid} player_keys={list(p.keys())[:40]}"
                )
                continue

            player_id = player_id_candidate
            player_key = player_id if player_id else f"{normalize_player_name(player_name)}|{tid}"

            p_stat_obj = p.get("stats", p)
            if not isinstance(p_stat_obj, dict):
                p_stat_obj = p

            fgm, fga = _extract_made_attempted(
                p_stat_obj,
                {"fieldGoalsMade", "fgm", "FGM"},
                {"fieldGoalsAttempted", "fga", "FGA"},
                {"fieldGoals", "FG", "fg"},
            )
            ftm, fta = _extract_made_attempted(
                p_stat_obj,
                {"freeThrowsMade", "ftm", "FTM"},
                {"freeThrowsAttempted", "fta", "FTA"},
                {"freeThrows", "FT", "ft"},
            )
            tpm, tpa = _extract_made_attempted(
                p_stat_obj,
                {"threePointFieldGoalsMade", "threePointsMade", "tpm", "3PM"},
                {"threePointFieldGoalsAttempted", "threePointsAttempted", "tpa", "3PA"},
                {"threePoints", "3PT", "threePointFieldGoals"},
            )

            minutes_raw = (
                _find_first_stat_value(
                    p_stat_obj,
                    {"minutes", "min", "MIN", "minutesPlayed", "timePlayed"},
                )
                or p.get("minutes")
            )

            row = {
                "game_id": game_id,
                "game_date": game_date,
                "team_id": _safe_team_id(tid),
                "opponent_team_id": _safe_team_id(opp_tid),
                "team": str(team_name or "").strip(),
                "opponent": str(opp_name or "").strip(),
                "player_id": player_id,
                "player_key": player_key,
                "player_name": player_name,
                "minutes": _parse_minutes_value(minutes_raw),
                "points": _to_stat_float(
                    _find_first_stat_value(p_stat_obj, {"points", "pts", "PTS"})
                ),
                "fgm": fgm,
                "fga": fga,
                "ftm": ftm,
                "fta": fta,
                "tpm": tpm,
                "tpa": tpa,
                "orb": _to_stat_float(
                    _find_first_stat_value(
                        p_stat_obj,
                        {"offensiveRebounds", "orb", "ORB", "offReb"},
                    )
                ),
                "drb": _to_stat_float(
                    _find_first_stat_value(
                        p_stat_obj,
                        {"defensiveRebounds", "drb", "DRB", "defReb"},
                    )
                ),
                "trb": _to_stat_float(
                    _find_first_stat_value(
                        p_stat_obj,
                        {"totalRebounds", "rebounds", "trb", "TRB", "REB"},
                    )
                ),
                "ast": _to_stat_float(
                    _find_first_stat_value(p_stat_obj, {"assists", "ast", "AST"})
                ),
                "tov": _to_stat_float(
                    _find_first_stat_value(
                        p_stat_obj, {"turnovers", "tov", "TO", "TOV", "turnover"}
                    )
                ),
                "stl": _to_stat_float(
                    _find_first_stat_value(p_stat_obj, {"steals", "stl", "STL"})
                ),
                "blk": _to_stat_float(
                    _find_first_stat_value(p_stat_obj, {"blocks", "blk", "BLK"})
                ),
                "pf": _to_stat_float(
                    _find_first_stat_value(
                        p_stat_obj,
                        {"personalFouls", "fouls", "pf", "PF"},
                    )
                ),
                "starter": bool(
                    p.get("starter")
                    or p.get("isStarter")
                    or p_stat_obj.get("starter")
                    or p_stat_obj.get("isStarter")
                ),
            }
            out.append(row)
    active_logger.info(
        f"PLAYER_EXTRACT_DONE game_id={game_id or 'unknown'} rows_out={len(out)}"
    )
    if PLAYER_DEBUG:
        active_logger.info(
            f"PLAYER_PARSE_SUMMARY game_id={game_id or 'unknown'} "
            f"team_boxes={total_team_boxes_seen} "
            f"player_entries={total_player_entries_seen} out_rows={len(out)} "
            f"skipped_missing_name={skipped_missing_player_name} "
            f"skipped_missing_player_id_and_name={skipped_missing_player_id_and_name} "
            f"skipped_shape={skipped_unknown_player_shape}"
        )
    return out


def bootstrap_d1_team_ids_from_api(client: SourceClient, season: int, logger: logging.Logger) -> tuple[set[str], dict]:
    d1_ids: set[str] = set()
    team_meta: dict = {}
    candidate_paths = [
        f"/teams/basketball-men/d1/{season}",
        "/teams/basketball-men/d1",
        f"/teams/basketball-men/{season}",
        "/teams/basketball-men",
    ]
    for path in candidate_paths:
        try:
            payload = client.get_json(path)
        except Exception:
            continue
        recs = extract_team_records(payload, source=f"endpoint:{path}")
        if not recs:
            continue
        for r in recs:
            tid = r["teamId"]
            _upsert_team_meta(
                team_meta,
                tid,
                {
                    "nameShort": r.get("nameShort"),
                    "nameFull": r.get("nameFull"),
                    "seoname": r.get("seoname"),
                    "name6Char": r.get("name6Char"),
                },
                r.get("source", f"endpoint:{path}"),
                r.get("division"),
                r.get("divisionName"),
            )
            is_d1 = is_d1_division_value(r.get("division"), r.get("divisionName")) or ("/d1" in path)
            if is_d1:
                d1_ids.add(tid)
        logger.info(f"D1_BOOTSTRAP endpoint={path} teams={len(recs)} d1_ids_now={len(d1_ids)}")
    return d1_ids, team_meta


def write_d1_cache(out_dir: str, d1_team_ids: set[str], team_meta: dict, logger: logging.Logger):
    ids_path = os.path.join(out_dir, "d1_team_ids.json")
    debug_path = os.path.join(out_dir, "d1_team_ids_debug.csv")
    os.makedirs(out_dir, exist_ok=True)
    ids_sorted = sorted({_safe_team_id(x) for x in d1_team_ids if _safe_team_id(x)})
    with open(ids_path, "w") as f:
        json.dump(ids_sorted, f, indent=2)

    rows = []
    for tid in ids_sorted:
        rec = team_meta.get(tid, {})
        rows.append({
            "teamId": tid,
            "nameShort": rec.get("nameShort", ""),
            "nameFull": rec.get("nameFull", ""),
            "seoname": rec.get("seoname", ""),
            "division": rec.get("division", ""),
            "divisionName": rec.get("divisionName", ""),
            "source": rec.get("source", ""),
        })
    pd.DataFrame(rows, columns=[
        "teamId", "nameShort", "nameFull", "seoname", "division", "divisionName", "source"
    ]).to_csv(debug_path, index=False)
    logger.info(f"D1_CACHE wrote ids={len(ids_sorted)} ids_path={ids_path} debug_path={debug_path}")


def load_d1_team_ids_cache(out_dir: str) -> set[str]:
    ids_path = os.path.join(out_dir, "d1_team_ids.json")
    if not os.path.exists(ids_path):
        return set()
    try:
        with open(ids_path, "r") as f:
            arr = json.load(f)
    except Exception:
        return set()
    if not isinstance(arr, list):
        return set()
    return {_safe_team_id(x) for x in arr if _safe_team_id(x)}


def load_team_directory_cache(out_dir: str) -> dict:
    path = os.path.join(out_dir, "team_directory.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        tid = canonical_team_id(k)
        if tid is None:
            continue
        rec = v if isinstance(v, dict) else {}
        out[tid] = {
            "teamId": tid,
            "nameShort": str(rec.get("nameShort", "") or ""),
            "nameFull": str(rec.get("nameFull", "") or ""),
            "seoname": str(rec.get("seoname", "") or ""),
            "name6Char": str(rec.get("name6Char", "") or ""),
            "source": str(rec.get("source", "") or ""),
        }
    return out


def extract_scoreboard_game_id(game_obj: dict) -> str:
    return str(
        game_obj.get("contestId")
        or game_obj.get("contestID")
        or game_obj.get("gameID")
        or game_obj.get("id")
        or ""
    ).strip()


def unwrap_scoreboard_game(item) -> dict:
    if not isinstance(item, dict):
        return {}
    game = item.get("game", item)
    return game if isinstance(game, dict) else {}


@dataclass
class HeartbeatState:
    logger: logging.Logger
    phase: str
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS
    stall_seconds: float = HEARTBEAT_STALL_SECONDS
    start_monotonic: float = field(default_factory=time.monotonic)
    last_heartbeat_monotonic: float = field(default_factory=time.monotonic)
    day: str = ""
    processed: int = 0
    fetched: int = 0
    failures: int = 0
    last_url: str = ""

    def touch(
        self,
        *,
        day: date | None = None,
        last_url: str | None = None,
        processed_inc: int = 0,
        fetched_inc: int = 0,
        failures_inc: int = 0,
        force: bool = False,
    ):
        now = time.monotonic()
        if day is not None:
            self.day = str(day)
        if last_url:
            self.last_url = last_url
        self.processed += int(processed_inc)
        self.fetched += int(fetched_inc)
        self.failures += int(failures_inc)

        if (now - self.last_heartbeat_monotonic) >= self.stall_seconds:
            elapsed = now - self.start_monotonic
            stalled = now - self.last_heartbeat_monotonic
            self.logger.warning(
                f"WATCHDOG phase={self.phase} stalled_seconds={stalled:.1f} "
                f"day={self.day or 'n/a'} last_url={self.last_url or 'n/a'} "
                f"processed={self.processed} fetched={self.fetched} failures={self.failures} "
                f"elapsed={elapsed:.1f}s"
            )
            self.last_heartbeat_monotonic = now
            return

        if force or (now - self.last_heartbeat_monotonic) >= self.interval_seconds:
            elapsed = now - self.start_monotonic
            self.logger.info(
                f"HEARTBEAT phase={self.phase} day={self.day or 'n/a'} "
                f"processed={self.processed} fetched={self.fetched} failures={self.failures} "
                f"elapsed={elapsed:.1f}s"
            )
            self.last_heartbeat_monotonic = now


def get_json_with_retry(
    client: "SourceClient",
    path: str,
    logger: logging.Logger,
    phase: str,
    item_key: str,
    heartbeat: HeartbeatState | None = None,
    max_attempts: int = HTTP_MAX_ATTEMPTS,
) -> dict | None:
    last_err: Exception | None = None
    status_code = None

    for attempt in range(1, max_attempts + 1):
        status_code = None
        if heartbeat is not None:
            heartbeat.touch(last_url=path)
        try:
            return client.get_json(path)
        except REQUESTS_HTTP_ERROR as ex:
            status_code = ex.response.status_code if ex.response is not None else None
            retryable = status_code in HTTP_RETRY_STATUS_CODES
            last_err = ex
        except (REQUESTS_TIMEOUT, REQUESTS_CONNECTION) as ex:
            retryable = True
            last_err = ex
        except REQUESTS_REQUEST as ex:
            retryable = False
            last_err = ex
        except Exception as ex:
            retryable = False
            last_err = ex

        if not retryable or attempt >= max_attempts:
            break

        backoff = min(8.0, 0.5 * (2 ** (attempt - 1)))
        logger.info(
            f"RETRY phase={phase} item={item_key} attempt={attempt}/{max_attempts} "
            f"status={status_code} backoff_seconds={backoff:.2f} path={path}"
        )
        if heartbeat is not None:
            heartbeat.touch(force=True)
        time.sleep(backoff)

    if heartbeat is not None:
        heartbeat.touch(failures_inc=1, force=True)
    logger.warning(
        f"{phase} request failed after retries item={item_key} status={status_code} "
        f"path={path} error={type(last_err).__name__ if last_err else 'unknown'}: {last_err}"
    )
    return None


def fetch_scoreboard_with_retry(
    client: "SourceClient",
    day: date,
    logger: logging.Logger,
    context: str,
    heartbeat: HeartbeatState | None = None,
) -> dict | None:
    path = f"/scoreboard/basketball-men/d1/{day.year}/{day.month:02d}/{day.day:02d}"
    return get_json_with_retry(
        client=client,
        path=path,
        logger=logger,
        phase=context,
        item_key=str(day),
        heartbeat=heartbeat,
    )


def fetch_boxscore_with_retry(
    client: "SourceClient",
    game_id: str,
    logger: logging.Logger,
    context: str,
    heartbeat: HeartbeatState | None = None,
) -> dict | None:
    return get_json_with_retry(
        client=client,
        path=f"/game/{game_id}/boxscore",
        logger=logger,
        phase=context,
        item_key=game_id,
        heartbeat=heartbeat,
    )


def _games_long_columns() -> list[str]:
    return [
        "game_id",
        "game_date",
        "team",
        "opponent",
        "location",
        "team_id",
        "opponent_team_id",
        "pts_for",
        "pts_against",
        "fga",
        "fta",
        "orb",
        "to",
        "opp_fga",
        "opp_fta",
        "opp_orb",
        "opp_to",
        "team_conf",
        "opp_conf",
        "is_d1_team",
        "is_d1_opponent",
    ]


def _empty_games_long_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_games_long_columns())


def _client_base_url(client: "SourceClient" | None) -> str:
    return str(getattr(client, "base_url", "") or "").strip()


def _estimate_scoreboard_possessions(team_score: int, opponent_score: int) -> int:
    avg_score = (float(team_score) + float(opponent_score)) / 2.0
    return int(round(float(np.clip(avg_score, 55.0, 80.0))))


def fetch_espn_games_for_day(
    day: date,
    logger: logging.Logger,
    d1_team_ids: set[str] | None = None,
    d1_team_meta: dict | None = None,
) -> pd.DataFrame:
    if requests is None:
        logger.warning(
            f"GAMES ESPN unavailable day={day} reason=requests_unavailable"
        )
        return _empty_games_long_df()

    try:
        response = requests.get(
            ESPN_CBB_SCOREBOARD_URL,
            params={
                "dates": day.strftime("%Y%m%d"),
                "groups": "50",
                "limit": "400",
            },
            timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"unexpected payload type={type(payload).__name__}"
            )
    except Exception as ex:
        logger.warning(
            f"GAMES ESPN fetch_failed day={day} error={type(ex).__name__}: {ex}"
        )
        return _empty_games_long_df()

    rows: list[dict] = []
    d1_team_ids = d1_team_ids or set()

    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions", []) or []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        status = competition.get("status") if isinstance(competition.get("status"), dict) else {}
        status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
        if not bool(status_type.get("completed")):
            continue

        competitors = competition.get("competitors", []) or []
        if not isinstance(competitors, list) or len(competitors) < 2:
            continue

        home = None
        away = None
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            side = str(comp.get("homeAway", "")).strip().lower()
            if side == "home" and home is None:
                home = comp
            elif side == "away" and away is None:
                away = comp
        if home is None or away is None:
            ordered = [comp for comp in competitors if isinstance(comp, dict)]
            if len(ordered) >= 2:
                away = away or ordered[0]
                home = home or ordered[1]
        if not isinstance(home, dict) or not isinstance(away, dict):
            continue

        home_team = home.get("team", {}) if isinstance(home.get("team"), dict) else {}
        away_team = away.get("team", {}) if isinstance(away.get("team"), dict) else {}

        home_name = str(
            home_team.get("displayName")
            or home_team.get("shortDisplayName")
            or home_team.get("location")
            or home_team.get("name")
            or ""
        ).strip()
        away_name = str(
            away_team.get("displayName")
            or away_team.get("shortDisplayName")
            or away_team.get("location")
            or away_team.get("name")
            or ""
        ).strip()
        if not home_name or not away_name:
            continue

        home_score = safe_int(home.get("score"), default=-1)
        away_score = safe_int(away.get("score"), default=-1)
        if home_score < 0 or away_score < 0:
            continue

        home_team_id = (
            canonical_team_id(home_team.get("id") or home.get("id"))
            or _safe_team_id(home_team.get("id") or home.get("id"))
        )
        away_team_id = (
            canonical_team_id(away_team.get("id") or away.get("id"))
            or _safe_team_id(away_team.get("id") or away.get("id"))
        )
        neutral_site = bool(competition.get("neutralSite"))
        home_loc = "N" if neutral_site else "H"
        away_loc = "N" if neutral_site else "A"
        game_id = str(event.get("id") or competition.get("id") or "").strip()
        game_date = _scoreboard_game_date(competition or event, day)
        est_poss = _estimate_scoreboard_possessions(home_score, away_score)

        if d1_team_meta is not None:
            _upsert_team_meta(
                d1_team_meta,
                home_team_id,
                {
                    "teamId": home_team_id,
                    "nameShort": home_name,
                    "nameFull": home_name,
                    "seoname": home_team.get("abbreviation") or "",
                },
                source=f"espn_scoreboard:{game_id or game_date}",
            )
            _upsert_team_meta(
                d1_team_meta,
                away_team_id,
                {
                    "teamId": away_team_id,
                    "nameShort": away_name,
                    "nameFull": away_name,
                    "seoname": away_team.get("abbreviation") or "",
                },
                source=f"espn_scoreboard:{game_id or game_date}",
            )

        home_is_d1 = bool(home_team_id and home_team_id in d1_team_ids) if d1_team_ids else True
        away_is_d1 = bool(away_team_id and away_team_id in d1_team_ids) if d1_team_ids else True

        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "team_id": home_team_id,
                "opponent_team_id": away_team_id,
                "team": home_name,
                "opponent": away_name,
                "location": home_loc,
                "pts_for": home_score,
                "pts_against": away_score,
                "fga": est_poss,
                "fta": 0,
                "orb": 0,
                "to": 0,
                "opp_fga": est_poss,
                "opp_fta": 0,
                "opp_orb": 0,
                "opp_to": 0,
                "team_conf": "",
                "opp_conf": "",
                "is_d1_team": home_is_d1,
                "is_d1_opponent": away_is_d1,
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "team_id": away_team_id,
                "opponent_team_id": home_team_id,
                "team": away_name,
                "opponent": home_name,
                "location": away_loc,
                "pts_for": away_score,
                "pts_against": home_score,
                "fga": est_poss,
                "fta": 0,
                "orb": 0,
                "to": 0,
                "opp_fga": est_poss,
                "opp_fta": 0,
                "opp_orb": 0,
                "opp_to": 0,
                "team_conf": "",
                "opp_conf": "",
                "is_d1_team": away_is_d1,
                "is_d1_opponent": home_is_d1,
            }
        )

    out = pd.DataFrame(rows, columns=_games_long_columns()) if rows else _empty_games_long_df()
    logger.info(f"GAMES loaded source=espn rows={len(out)} day={day}")
    return out


def rebuild_full_season_games(
    logger: logging.Logger | None = None,
    start_date: date = SEASON_START,
    end_date: date | None = None,
    out_dir: str | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    project_root = _default_project_root()
    outputs_dir = out_dir or os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    hist_path = os.path.join(outputs_dir, "games_history.csv")
    final_day = end_date or datetime.now(tz=NY).date()

    if start_date > final_day:
        logger.warning(
            "GAMES full_season_rebuild skipped "
            f"reason=invalid_range start_date={start_date} end_date={final_day}"
        )
        return _empty_games_long_df()

    frames: list[pd.DataFrame] = []
    days_processed = 0
    for day_value in daterange(start_date, final_day):
        days_processed += 1
        try:
            day_df = fetch_espn_games_for_day(day_value, logger)
        except Exception as ex:
            logger.warning(
                "GAMES full_season_rebuild day_failed "
                f"day={day_value} error={type(ex).__name__}: {ex}"
            )
            continue
        if day_df is None or day_df.empty:
            continue
        frames.append(day_df)

    rebuilt = pd.concat(frames, ignore_index=True) if frames else _empty_games_long_df()
    if not rebuilt.empty:
        rebuilt = validate_games_df(rebuilt, logger)
        rebuilt = filter_strict_d1_games(
            rebuilt,
            logger,
            label="GAMES full_season_rebuild",
        )
        if "game_id" in rebuilt.columns and "team_id" in rebuilt.columns:
            with_ids = rebuilt["team_id"].astype(str).str.strip() != ""
            rebuilt_with_ids = rebuilt.loc[with_ids].drop_duplicates(
                subset=["game_id", "team_id"],
                keep="last",
            )
            rebuilt_without_ids = rebuilt.loc[~with_ids].drop_duplicates(
                subset=["game_id", "team"],
                keep="last",
            )
            rebuilt = pd.concat(
                [rebuilt_with_ids, rebuilt_without_ids],
                ignore_index=True,
            )
        else:
            rebuilt = rebuilt.drop_duplicates(
                subset=["game_id", "team", "opponent"],
                keep="last",
            )
        rebuilt = rebuilt.sort_values(
            ["game_date", "game_id", "team"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    if rebuilt.empty and os.path.exists(hist_path):
        logger.warning(
            "GAMES full_season_rebuild produced no rows; keeping existing history "
            f"path={os.path.abspath(hist_path)} days={days_processed}"
        )
        try:
            existing = pd.read_csv(hist_path)
            logger.info(
                f"GAMES full_season_rebuild rows={len(existing)} days={days_processed} "
                f"path={os.path.abspath(hist_path)} kept_existing=true"
            )
            return existing
        except Exception:
            pass

    rebuilt.to_csv(hist_path, index=False)
    logger.info(
        f"GAMES full_season_rebuild rows={len(rebuilt)} days={days_processed} "
        f"path={os.path.abspath(hist_path)}"
    )
    return rebuilt


def build_team_directory_full_season(
    client: "SourceClient",
    season: int,
    logger: logging.Logger,
    start_date: date,
    end_date: date,
    return_stats: bool = False,
) -> dict | tuple[dict, dict]:
    team_meta: dict = {}
    game_ids_seen: set[str] = set()
    fetched_game_ids: set[str] = set()
    days = 0
    boxscores = 0
    fetch_failures = 0
    skipped_invalid_ids = 0
    if not _client_base_url(client):
        logger.warning(
            "TEAM_DIR no NCAA_API_BASE_URL set; using ESPN scoreboard fallback"
        )
        for d in daterange(start_date, end_date):
            days += 1
            day_df = fetch_espn_games_for_day(d, logger)
            if day_df.empty:
                continue
            fetched_game_ids.update(
                str(x) for x in day_df["game_id"].dropna().astype(str).unique().tolist()
            )
            for id_col, name_col in [("team_id", "team"), ("opponent_team_id", "opponent")]:
                if id_col not in day_df.columns or name_col not in day_df.columns:
                    continue
                for _, row in day_df[[id_col, name_col, "game_id"]].drop_duplicates().iterrows():
                    tid = _safe_team_id(row.get(id_col))
                    team_name = str(row.get(name_col) or "").strip()
                    if not tid or not team_name:
                        continue
                    _upsert_team_meta(
                        team_meta,
                        tid,
                        {
                            "teamId": tid,
                            "nameShort": team_name,
                            "nameFull": team_name,
                        },
                        source=f"espn_scoreboard:{row.get('game_id') or d.isoformat()}",
                    )
        summary = {
            "total_teams": len(team_meta),
            "total_game_ids_fetched": len(fetched_game_ids),
            "total_failures": 0,
            "total_invalid_team_ids": 0,
        }
        logger.info(
            f"TEAM_DIR build completed season={season} days={days} games={len(fetched_game_ids)} "
            f"boxscores=0 teams={len(team_meta)} skipped_invalid_ids=0 fetch_failures=0 source=espn"
        )
        if return_stats:
            return team_meta, summary
        return team_meta
    heartbeat = HeartbeatState(logger=logger, phase="TEAM_DIR")
    for d in daterange(start_date, end_date):
        days += 1
        heartbeat.touch(day=d, force=True)
        scoreboard = fetch_scoreboard_with_retry(
            client=client,
            day=d,
            logger=logger,
            context="TEAM_DIR_SCOREBOARD",
            heartbeat=heartbeat,
        )
        if scoreboard is None:
            fetch_failures += 1
            heartbeat.touch(day=d, processed_inc=1, failures_inc=1)
            continue
        heartbeat.touch(day=d, processed_inc=1, fetched_inc=1)

        if not isinstance(scoreboard, dict):
            fetch_failures += 1
            logger.warning(
                f"TEAM_DIR scoreboard payload invalid date={d} type={type(scoreboard).__name__}"
            )
            heartbeat.touch(day=d, failures_inc=1, force=True)
            continue

        games = scoreboard.get("games", []) or []
        for item in games:
            g = unwrap_scoreboard_game(item)
            game_id = extract_scoreboard_game_id(g)
            if not game_id or game_id in game_ids_seen:
                continue
            game_ids_seen.add(game_id)
            if game_id in fetched_game_ids:
                continue
            fetched_game_ids.add(game_id)

            bs = fetch_boxscore_with_retry(
                client, game_id, logger, context="TEAM_DIR_BOXSCORE", heartbeat=heartbeat
            )
            if bs is None:
                fetch_failures += 1
                heartbeat.touch(day=d, processed_inc=1, failures_inc=1)
                continue

            boxscores += 1
            heartbeat.touch(day=d, processed_inc=1, fetched_inc=1)
            stats = {"invalid_team_ids": 0}
            team_recs = extract_boxscore_team_records_strict(bs, game_id, logger, stats=stats)
            skipped_invalid_ids += int(stats.get("invalid_team_ids", 0))
            for rec in team_recs:
                _upsert_team_meta(
                    team_meta,
                    rec["teamId"],
                    rec,
                    source=rec.get("source", f"boxscore:{game_id}"),
                )
        heartbeat.touch(day=d, force=True)
        if days % 14 == 0:
            logger.info(
                f"TEAM_DIR build progress days={days} teams={len(team_meta)} boxscores={boxscores} "
                f"skipped_invalid_ids={skipped_invalid_ids} fetch_failures={fetch_failures}"
            )

    summary = {
        "total_teams": len(team_meta),
        "total_game_ids_fetched": boxscores,
        "total_failures": fetch_failures,
        "total_invalid_team_ids": skipped_invalid_ids,
    }
    logger.info(
        f"TEAM_DIR build completed season={season} days={days} games={len(game_ids_seen)} "
        f"boxscores={boxscores} teams={len(team_meta)} "
        f"skipped_invalid_ids={skipped_invalid_ids} fetch_failures={fetch_failures}"
    )
    if return_stats:
        return team_meta, summary
    return team_meta


def rebuild_d1_ids_from_scoreboard(
    client: "SourceClient",
    logger: logging.Logger,
    start_date: date,
    end_date: date,
    mode: str,
    days_limit: int | None = None,
    max_boxscores_total: int | None = None,
) -> set[str]:
    if end_date < start_date:
        logger.info(
            f"D1_REBUILD mode={mode} skipped: start_date={start_date} > end_date={end_date}"
        )
        return set()
    if not _client_base_url(client):
        logger.warning(
            "D1_REBUILD no NCAA_API_BASE_URL set; using ESPN scoreboard fallback"
        )
        d1_ids: set[str] = set()
        rebuild_start = start_date
        if days_limit is not None and days_limit > 0:
            limit_start = end_date - timedelta(days=days_limit - 1)
            rebuild_start = max(rebuild_start, limit_start)
        days = 0
        for d in daterange(rebuild_start, end_date):
            days += 1
            day_df = fetch_espn_games_for_day(d, logger)
            if day_df.empty:
                continue
            for col in ["team_id", "opponent_team_id"]:
                if col not in day_df.columns:
                    continue
                d1_ids.update(
                    tid
                    for tid in day_df[col].apply(_safe_team_id).tolist()
                    if tid
                )
        d1_ids_sorted = sorted(d1_ids)
        logger.info(
            "D1_REBUILD "
            f"mode={mode} range={rebuild_start}..{end_date} days={days} "
            f"games_seen=n/a boxscores_fetched=0 d1_ids_count={len(d1_ids)} source=espn"
        )
        logger.info(f"D1_REBUILD sample={d1_ids_sorted[:10]}")
        return d1_ids

    rebuild_start = start_date
    if days_limit is not None and days_limit > 0:
        limit_start = end_date - timedelta(days=days_limit - 1)
        rebuild_start = max(rebuild_start, limit_start)

    d1_ids: set[str] = set()
    games_seen: set[str] = set()
    boxscores_fetched = 0
    days = 0
    heartbeat = HeartbeatState(logger=logger, phase="D1_REBUILD")

    for d in daterange(rebuild_start, end_date):
        days += 1
        heartbeat.touch(day=d, force=True)
        scoreboard = fetch_scoreboard_with_retry(
            client=client,
            day=d,
            logger=logger,
            context="D1_REBUILD_SCOREBOARD",
            heartbeat=heartbeat,
        )
        if scoreboard is None:
            heartbeat.touch(day=d, processed_inc=1, failures_inc=1)
            continue
        heartbeat.touch(day=d, processed_inc=1, fetched_inc=1)

        if not isinstance(scoreboard, dict):
            logger.warning(
                f"D1_REBUILD scoreboard payload invalid date={d} type={type(scoreboard).__name__}"
            )
            heartbeat.touch(day=d, failures_inc=1, force=True)
            continue

        games = scoreboard.get("games", []) or []
        for item in games:
            g = unwrap_scoreboard_game(item)
            game_id = extract_scoreboard_game_id(g)
            if not game_id or game_id in games_seen:
                continue
            games_seen.add(game_id)

            bs = fetch_boxscore_with_retry(
                client, game_id, logger, context="D1_REBUILD_BOXSCORE", heartbeat=heartbeat
            )
            if bs is None:
                heartbeat.touch(day=d, processed_inc=1, failures_inc=1)
                continue
            boxscores_fetched += 1
            heartbeat.touch(day=d, processed_inc=1, fetched_inc=1)

            for rec in extract_boxscore_team_records_strict(bs, game_id, logger):
                tid = canonical_team_id(rec.get("teamId"))
                if tid:
                    d1_ids.add(tid)

            if (
                max_boxscores_total is not None
                and max_boxscores_total > 0
                and boxscores_fetched >= max_boxscores_total
            ):
                heartbeat.touch(day=d, force=True)
                logger.info(
                    f"D1_REBUILD diagnostic cap reached boxscores_fetched={boxscores_fetched}"
                )
                d1_ids_sorted = sorted(d1_ids)
                logger.info(
                    "D1_REBUILD "
                    f"mode={mode} range={rebuild_start}..{end_date} days={days} "
                    f"games_seen={len(games_seen)} boxscores_fetched={boxscores_fetched} "
                    f"d1_ids_count={len(d1_ids)}"
                )
                logger.info(f"D1_REBUILD sample={d1_ids_sorted[:10]}")
                return d1_ids

        heartbeat.touch(day=d, force=True)

    d1_ids_sorted = sorted(d1_ids)
    logger.info(
        "D1_REBUILD "
        f"mode={mode} range={rebuild_start}..{end_date} days={days} "
        f"games_seen={len(games_seen)} boxscores_fetched={boxscores_fetched} "
        f"d1_ids_count={len(d1_ids)}"
    )
    logger.info(f"D1_REBUILD sample={d1_ids_sorted[:10]}")
    return d1_ids


def write_team_directory_cache(out_dir: str, team_meta: dict, logger: logging.Logger):
    path = os.path.join(out_dir, "team_directory.json")
    os.makedirs(out_dir, exist_ok=True)
    clean = {}
    for tid, rec in sorted(team_meta.items()):
        sid = canonical_team_id(tid)
        if sid is None:
            continue
        clean[sid] = {
            "teamId": sid,
            "nameShort": str(rec.get("nameShort", "") or ""),
            "nameFull": str(rec.get("nameFull", "") or ""),
            "seoname": str(rec.get("seoname", "") or ""),
            "name6Char": str(rec.get("name6Char", "") or ""),
            "source": str(rec.get("source", "") or ""),
        }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"TEAM_DIR wrote team_directory.json teams={len(clean)} path={path}")


def build_team_directory_names_map(team_meta: dict) -> dict[str, str]:
    return ti.build_team_id_map(
        aliases=TEAM_NAME_ALIASES,
        existing_map=ti.load_team_id_map(),
        team_meta=team_meta,
    )


def write_team_directory_names_cache(out_dir: str, names_map: dict[str, str], logger: logging.Logger):
    path = os.path.join(out_dir, "team_directory_names.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(sorted(names_map.items())), f, indent=2)
    team_id_map_path = ti.save_team_id_map(
        names_map,
        Path(out_dir) / "team_id_map.json",
    )
    logger.info(
        f"TEAM_DIR wrote team_directory_names.json names={len(names_map)} path={path} "
        f"team_id_map_path={team_id_map_path}"
    )


def load_team_directory_names_cache(out_dir: str, logger: logging.Logger) -> dict[str, str]:
    path = os.path.join(out_dir, "team_directory_names.json")
    combined = ti.load_team_id_map(Path(out_dir) / "team_id_map.json")
    if not os.path.exists(path):
        return combined
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as ex:
        logger.warning(f"TEAM_DIR failed to load team_directory_names.json path={path} err={ex}")
        return combined
    if not isinstance(payload, dict):
        logger.warning(f"TEAM_DIR invalid team_directory_names.json shape path={path}")
        return combined

    clean: dict[str, str] = dict(combined)
    for raw_name, raw_tid in payload.items():
        name_norm = apply_name_alias(normalize_name(raw_name))
        tid = canonical_team_id(raw_tid)
        if name_norm and tid:
            clean[name_norm] = tid
    logger.info(
        f"TEAM_DIR loaded team_directory_names.json names={len(clean)} path={path}"
    )
    return clean


def _save_team_aliases_if_changed(
    *,
    before_aliases: dict[str, str],
    after_aliases: dict[str, str],
    logger: logging.Logger,
    context: str,
) -> None:
    if dict(before_aliases) == dict(after_aliases):
        return
    alias_path = ti.save_team_aliases(after_aliases)
    TEAM_NAME_ALIASES.clear()
    TEAM_NAME_ALIASES.update(after_aliases)
    logger.info(
        "TEAM_ALIAS updated context=%s aliases=%s path=%s",
        context,
        len(after_aliases),
        alias_path,
    )


def _build_team_identity_map(
    *,
    out_dir: str | None = None,
    team_meta: dict | None = None,
    games_df: pd.DataFrame | None = None,
    ratings_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    path = Path(out_dir) / "team_id_map.json" if out_dir else ti.TEAM_ID_MAP_PATH
    return ti.build_team_id_map(
        aliases=TEAM_NAME_ALIASES,
        existing_map=ti.load_team_id_map(path),
        team_meta=team_meta,
        games_df=games_df,
        ratings_df=ratings_df,
    )


def _resolve_team_match(
    name: str,
    *,
    team_id_map: dict[str, str],
    logger: logging.Logger | None = None,
    source: str = "",
    remember: bool = True,
    allow_fuzzy: bool = True,
) -> ti.TeamMatchResult:
    return ti.resolve_team_match(
        name,
        team_id_map,
        TEAM_NAME_ALIASES,
        logger=logger,
        source=source,
        remember=remember,
        allow_fuzzy=allow_fuzzy,
    )


def merge_team_directory_from_games_df(team_meta: dict, games_df: pd.DataFrame):
    if games_df is None or games_df.empty:
        return
    for id_col, name_col, src in [
        ("team_id", "team", "games_history:team"),
        ("opponent_team_id", "opponent", "games_history:opponent"),
    ]:
        if id_col not in games_df.columns or name_col not in games_df.columns:
            continue
        sub = games_df[[id_col, name_col]].copy()
        for _, r in sub.iterrows():
            tid = _safe_team_id(r.get(id_col))
            if not tid:
                continue
            nm = r.get(name_col)
            obj = {"nameShort": nm} if isinstance(nm, str) else {}
            _upsert_team_meta(team_meta, tid, obj, source=src)


US_STATE_ABBREV_TOKENS = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "ia", "id", "il",
    "in", "ks", "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms", "mt", "nc", "nd", "ne",
    "nh", "nj", "nm", "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut",
    "va", "vt", "wa", "wi", "wv", "wy",
}

MASTER_TEAM_NAME_OVERRIDES_RAW: dict[str, list[str]] = {
    "South Florida": ["University of South Florida"],
    "Florida Atlantic": ["Florida Atlantic University"],
    "Central Arkansas": ["University of Central Arkansas"],
    "West Georgia": ["University of West Georgia"],
    "Eastern Kentucky": ["Eastern Kentucky University"],
    "North Alabama": ["University of North Alabama"],
    "Eastern Washington": ["Eastern Washington University"],
    "Northern Colorado": ["University of Northern Colorado"],
    "Northern Arizona": ["Northern Arizona University"],
    "Charleston Southern": ["Charleston Southern University"],
    "College of Charleston": ["College of Charleston South Carolina"],
    "North Carolina A&T": ["North Carolina A and T State University"],
    "Middle Tennessee": ["Middle Tennessee State University"],
    "Western Kentucky": ["Western Kentucky University"],
    "Northern Kentucky": ["Northern Kentucky University"],
    "Eastern Michigan": ["Eastern Michigan University"],
    "Central Michigan": ["Central Michigan University"],
    "Western Michigan": ["Western Michigan University"],
    "North Carolina Central": ["North Carolina Central University"],
    "Southern Illinois": ["Southern Illinois University at Carbondale"],
    "Central Connecticut State": ["Central Connecticut State University"],
    "Southeast Missouri State": ["Southeast Missouri State University"],
    "Eastern Illinois": ["Eastern Illinois University"],
    "Southern Indiana": ["University of Southern Indiana"],
    "Western Illinois": ["Western Illinois University"],
    "Western Carolina": ["Western Carolina University"],
    "Texas A&M Corpus Christi": ["A and M Corpus Christi"],
    "Southeastern Louisiana": ["Southeastern Louisiana University"],
    "Georgia Southern": ["Georgia Southern University"],
    "Arkansas Pine Bluff": ["University of Arkansas Pine Bluff"],
    "Southern": ["Southern University", "Southern U.", "Southern University Baton Rouge"],
    "Prairie View A&M": ["Prairie View A and M University"],
    "Alcorn State": ["Alcorn State University"],
    "Mississippi Valley State": ["Mississippi Valley State University"],
    "Miami (OH)": ["Miami OH"],
}

MASTER_TEAM_NAME_OVERRIDES: dict[str, list[str]] = {
    apply_name_alias(normalize_name(raw_name)): [
        apply_name_alias(normalize_name(v))
        for v in values
        if apply_name_alias(normalize_name(v))
    ]
    for raw_name, values in MASTER_TEAM_NAME_OVERRIDES_RAW.items()
}

WHITELIST_ALIAS_NORMALIZATION_RAW: dict[str, list[str]] = {
    "Miami (OH)": ["miami oh", "miami university ohio"],
}

WHITELIST_ALIAS_NORMALIZATION: dict[str, list[str]] = {
    apply_name_alias(normalize_name(raw_name)): [
        apply_name_alias(normalize_name(v))
        for v in values
        if apply_name_alias(normalize_name(v))
    ]
    for raw_name, values in WHITELIST_ALIAS_NORMALIZATION_RAW.items()
}

WHITELIST_PREFERRED_TEAM_ID_RAW: dict[str, str] = {
    "Miami (OH)": "1535",
}

WHITELIST_PREFERRED_TEAM_ID: dict[str, str] = {
    apply_name_alias(normalize_name(raw_name)): str(team_id)
    for raw_name, team_id in WHITELIST_PREFERRED_TEAM_ID_RAW.items()
    if canonical_team_id(team_id)
}


def _normalize_conf_name(raw: str) -> str:
    s = _norm_text(raw)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _normalize_team_match_key(raw_name) -> str:
    variants = ti.expand_team_name_variants(raw_name)
    return variants[0] if variants else ""


_TEAM_MATCH_TOKEN_STOPWORDS = {
    "and",
    "college",
    "of",
    "the",
    "university",
    "ny",
    "nc",
    "tx",
    "mn",
    "ri",
    "me",
    "dc",
}


def _team_match_tokens(name) -> set[str]:
    key = _normalize_team_match_key(name)
    if not key:
        return set()
    return {
        token
        for token in key.split()
        if token and token not in _TEAM_MATCH_TOKEN_STOPWORDS
    }


def _resolve_team_id_from_name(name, normalized_name_map: dict[str, str]) -> str:
    return _resolve_team_match(
        name,
        team_id_map=normalized_name_map,
        logger=None,
        source="resolve_team_id",
        remember=False,
        allow_fuzzy=True,
    ).team_id


def _expand_master_team_name_variants(raw_name: str) -> set[str]:
    name = str(raw_name or "").strip()
    if not name:
        return set()

    raw_variants = {name}
    base = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if base:
        raw_variants.add(base)

    for grp in re.findall(r"\(([^)]*)\)", name):
        grp = grp.strip()
        if not grp:
            continue
        grp_lc = grp.lower()
        if len(grp_lc) >= 3 or grp_lc not in US_STATE_ABBREV_TOKENS:
            raw_variants.add(grp)
        for part in re.split(r"[\\/|;]", grp):
            part = part.strip()
            part_lc = part.lower()
            if not part:
                continue
            if len(part_lc) < 3 and part_lc in US_STATE_ABBREV_TOKENS:
                continue
            if len(part_lc) < 2:
                continue
            raw_variants.add(part)

    variants: set[str] = set()
    for raw_variant in raw_variants:
        norm = apply_name_alias(normalize_name(raw_variant))
        if norm:
            variants.add(norm)

    base_norm = apply_name_alias(normalize_name(name))
    for extra in MASTER_TEAM_NAME_OVERRIDES.get(base_norm, []):
        if extra:
            variants.add(extra)
    return variants


def _parse_master_whitelist_text(raw_text: str, logger: logging.Logger) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_conf = ""
    for ln_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = str(raw_line).strip()
        if not line:
            continue
        if "," not in line:
            current_conf = line
            continue
        team_name, conf_from_row = [x.strip() for x in line.split(",", 1)]
        conference = conf_from_row or current_conf
        if not team_name or not conference:
            logger.warning(f"D1_MASTER malformed row line={ln_no} value={line!r}")
            continue
        rows.append({"team": team_name, "conference": conference})
    return rows


def _normalize_whitelist_text(raw_text: str) -> str:
    txt = str(raw_text or "")
    repl = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for k, v in repl.items():
        txt = txt.replace(k, v)
    return txt


def _normalize_whitelist_value(v: object) -> str:
    s = _normalize_whitelist_text(str(v or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_master_whitelist_json(
    payload: object,
    logger: logging.Logger,
    expected_season: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(payload, list):
        source_rows = payload
        payload_kind = "rows"
        season_value = expected_season
    elif isinstance(payload, dict):
        season_value = payload.get("season")
        try:
            season_num = int(season_value)
        except Exception:
            logger.warning(f"D1_MASTER invalid or missing season value={season_value!r}")
            return rows
        if season_num != int(expected_season):
            logger.warning(
                f"D1_MASTER season mismatch file_season={season_num} cfg_season={expected_season}"
            )
            return rows

        if "teams" in payload:
            source_rows = payload.get("teams", [])
            payload_kind = "teams"
        elif "rows" in payload:
            source_rows = payload.get("rows", [])
            payload_kind = "rows"
        elif "ids" in payload and "id_to_conf" in payload and "id_to_name" in payload:
            ids = payload.get("ids", [])
            id_to_conf = payload.get("id_to_conf", {})
            id_to_name = payload.get("id_to_name", {})
            if not isinstance(ids, list) or not isinstance(id_to_conf, dict) or not isinstance(id_to_name, dict):
                logger.warning("D1_MASTER invalid ids/id_to_conf/id_to_name shapes")
                return rows
            payload_kind = "ids"
            for idx, raw_id in enumerate(ids):
                sid = _safe_team_id(raw_id)
                if not sid:
                    logger.warning(f"D1_MASTER malformed ids row index={idx} value={raw_id!r}")
                    continue
                team_name = _normalize_whitelist_value(id_to_name.get(sid, ""))
                conference = _normalize_whitelist_value(id_to_conf.get(sid, ""))
                if not team_name or not conference:
                    logger.warning(
                        f"D1_MASTER malformed ids mapping team_id={sid!r} "
                        f"name={team_name!r} conf={conference!r}"
                    )
                    continue
                rows.append({"team": team_name, "conference": conference})
            return rows
        else:
            source_rows = []
            payload_kind = "unknown"
    else:
        source_rows = []
        payload_kind = "unknown"

    if not isinstance(source_rows, list):
        logger.warning(f"D1_MASTER invalid JSON shape kind={payload_kind} type={type(source_rows).__name__}")
        return rows

    for idx, row in enumerate(source_rows):
        team_name = ""
        conference = ""
        if isinstance(row, dict):
            team_name = _normalize_whitelist_value(row.get("team") or row.get("name") or "")
            conference = _normalize_whitelist_value(row.get("conference") or row.get("conf") or "")
        elif isinstance(row, str):
            line = _normalize_whitelist_value(row)
            if "," in line:
                team_name, conference = [_normalize_whitelist_value(x) for x in line.split(",", 1)]
        if not team_name or not conference:
            logger.warning(f"D1_MASTER malformed JSON row index={idx} value={row!r}")
            continue
        rows.append({"team": team_name, "conference": conference})
    return rows


def load_d1_master_whitelist(path: str, logger: logging.Logger, expected_season: int) -> list[dict[str, str]]:
    if not os.path.exists(path):
        logger.warning(f"D1_MASTER whitelist file missing path={path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_text = _normalize_whitelist_text(raw_text)

    parsed_rows: list[dict[str, str]] = []
    try:
        payload = json.loads(raw_text)
        parsed_rows = _parse_master_whitelist_json(payload, logger, expected_season)
        if parsed_rows:
            logger.info(f"D1_MASTER parsed JSON format path={path} rows={len(parsed_rows)}")
    except json.JSONDecodeError:
        parsed_rows = []

    if not parsed_rows:
        parsed_rows = _parse_master_whitelist_text(raw_text, logger)
        if parsed_rows:
            logger.info(f"D1_MASTER parsed text format path={path} rows={len(parsed_rows)}")

    if not parsed_rows:
        logger.warning(f"D1_MASTER no valid rows found path={path}")
        return []

    rows: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    confs: set[str] = set()
    for row in parsed_rows:
        team_name = _normalize_whitelist_value(row.get("team", ""))
        conference = _normalize_whitelist_value(row.get("conference", ""))
        team_norm = apply_name_alias(normalize_name(team_name))
        conf_norm = _normalize_conf_name(conference)
        if not team_name or not conference or not team_norm or not conf_norm:
            continue
        key = (team_norm, conf_norm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        confs.add(conference)
        rows.append({"team": team_name, "conference": conference})

    logger.info(
        f"D1_MASTER loaded path={path} teams={len(rows)} conferences={len(confs)}"
    )
    return rows


def build_d1_ids_from_master_whitelist(
    master_rows: list[dict[str, str]],
    team_name_map: dict[str, str],
    games_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[set[str], dict[str, str], dict]:
    if not master_rows:
        stats = {
            "master_rows": 0,
            "matched_rows": 0,
            "ambiguous_rows": 0,
            "unresolved_rows": 0,
        }
        return set(), {}, stats

    team_id_map = dict(team_name_map)
    before_team_id_map = dict(team_id_map)
    before_aliases = dict(TEAM_NAME_ALIASES)
    d1_ids: set[str] = set()
    d1_conf_by_id: dict[str, str] = {}
    ambiguous_rows: list[dict[str, object]] = []
    unresolved_rows: list[str] = []

    for row in master_rows:
        team_name = str(row.get("team", "")).strip()
        conference = str(row.get("conference", "")).strip()
        if not team_name or not conference:
            continue

        team_name_norm = apply_name_alias(normalize_name(team_name))
        candidate_names: list[str] = [team_name]
        for name_variant in sorted(_expand_master_team_name_variants(team_name)):
            if name_variant:
                candidate_names.append(name_variant)
        cleaned_team_name = _normalize_whitelist_match_name(team_name)
        if cleaned_team_name:
            candidate_names.append(cleaned_team_name)
        for override_name in MASTER_TEAM_NAME_OVERRIDES.get(team_name_norm, []):
            if override_name:
                candidate_names.append(override_name)
        alias_variants = WHITELIST_ALIAS_NORMALIZATION.get(team_name_norm, [])
        for alias_name in alias_variants:
            if alias_name:
                candidate_names.append(alias_name)

        unique_candidate_names: list[str] = []
        seen_candidate_names: set[str] = set()
        for candidate_name in candidate_names:
            candidate_key = str(candidate_name or "").strip()
            if candidate_key and candidate_key not in seen_candidate_names:
                seen_candidate_names.add(candidate_key)
                unique_candidate_names.append(candidate_key)

        candidate_matches: list[ti.TeamMatchResult] = []
        for candidate_name in unique_candidate_names:
            match = _resolve_team_match(
                candidate_name,
                team_id_map=team_id_map,
                logger=None,
                source="whitelist",
                remember=False,
                allow_fuzzy=True,
            )
            if match.matched:
                candidate_matches.append(match)
        method_rank = {"direct": 3, "alias": 2, "fuzzy": 1}
        best_method_rank = max(
            (method_rank.get(match.method, 0) for match in candidate_matches),
            default=0,
        )
        ranked_candidate_matches = [
            match
            for match in candidate_matches
            if method_rank.get(match.method, 0) == best_method_rank
        ]

        preferred_tid_raw = WHITELIST_PREFERRED_TEAM_ID.get(team_name_norm, "")
        preferred_tid = canonical_team_id(preferred_tid_raw)
        unique_candidate_ids = {
            match.team_id
            for match in ranked_candidate_matches
            if match.team_id
        }
        selected_match: ti.TeamMatchResult | None = None
        if preferred_tid and preferred_tid in unique_candidate_ids:
            selected_match = next(
                (
                    match
                    for match in ranked_candidate_matches
                    if match.team_id == preferred_tid
                ),
                None,
            )
        elif len(unique_candidate_ids) == 1:
            selected_match = sorted(
                ranked_candidate_matches,
                key=lambda match: (
                    method_rank.get(match.method, 0),
                    match.score,
                    len(match.input_key),
                ),
                reverse=True,
            )[0]

        if selected_match is not None:
            confirmed_match = _resolve_team_match(
                selected_match.input_key or team_name,
                team_id_map=team_id_map,
                logger=None,
                source="whitelist",
                remember=True,
                allow_fuzzy=True,
            )
            selected_tid = confirmed_match.team_id or selected_match.team_id
            if selected_tid:
                d1_ids.add(selected_tid)
                if selected_tid not in d1_conf_by_id:
                    d1_conf_by_id[selected_tid] = conference
                continue

        if unique_candidate_ids:
            ambiguous_rows.append(
                {
                    "team": team_name,
                    "conference": conference,
                    "candidates": sorted(unique_candidate_ids),
                }
            )
        else:
            unresolved_rows.append(team_name)
            logger.warning("WARNING TEAM_MATCH_FAILED name=%r source=whitelist", team_name)

    total = len(master_rows)
    matched = total - len(ambiguous_rows) - len(unresolved_rows)
    match_rate = (float(matched) / float(total)) if total > 0 else 0.0
    ti.log_team_match_coverage(
        logger,
        scope="whitelist",
        matched=matched,
        total=total,
    )
    logger.info(
        f"WHITELIST_MATCH total={total} matched={matched} "
        f"ambiguous={len(ambiguous_rows)} missing={len(unresolved_rows)} "
        f"match_rate={match_rate:.4f}"
    )
    if ambiguous_rows:
        ambiguous_names = [
            f"{row['team']} -> {','.join(row['candidates'])}" for row in ambiguous_rows
        ]
        logger.warning(f"WHITELIST_MATCH ambiguous_names={ambiguous_names}")
    if unresolved_rows:
        logger.warning(f"WHITELIST_MATCH missing_names={unresolved_rows}")

    fallback_used = False
    if match_rate < 0.50:
        logger.warning(
            "WHITELIST_MATCH degraded mode - falling back to games_history teams"
        )
        fallback_ids: set[str] = set()
        fallback_conf_by_id: dict[str, str] = {}
        fallback_name_to_id: dict[str, str] = {}
        for raw_name, raw_tid in team_name_map.items():
            tid = _safe_team_id(raw_tid)
            if not tid:
                continue
            for key in {
                apply_name_alias(normalize_name(raw_name)),
                _normalize_whitelist_match_name(raw_name),
            }:
                if key and key not in fallback_name_to_id:
                    fallback_name_to_id[key] = tid

        games_source = games_df if games_df is not None else pd.DataFrame()
        for team_col, id_col, conf_col in [
            ("team", "team_id", "team_conf"),
            ("opponent", "opponent_team_id", "opp_conf"),
        ]:
            if games_source.empty or team_col not in games_source.columns:
                continue
            team_series = games_source[team_col].fillna("").astype(str)
            id_series = (
                games_source[id_col].fillna("").astype(str)
                if id_col in games_source.columns
                else pd.Series("", index=games_source.index, dtype="object")
            )
            conf_series = (
                games_source[conf_col].fillna("").astype(str)
                if conf_col in games_source.columns
                else pd.Series("", index=games_source.index, dtype="object")
            )
            for idx, raw_name in team_series.items():
                tid = _safe_team_id(id_series.at[idx])
                if not tid:
                    for key in {
                        apply_name_alias(normalize_name(raw_name)),
                        _normalize_whitelist_match_name(raw_name),
                    }:
                        if key and key in fallback_name_to_id:
                            tid = fallback_name_to_id[key]
                            break
                if not tid:
                    continue
                fallback_ids.add(tid)
                conf_value = str(conf_series.at[idx]).strip()
                if conf_value and tid not in fallback_conf_by_id:
                    fallback_conf_by_id[tid] = conf_value

        if fallback_ids:
            d1_ids = fallback_ids
            if fallback_conf_by_id:
                d1_conf_by_id.update(fallback_conf_by_id)
            fallback_used = True
        logger.warning(
            "WHITELIST_MATCH fallback_used=%s match_rate=%.4f fallback_ids=%s",
            str(bool(fallback_used)).lower(),
            match_rate,
            len(d1_ids),
        )

    if team_id_map != before_team_id_map:
        ti.save_team_id_map(team_id_map)
        logger.info("TEAM_ID_MAP updated context=whitelist names=%s", len(team_id_map))
    _save_team_aliases_if_changed(
        before_aliases=before_aliases,
        after_aliases=TEAM_NAME_ALIASES,
        logger=logger,
        context="whitelist",
    )

    stats = {
        "master_rows": total,
        "matched_rows": matched,
        "ambiguous_rows": len(ambiguous_rows),
        "unresolved_rows": len(unresolved_rows),
        "fallback_used": fallback_used,
        "match_rate": match_rate,
    }
    return d1_ids, d1_conf_by_id, stats


def map_team_names_to_ids(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if "team_id" not in out.columns:
        out["team_id"] = ""
    if "opponent_team_id" not in out.columns:
        out["opponent_team_id"] = ""

    out["team_id"] = out["team_id"].apply(_safe_team_id)
    out["opponent_team_id"] = out["opponent_team_id"].apply(_safe_team_id)
    resolve_cache: dict[str, str] = {}

    def resolve(name):
        if not isinstance(name, str):
            return ""
        cached = resolve_cache.get(name)
        if cached is not None:
            return cached
        resolved = _resolve_team_match(
            name,
            team_id_map=name_map,
            logger=None,
            source="map_team_names_to_ids",
            remember=True,
            allow_fuzzy=True,
        ).team_id
        resolve_cache[name] = resolved
        return resolved

    if "team" in out.columns:
        out["team_id"] = out["team_id"].where(
            out["team_id"] != "",
            out["team"].apply(resolve),
        )

    if "opponent" in out.columns:
        out["opponent_team_id"] = out["opponent_team_id"].where(
            out["opponent_team_id"] != "",
            out["opponent"].apply(resolve),
        )

    return out


def backfill_ids_from_directory(
    games_df: pd.DataFrame,
    names_map: dict[str, str],
    out_dir: str,
    logger: logging.Logger,
    valid_team_ids: set[str] | None = None,
) -> pd.DataFrame:
    if games_df is None or games_df.empty:
        return games_df

    g = games_df.copy()
    for c in ["team_id", "opponent_team_id"]:
        if c not in g.columns:
            g[c] = ""
        g[c] = g[c].apply(_safe_team_id)
    if valid_team_ids:
        g["team_id"] = g["team_id"].apply(lambda x: x if x in valid_team_ids else "")
        g["opponent_team_id"] = g["opponent_team_id"].apply(lambda x: x if x in valid_team_ids else "")

    before_missing_team_id = int((g["team_id"] == "").sum())
    before_missing_opp_id = int((g["opponent_team_id"] == "").sum())
    normalized_names_map: dict[str, str] = dict(names_map)
    before_team_id_map = dict(normalized_names_map)
    before_aliases = dict(TEAM_NAME_ALIASES)
    resolve_cache: dict[str, str] = {}

    def resolve_cached(name: str) -> str:
        cached = resolve_cache.get(name)
        if cached is not None:
            return cached
        resolved = _resolve_team_match(
            name,
            team_id_map=normalized_names_map,
            logger=logger,
            source="history_backfill",
            remember=True,
            allow_fuzzy=True,
        ).team_id
        resolve_cache[name] = resolved
        return resolved

    debug_rows = []
    missing_mask = (g["team_id"] == "") | (g["opponent_team_id"] == "")
    idxs = g.index[missing_mask].tolist()
    for idx in idxs:
        team_name = g.at[idx, "team"] if "team" in g.columns else ""
        opp_name = g.at[idx, "opponent"] if "opponent" in g.columns else ""
        matched_team_id = g.at[idx, "team_id"]
        matched_opp_id = g.at[idx, "opponent_team_id"]
        sources = []

        if matched_team_id == "":
            found = resolve_cached(team_name)
            if found:
                g.at[idx, "team_id"] = found
                matched_team_id = found
                sources.append("team_name_map")

        if matched_opp_id == "":
            found = resolve_cached(opp_name)
            if found:
                g.at[idx, "opponent_team_id"] = found
                matched_opp_id = found
                sources.append("opp_name_map")

        debug_rows.append({
            "row_index": int(idx),
            "team_name": team_name,
            "matched_team_id": matched_team_id,
            "opp_name": opp_name,
            "matched_opp_id": matched_opp_id,
            "match_source": "|".join(sources) if sources else "no_match",
        })

    after_missing_team_id = int((g["team_id"] == "").sum())
    after_missing_opp_id = int((g["opponent_team_id"] == "").sum())
    logger.info(
        f"ID_BACKFILL before_missing_team_id={before_missing_team_id} after_missing_team_id={after_missing_team_id} "
        f"before_missing_opp_id={before_missing_opp_id} after_missing_opp_id={after_missing_opp_id}"
    )
    ti.log_team_match_coverage(
        logger,
        scope="history_backfill",
        matched=(before_missing_team_id - after_missing_team_id) + (before_missing_opp_id - after_missing_opp_id),
        total=before_missing_team_id + before_missing_opp_id,
    )

    debug_path = os.path.join(out_dir, "id_backfill_debug.csv")
    pd.DataFrame(debug_rows, columns=[
        "row_index", "team_name", "matched_team_id", "opp_name", "matched_opp_id", "match_source"
    ]).to_csv(debug_path, index=False)
    logger.info(f"ID_BACKFILL wrote debug path={debug_path} rows={len(debug_rows)}")

    if after_missing_team_id > 0 or after_missing_opp_id > 0:
        missing_rows = g[(g["team_id"] == "") | (g["opponent_team_id"] == "")]
        sample = missing_rows[["team", "opponent", "team_id", "opponent_team_id"]].head(20).to_dict(orient="records")
        logger.warning(
            f"ID_BACKFILL unresolved team_id={after_missing_team_id} opponent_team_id={after_missing_opp_id} sample={sample}"
        )

    if normalized_names_map != before_team_id_map:
        ti.save_team_id_map(normalized_names_map, Path(out_dir) / "team_id_map.json")
        logger.info("TEAM_ID_MAP updated context=history_backfill names=%s", len(normalized_names_map))
    _save_team_aliases_if_changed(
        before_aliases=before_aliases,
        after_aliases=TEAM_NAME_ALIASES,
        logger=logger,
        context="history_backfill",
    )

    return g


def enrich_ids_from_history_boxscores(
    client: "SourceClient", games_df: pd.DataFrame, team_meta: dict, logger: logging.Logger
) -> pd.DataFrame:
    if games_df is None or games_df.empty or "game_id" not in games_df.columns:
        return games_df
    if not _client_base_url(client):
        logger.info(
            "ID_BACKFILL_BOX skipped reason=ncaa_api_base_url_unset"
        )
        return games_df

    g = games_df.copy()
    for c in ["team_id", "opponent_team_id"]:
        if c not in g.columns:
            g[c] = ""
        g[c] = g[c].apply(_safe_team_id)

    unresolved = g[(g["team_id"] == "") | (g["opponent_team_id"] == "")]
    if unresolved.empty:
        return g

    game_ids = [str(x) for x in unresolved["game_id"].dropna().astype(str).unique().tolist()]
    filled_team = 0
    filled_opp = 0
    fetched = 0
    heartbeat = HeartbeatState(logger=logger, phase="ID_BACKFILL_BOX")
    for gid in game_ids:
        heartbeat.touch(last_url=f"/game/{gid}/boxscore")
        idxs = g.index[(g["game_id"].astype(str) == gid) & ((g["team_id"] == "") | (g["opponent_team_id"] == ""))]
        if len(idxs) == 0:
            continue
        bs = fetch_boxscore_with_retry(
            client, gid, logger, context="ID_BACKFILL_BOX", heartbeat=heartbeat
        )
        if bs is None:
            heartbeat.touch(processed_inc=1, failures_inc=1, force=True)
            continue
        fetched += 1
        heartbeat.touch(processed_inc=1, fetched_inc=1)
        team_recs = extract_boxscore_team_records_strict(bs, gid, logger)

        local_name_to_id = {}
        local_ids = set()
        for rec in team_recs:
            tid = rec["teamId"]
            local_ids.add(tid)
            _upsert_team_meta(team_meta, tid, rec, source=f"history_boxscore:{gid}")
            for raw in [
                rec.get("nameShort"),
                rec.get("nameFull"),
                rec.get("name6Char"),
                str(rec.get("seoname", "")).replace("-", " "),
            ]:
                n = apply_name_alias(normalize_name(raw))
                if n and n not in local_name_to_id:
                    local_name_to_id[n] = tid

        for idx in idxs:
            team_name = g.at[idx, "team"] if "team" in g.columns else ""
            opp_name = g.at[idx, "opponent"] if "opponent" in g.columns else ""
            team_id = g.at[idx, "team_id"]
            opp_id = g.at[idx, "opponent_team_id"]

            if team_id == "":
                found = _safe_team_id(local_name_to_id.get(apply_name_alias(normalize_name(team_name)), ""))
                if found:
                    g.at[idx, "team_id"] = found
                    team_id = found
                    filled_team += 1

            if opp_id == "":
                found = _safe_team_id(local_name_to_id.get(apply_name_alias(normalize_name(opp_name)), ""))
                if found:
                    g.at[idx, "opponent_team_id"] = found
                    opp_id = found
                    filled_opp += 1

            # If only one side matched in a two-team game, infer the other side.
            if len(local_ids) == 2:
                if team_id and not opp_id:
                    other = [x for x in local_ids if x != team_id]
                    if other:
                        g.at[idx, "opponent_team_id"] = other[0]
                        filled_opp += 1
                if opp_id and not team_id:
                    other = [x for x in local_ids if x != opp_id]
                    if other:
                        g.at[idx, "team_id"] = other[0]
                        filled_team += 1

    logger.info(
        f"ID_BACKFILL_BOX fetched_games={fetched}/{len(game_ids)} "
        f"filled_team_id={filled_team} filled_opponent_team_id={filled_opp}"
    )
    return g


def build_team_name_index(team_meta: dict) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for tid, rec in team_meta.items():
        for raw in [
            rec.get("nameShort", ""),
            rec.get("nameFull", ""),
            str(rec.get("seoname", "")).replace("-", " "),
        ]:
            norm = normalize_name(raw)
            if not norm:
                continue
            idx.setdefault(norm, set()).add(tid)
    return idx


def fallback_build_d1_ids_from_names(
    school_names_path: str, team_meta: dict, out_dir: str, logger: logging.Logger
) -> tuple[set[str], int]:
    if not os.path.exists(school_names_path):
        logger.info(f"D1_FALLBACK skipped: file not found {school_names_path}")
        return set(), -1

    with open(school_names_path, "r") as f:
        raw_names = [line.strip() for line in f]
    input_names = []
    seen = set()
    for name in raw_names:
        if not name:
            continue
        # Skip obvious conference/header rows.
        if name.lower().startswith("conference:") or name.lower().endswith("conference"):
            continue
        if name in seen:
            continue
        seen.add(name)
        input_names.append(name)

    name_idx = build_team_name_index(team_meta)
    matched_ids: set[str] = set()
    debug_rows = []
    unmatched = 0

    for input_name in input_names:
        n = normalize_name(input_name)
        matched_tid = ""
        match_type = ""
        for field, candidate in [
            ("nameShort", n),
            ("nameFull", n),
            ("seoname", n),
        ]:
            tids = sorted(name_idx.get(candidate, set()))
            if len(tids) == 1:
                matched_tid = tids[0]
                match_type = field
                break
        if matched_tid:
            matched_ids.add(matched_tid)
            rec = team_meta.get(matched_tid, {})
            debug_rows.append({
                "input_name": input_name,
                "matched_teamId": matched_tid,
                "matched_nameShort": rec.get("nameShort", ""),
                "matched_nameFull": rec.get("nameFull", ""),
                "matched_seoname": rec.get("seoname", ""),
                "match_type": match_type,
            })
        else:
            unmatched += 1
            debug_rows.append({
                "input_name": input_name,
                "matched_teamId": "",
                "matched_nameShort": "",
                "matched_nameFull": "",
                "matched_seoname": "",
                "match_type": "unmatched",
            })

    debug_path = os.path.join(out_dir, "d1_team_ids_debug.csv")
    pd.DataFrame(debug_rows, columns=[
        "input_name", "matched_teamId", "matched_nameShort", "matched_nameFull", "matched_seoname", "match_type"
    ]).to_csv(debug_path, index=False)
    logger.info(
        f"D1_FALLBACK matched_ids={len(matched_ids)} unmatched={unmatched} source={school_names_path} debug_path={debug_path}"
    )
    return matched_ids, unmatched

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def setup_logger(log_path: str) -> logging.Logger:
    logging_cfg = (
        load_public_config().get("logging", {})
        if isinstance(load_public_config().get("logging", {}), dict)
        else {}
    )
    debug_enabled = to_bool_flag(os.environ.get("HOOPSIQ_DEBUG", False))
    level_name = str(
        logging_cfg.get("debug_level" if debug_enabled else "level", "INFO")
    ).upper()
    fmt_key = "debug_format" if debug_enabled else "format"
    fmt_value = str(logging_cfg.get(fmt_key, "%(asctime)s %(levelname)s %(message)s"))

    logger = logging.getLogger("ncaab_ranker")
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter(fmt_value)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

@dataclass
class SourceClient:
    base_url: str
    timeout: tuple[float, float] = (HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S)

    def get_json(self, path: str, params: dict | None = None) -> dict:
        if requests is None:
            raise RuntimeError(
                "requests package is required for network operations. "
                "Install dependencies from requirements.txt."
            )
        url = self.base_url.rstrip('/') + '/' + path.lstrip('/')
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

def fetch_games_stub(
    client: SourceClient,
    season: int,
    logger: logging.Logger,
    day: date | None = None,
    progress: dict | None = None,
    phase: str = "FETCH_GAMES",
    heartbeat: HeartbeatState | None = None,
    max_boxscores: int | None = None,
    player_rows_sink: list[dict] | None = None,
    debug_out_dir: str | None = None,
) -> pd.DataFrame:
    global _PLAYER_DUMPED_ONCE
    logger.info("Fetching NCAA men's basketball game results")

    now = datetime.now(tz=NY)

    if day is not None:
        chosen_path = f"/scoreboard/basketball-men/d1/{day.year}/{day.month:02d}/{day.day:02d}"
    else:
        chosen_path = "/scoreboard/basketball-men/d1"

    actual_day = None
    try:
        parts = chosen_path.strip("/").split("/")
        y, m, d = int(parts[-3]), int(parts[-2]), int(parts[-1])
        actual_day = date(y, m, d)
    except Exception:
        actual_day = now.date()

    logger.info(f"Using scoreboard endpoint: {chosen_path}")
    scoreboard = None
    if _client_base_url(client):
        scoreboard = get_json_with_retry(
            client=client,
            path=chosen_path,
            logger=logger,
            phase=f"{phase}_SCOREBOARD",
            item_key=actual_day.isoformat() if actual_day else chosen_path,
            heartbeat=heartbeat,
        )
    else:
        logger.info(f"{phase} NCAA_API_BASE_URL unset; using ESPN fallback day={actual_day}")

    if scoreboard is None:
        espn_df = fetch_espn_games_for_day(
            day=actual_day,
            logger=logger,
            d1_team_ids=progress.get("d1_team_ids") if isinstance(progress, dict) else None,
            d1_team_meta=progress.get("d1_team_meta") if isinstance(progress, dict) else None,
        )
        if heartbeat is not None:
            if espn_df.empty:
                heartbeat.touch(day=actual_day, failures_inc=1, force=True)
            else:
                heartbeat.touch(day=actual_day, fetched_inc=1, force=True)
        if espn_df.empty:
            logger.warning(f"{phase} scoreboard unavailable day={actual_day}")
        return espn_df
    if not isinstance(scoreboard, dict):
        logger.warning(
            f"{phase} scoreboard payload invalid day={actual_day} type={type(scoreboard).__name__}"
        )
        espn_df = fetch_espn_games_for_day(
            day=actual_day,
            logger=logger,
            d1_team_ids=progress.get("d1_team_ids") if isinstance(progress, dict) else None,
            d1_team_meta=progress.get("d1_team_meta") if isinstance(progress, dict) else None,
        )
        if espn_df.empty:
            return _empty_games_long_df()
        return espn_df
    if heartbeat is not None:
        heartbeat.touch(day=actual_day, fetched_inc=1)
    games = scoreboard.get("games", []) or []

    states = sorted({
        str(unwrap_scoreboard_game(item).get("gameState", "")).strip().lower()
        for item in games
    })
    logger.info(f"DEBUG scoreboard states seen: {states}")

    d1_team_ids = None
    d1_team_meta = None
    if isinstance(progress, dict):
        d1_team_ids = progress.get("d1_team_ids")
        d1_team_meta = progress.get("d1_team_meta")



    # 2) Fetch one game's boxscore and return team stats + metadata
    def fetch_boxscore(game_id: str) -> tuple[dict, dict, dict, str, str, dict, dict]:
        bs = fetch_boxscore_with_retry(
            client, game_id, logger, context=f"{phase}_BOXSCORE", heartbeat=heartbeat
        )
        if bs is None:
            raise RuntimeError("boxscore unavailable")

        teams = bs.get("teams", []) or []
        tb = bs.get("teamBoxscore", []) or []

        # Identify home/away teamIds from the teams list
        home_id = None
        away_id = None
        home_team_obj = {}
        away_team_obj = {}
        for t in teams:
            tid = str(t.get("teamId") or "")
            if not tid:
                continue
            if bool(t.get("isHome")):
                home_id = tid
                home_team_obj = t
            else:
                away_id = tid
                away_team_obj = t

        # Map teamId -> teamStats
        stats_by_id = {}
        for row in tb:
            tid = str(row.get("teamId") or "")
            if not tid:
                continue
            stats_by_id[tid] = row.get("teamStats", {}) or {}

        home_stats = stats_by_id.get(str(home_id), {}) if home_id else {}
        away_stats = stats_by_id.get(str(away_id), {}) if away_id else {}

        # one-time sanity debug
        if not hasattr(fetch_boxscore, "_dbg"):
            fetch_boxscore._dbg = True
            logger.info(f"DEBUG teamStats home keys sample: {list(home_stats.keys())[:40]}")
            logger.info(f"DEBUG teamStats away keys sample: {list(away_stats.keys())[:40]}")

        return home_stats, away_stats, bs, _safe_team_id(home_id), _safe_team_id(away_id), home_team_obj, away_team_obj

    # 3) Helper to normalize stats keys into what our pipeline expects
    def normalize_team_stats(stats: dict) -> dict:
        # Recursively search for a stat by key name
        def find_key(obj, keys):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in keys:
                        return v
                for v in obj.values():
                    out = find_key(v, keys)
                    if out is not None:
                        return out
            elif isinstance(obj, list):
                for it in obj:
                    out = find_key(it, keys)
                    if out is not None:
                        return out
            return None

        # Parse "made-attempted" like "25-60" -> attempts = 60
        def parse_made_att(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val)
            m = re.search(r"(\d+)\s*-\s*(\d+)", s)
            if m:
                return int(m.group(2))
            return None

        def safe_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        # direct numeric keys
        fga = find_key(stats, {"fieldGoalsAttempted", "fga", "FGA", "fgAtt", "fg_attempts", "fgA", "fieldGoalsAtt"})
        fta = find_key(stats, {"freeThrowsAttempted", "fta", "FTA", "ftAtt", "ft_attempts", "ftA", "freeThrowsAtt"})
        orb = find_key(stats, {"offensiveRebounds", "orb", "ORB", "offReb", "off_reb", "oReb", "offensiveReb"})
        tov = find_key(stats, {"turnovers", "to", "TOV", "turnover", "TO", "tov"})

        # if attempts aren't direct numbers, try string formats
        if fga is None:
            fga = parse_made_att(find_key(stats, {"fieldGoals", "FG", "fg"}))
        if fta is None:
            fta = parse_made_att(find_key(stats, {"freeThrows", "FT", "ft"}))

        return {
            "fga": safe_int(fga),
            "fta": safe_int(fta),
            "orb": safe_int(orb),
            "to": safe_int(tov),
        }

    rows = []
    final_games_by_id = {}
    for item in games:
        g = unwrap_scoreboard_game(item)
        state = str(g.get("gameState", "")).strip().lower()
        if not state.startswith("final"):
            if state not in {"pre", "live"}:
                logger.info(f"Skipping unknown gameState: {state}")
            continue
        game_id = extract_scoreboard_game_id(g)
        if game_id:
            final_games_by_id[game_id] = g

    final_game_ids = sorted(final_games_by_id.keys())

    if progress is not None:
        progress["games_total_est"] = progress.get("games_total_est", 0) + len(final_game_ids)
        if progress.get("pbar") is None:
            progress["pbar"] = tqdm(
                total=progress["games_total_est"],
                unit="game",
                dynamic_ncols=True,
                smoothing=0.05,
            )
        else:
            progress["pbar"].total = progress["games_total_est"]
            progress["pbar"].refresh()

    # TEMP DEBUG: inspect one boxscore structure for one final game
    if final_game_ids and not hasattr(fetch_games_stub, "_debugged_one"):
        fetch_games_stub._debugged_one = True
        debug_game_id = final_game_ids[0]
        bs = fetch_boxscore_with_retry(
            client, debug_game_id, logger, context=f"{phase}_DEBUG_BOXSCORE", heartbeat=heartbeat
        )
        if isinstance(bs, dict):
            logger.info(f"DEBUG boxscore type(teamBoxscore)={type(bs.get('teamBoxscore')).__name__}")
            logger.info(f"DEBUG boxscore top keys={list(bs.keys())[:40]}")
            tb = bs.get("teamBoxscore")
            if isinstance(tb, dict):
                logger.info(f"DEBUG teamBoxscore dict keys sample={list(tb.keys())[:10]}")
            elif isinstance(tb, list):
                logger.info(f"DEBUG teamBoxscore list len={len(tb)} sample0 keys={list((tb[0] or {}).keys())[:40] if tb else []}")
            else:
                logger.info(f"DEBUG teamBoxscore value sample={str(tb)[:200]}")
            teams_obj = bs.get("teams", [])
            logger.info(f"DEBUG teams len={len(teams_obj)} sample0 keys={list((teams_obj[0] or {}).keys())[:40] if teams_obj else []}")

    boxscores_attempted = 0
    for game_id in final_game_ids:
        if max_boxscores is not None and max_boxscores > 0 and boxscores_attempted >= max_boxscores:
            logger.info(
                f"{phase} diagnostic cap reached max_boxscores={max_boxscores} day={actual_day}"
            )
            break
        boxscores_attempted += 1
        g = final_games_by_id[game_id]
        home = g.get("home", {}) or {}
        away = g.get("away", {}) or {}
        home_team_id = (
            canonical_team_id(home.get("teamId") or home.get("teamID") or home.get("id"))
            or _safe_team_id(home.get("teamId") or home.get("teamID") or home.get("id"))
        )
        away_team_id = (
            canonical_team_id(away.get("teamId") or away.get("teamID") or away.get("id"))
            or _safe_team_id(away.get("teamId") or away.get("teamID") or away.get("id"))
        )

        if d1_team_meta is not None:
            _upsert_team_meta(d1_team_meta, home_team_id, home, source=f"scoreboard:{game_id}")
            _upsert_team_meta(d1_team_meta, away_team_id, away, source=f"scoreboard:{game_id}")

        home_name = home.get("names", {}).get("short") or home.get("names", {}).get("full") or home.get("name")
        away_name = away.get("names", {}).get("short") or away.get("names", {}).get("full") or away.get("name")

        home_score = safe_int(home.get("score"))
        away_score = safe_int(away.get("score"))

        game_date = actual_day.strftime("%Y-%m-%d")

        # Pull boxscore stats for the game
        bs_payload = {}
        box_home_id = ""
        box_away_id = ""
        box_home_obj = {}
        box_away_obj = {}
        try:
            (
                home_raw_stats,
                away_raw_stats,
                bs_payload,
                box_home_id,
                box_away_id,
                box_home_obj,
                box_away_obj,
            ) = fetch_boxscore(game_id)
            if heartbeat is not None:
                heartbeat.touch(day=actual_day, processed_inc=1, fetched_inc=1)
        except Exception as e:
            logger.info(f"Boxscore fetch failed for game_id={game_id} ({e}). Using zeros.")
            home_raw_stats, away_raw_stats = {}, {}
            bs_payload = {}
            if heartbeat is not None:
                heartbeat.touch(day=actual_day, processed_inc=1, failures_inc=1)
        if isinstance(bs_payload, dict):
            tb_obj = (
                bs_payload.get("teamBoxscore")
                if "teamBoxscore" in bs_payload
                else bs_payload.get("teamBoxscores")
            )
            tb_exists = ("teamBoxscore" in bs_payload) or ("teamBoxscores" in bs_payload)
            tb_type = type(tb_obj).__name__ if tb_exists else "missing"
            tb_len = len(tb_obj) if isinstance(tb_obj, list) else "n/a"
            logger.info(
                f"{phase} PLAYER_BOX_PAYLOAD game_id={game_id} game_date={game_date} "
                f"payload_type={type(bs_payload).__name__} "
                f"payload_keys={list(bs_payload.keys())[:20]} "
                f"teamBoxscore_exists={tb_exists} teamBoxscore_type={tb_type} "
                f"teamBoxscore_len={tb_len}"
            )
        else:
            logger.info(
                f"{phase} PLAYER_BOX_PAYLOAD game_id={game_id} game_date={game_date} "
                f"payload_type={type(bs_payload).__name__} payload_keys=[] "
                "teamBoxscore_exists=False teamBoxscore_type=missing teamBoxscore_len=n/a"
            )

        if not home_team_id:
            home_team_id = canonical_team_id(box_home_id) or _safe_team_id(box_home_id)
        if not away_team_id:
            away_team_id = canonical_team_id(box_away_id) or _safe_team_id(box_away_id)

        game_division = bs_payload.get("division")
        game_division_name = bs_payload.get("divisionName")

        home_team_id_canon = canonical_team_id(home_team_id)
        away_team_id_canon = canonical_team_id(away_team_id)

        if d1_team_meta is not None:
            home_meta_obj = box_home_obj or {
                "nameShort": home_name,
                "nameFull": home.get("names", {}).get("full") or home_name,
                "seoname": home.get("seoname"),
            }
            away_meta_obj = box_away_obj or {
                "nameShort": away_name,
                "nameFull": away.get("names", {}).get("full") or away_name,
                "seoname": away.get("seoname"),
            }
            _upsert_team_meta(
                d1_team_meta,
                home_team_id,
                home_meta_obj,
                source=f"boxscore:{game_id}",
                division=game_division,
                division_name=game_division_name,
            )
            _upsert_team_meta(
                d1_team_meta,
                away_team_id,
                away_meta_obj,
                source=f"boxscore:{game_id}",
                division=game_division,
                division_name=game_division_name,
            )

        home_is_d1 = bool(d1_team_ids is not None and home_team_id_canon in d1_team_ids)
        away_is_d1 = bool(d1_team_ids is not None and away_team_id_canon in d1_team_ids)

        if player_rows_sink is not None and isinstance(bs_payload, dict) and bs_payload:
            game_meta = {
                "game_id": game_id,
                "game_date": game_date,
                "home_team_id": home_team_id_canon or home_team_id,
                "away_team_id": away_team_id_canon or away_team_id,
                "home_name": home_name,
                "away_name": away_name,
            }
            try:
                extracted = parse_player_rows_from_boxscore(
                    bs_payload, game_meta, logger=logger
                )
                player_rows_sink.extend(extracted)
                logger.info(
                    f"PLAYER_EXTRACT game_id={game_id} extracted_rows={len(extracted)}"
                )
            except Exception as ex:
                logger.warning(
                    f"{phase} player extract failed game_id={game_id} "
                    f"error={type(ex).__name__}: {ex}"
                )

        home_stats = normalize_team_stats(home_raw_stats)
        away_stats = normalize_team_stats(away_raw_stats)

        # Conferences (best effort)
        def conf_name(team_obj: dict) -> str:
            confs = team_obj.get("conferences", [])
            if isinstance(confs, list) and len(confs) > 0 and isinstance(confs[0], dict):
                return confs[0].get("conferenceName", "") or confs[0].get("name", "")
            return team_obj.get("conference", "") if isinstance(team_obj, dict) else ""

        home_conf = conf_name(home)
        away_conf = conf_name(away)

        if home_name and away_name:
            rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "team_id": home_team_id_canon or home_team_id,
                "opponent_team_id": away_team_id_canon or away_team_id,
                "team": home_name,
                "opponent": away_name,
                "location": "H",
                "pts_for": home_score,
                "pts_against": away_score,
                **home_stats,
                "opp_fga": away_stats["fga"],
                "opp_fta": away_stats["fta"],
                "opp_orb": away_stats["orb"],
                "opp_to":  away_stats["to"],
                "team_conf": home_conf,
                "opp_conf": away_conf,
                "is_d1_team": home_is_d1,
                "is_d1_opponent": away_is_d1,
            })

            rows.append({
                "game_id": game_id,
                "game_date": game_date,
                "team_id": away_team_id_canon or away_team_id,
                "opponent_team_id": home_team_id_canon or home_team_id,
                "team": away_name,
                "opponent": home_name,
                "location": "A",
                "pts_for": away_score,
                "pts_against": home_score,
                **away_stats,
                "opp_fga": home_stats["fga"],
                "opp_fta": home_stats["fta"],
                "opp_orb": home_stats["orb"],
                "opp_to":  home_stats["to"],
                "team_conf": away_conf,
                "opp_conf": home_conf,
                "is_d1_team": away_is_d1,
                "is_d1_opponent": home_is_d1,
            })

        if progress is not None and progress.get("pbar") is not None:
            progress["games_done"] = progress.get("games_done", 0) + 1
            pbar = progress["pbar"]
            pbar.update(1)
            rate = pbar.format_dict.get("rate", None)
            remaining = (pbar.total - pbar.n)
            if rate and rate > 0:
                eta_seconds = remaining / rate
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "?"
            day_label = actual_day.isoformat() if actual_day else "unknown-day"
            pbar.set_postfix_str(f"{day_label} | {pbar.n}/{pbar.total} | ETA {eta_str}")
    if not rows:
        return _empty_games_long_df()

    df = pd.DataFrame(rows)
    logger.info(f"Fetched {len(df)} team-games (finals) with boxscores where available")
    return df

def possessions(fga, fta, orb, to):
    return fga - orb + to + 0.475 * fta

def time_weight(game_date: pd.Timestamp, decay_factor_days: float, asof: pd.Timestamp) -> float:
    if decay_factor_days <= 0:
        return 1.0
    delta = max(int((asof - game_date).days), 0)
    return math.exp(-(delta / decay_factor_days))

def build_design_matrix(games: pd.DataFrame, team_col: str = "team", opp_col: str = "opponent"):
    teams = pd.Index(sorted(pd.unique(games[team_col])))
    team_to_i = {t:i for i,t in enumerate(teams)}
    n = len(teams)
    p = 1 + n + n + 1
    X = np.zeros((len(games), p), dtype=float)
    target_col = "ppp_model" if "ppp_model" in games.columns else "ppp_for"
    y = games[target_col].to_numpy(dtype=float)
    w = games["w"].to_numpy(dtype=float)
    X[:,0] = 1.0
    for r,(t,opp,loc) in enumerate(zip(games[team_col], games[opp_col], games["loc_sign"])):
        ti = team_to_i[t]
        oi = team_to_i[opp]
        X[r, 1 + ti] = 1.0
        X[r, 1 + n + oi] = -1.0
        X[r, -1] = loc
    return X, y, w, teams

def weighted_ridge(X, y, w, lam):
    W = w.reshape(-1,1)
    Xw = X * np.sqrt(W)
    yw = y * np.sqrt(w)
    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw
    I = np.eye(X.shape[1])
    I[0,0] = 0.0
    return np.linalg.solve(XtX + lam*I, Xty)

def compute_ratings(
    games: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
    asof: pd.Timestamp | datetime | date | None = None,
) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["Rank","team_id","team","AdjOE","AdjDE","AdjEM","Pace","Games"])

    games = games.copy()
    logger.info(f"CHECK compute_ratings input rows={len(games)}")

    required = [
        "pts_for","pts_against","fga","fta","orb","to",
        "opp_fga","opp_fta","opp_orb","opp_to",
        "location","team","opponent","game_date"
    ]

    missing = [c for c in required if c not in games.columns]
    logger.info(f"CHECK missing columns={missing}")

    present = [c for c in required if c in games.columns]
    if present:
        na_counts = games[present].isna().sum()
        logger.info("CHECK NA counts:\n" + na_counts.to_string())

    games["poss_team"] = possessions(games["fga"], games["fta"], games["orb"], games["to"])
    # Estimate game possessions using both teams’ stats (more stable)
    opp_poss = possessions(
        games["opp_fga"], games["opp_fta"], games["opp_orb"], games["opp_to"]
    )
    games["poss"] = 0.5 * (games["poss_team"] + opp_poss)
    games["poss"] = pd.to_numeric(games["poss"], errors="coerce").fillna(0)
    games["poss"] = games["poss"].clip(lower=45, upper=90)
    games["ppp_for"] = games["pts_for"] / games["poss"].replace(0, np.nan)
    games["ppp_for"] = games["ppp_for"].fillna(0)

    loc_map = {"H": 1.0, "A": -1.0, "N": 0.0}
    games["loc_sign"] = games["location"].map(loc_map).fillna(0.0)

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    decay_factor_raw = model_cfg.get(
        "recency_decay_factor_days",
        model_cfg.get("recency_half_life_days", DEFAULT_RECENCY_DECAY_FACTOR_DAYS),
    )
    blowout_cap_raw = model_cfg.get("blowout_cap_em_per100", 25.0)
    try:
        decay_factor = float(decay_factor_raw)
    except Exception:
        decay_factor = DEFAULT_RECENCY_DECAY_FACTOR_DAYS
    if decay_factor <= 0:
        decay_factor = DEFAULT_RECENCY_DECAY_FACTOR_DAYS
    try:
        blowout_cap = float(blowout_cap_raw)
    except Exception:
        blowout_cap = 25.0
    if blowout_cap <= 0:
        blowout_cap = 25.0
    if asof is None:
        asof_ts = pd.Timestamp.now(tz=NY).normalize()
    else:
        asof_ts = pd.Timestamp(asof)
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize(NY)
        else:
            asof_ts = asof_ts.tz_convert(NY)
        asof_ts = asof_ts.normalize()
    
    # Parse dates safely
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

    # Ensure timezone-aware NY dates
    if games["game_date"].dt.tz is None:
        games["game_date"] = games["game_date"].dt.tz_localize(
            NY, ambiguous="NaT", nonexistent="NaT"
        )
    else:
        games["game_date"] = games["game_date"].dt.tz_convert(NY)

    bad_dates = games["game_date"].isna().sum()
    logger.info(f"CHECK bad game_date rows={bad_dates}/{len(games)}")
    logger.info(
        f"CHECK recency_decay_factor_days={decay_factor:.4f} "
        f"blowout_cap_em_per100={blowout_cap:.4f} "
        f"asof={asof_ts.date()}"
    )

    games = games.dropna(subset=["game_date"])

    games["w_time"] = games["game_date"].apply(
        lambda d: time_weight(d, decay_factor, asof_ts)
    )
    games["w"] = games["w_time"] * games["poss"]
    games["ppp_against"] = games["pts_against"] / games["poss"].replace(0, np.nan)
    games["ppp_against"] = games["ppp_against"].fillna(0)
    games["em_per100_raw"] = 100.0 * (games["ppp_for"] - games["ppp_against"])
    games["em_per100_capped"] = games["em_per100_raw"].clip(
        lower=-blowout_cap,
        upper=blowout_cap,
    )
    games["ppp_model"] = (
        0.5 * (games["ppp_for"] + games["ppp_against"])
        + (games["em_per100_capped"] / 200.0)
    )

    bad_w = (games["w"].isna() | (games["w"] <= 0)).sum()
    logger.info(f"CHECK bad weights rows={bad_w}/{len(games)}")

    games = games[~(games["w"].isna() | (games["w"] <= 0))]
    logger.info(f"CHECK rows after weight filter={len(games)}")

    # Use stable team IDs for model keys when present to avoid split identities
    # from short/full name variation.
    use_team_ids = "team_id" in games.columns and "opponent_team_id" in games.columns
    if use_team_ids:
        games["team_key"] = games["team_id"].apply(_safe_team_id)
        games["opp_key"] = games["opponent_team_id"].apply(_safe_team_id)
    else:
        games["team_key"] = ""
        games["opp_key"] = ""

    if "team" in games.columns:
        games.loc[games["team_key"] == "", "team_key"] = (
            games.loc[games["team_key"] == "", "team"].astype(str).map(normalize_name)
        )
    if "opponent" in games.columns:
        games.loc[games["opp_key"] == "", "opp_key"] = (
            games.loc[games["opp_key"] == "", "opponent"].astype(str).map(normalize_name)
        )

    before_keys = len(games)
    games = games[(games["team_key"] != "") & (games["opp_key"] != "")]
    if len(games) != before_keys:
        logger.info(f"CHECK dropped rows missing team keys={before_keys-len(games)}")
    if games.empty:
        logger.warning("CHECK no rows left after team key normalization")
        return pd.DataFrame(columns=["Rank","team_id","team","AdjOE","AdjDE","AdjEM","Pace","Games"])

    name_by_key = {}
    if "team" in games.columns:
        name_by_key = (
            games.assign(team_name=games["team"].astype(str).str.strip())
            .groupby("team_key")["team_name"]
            .agg(lambda s: s.value_counts().index[0] if not s.empty else "")
            .to_dict()
        )

    X,y,w,teams = build_design_matrix(games, team_col="team_key", opp_col="opp_key")
    lam = float(cfg["model"]["ridge_lambda"])
    b = weighted_ridge(X,y,w,lam)

    n = len(teams)
    intercept = b[0]
    O = b[1:1+n]
    D = b[1+n:1+2*n]

    AdjOE = 100.0 * (intercept + O)
    AdjDE = 100.0 * (intercept - D)
    AdjEM = AdjOE - AdjDE

    pace_weighted_num = (
        (games["poss"] * games["w_time"])
        .groupby(games["team_key"])
        .sum()
        .reindex(teams)
    )
    pace_weighted_den = (
        games.groupby("team_key")["w_time"]
        .sum()
        .replace(0, np.nan)
        .reindex(teams)
    )
    pace = (pace_weighted_num / pace_weighted_den).fillna(
        games.groupby("team_key")["poss"].mean().reindex(teams)
    ).to_numpy()
    gcount = games.groupby("team_key").size().reindex(teams).fillna(0).to_numpy()

    out = pd.DataFrame({
        "team_id": teams if use_team_ids else ["" for _ in teams],
        "team": [name_by_key.get(t, t) for t in teams],
        "AdjOE": AdjOE,
        "AdjDE": AdjDE,
        "AdjEM": AdjEM,
        "Pace": pace,
        "Games": gcount
    }).sort_values("AdjEM", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out)+1))
    return out

def compute_sos(games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    if games.empty or ratings.empty:
        return pd.DataFrame(columns=["team","SOS_Power"])
    if "team_id" in ratings.columns and "team_id" in games.columns and "opponent_team_id" in games.columns:
        r = ratings.set_index("team_id")["AdjEM"].to_dict()
        g = games.copy()
        g["team_id"] = g["team_id"].apply(_safe_team_id)
        g["opponent_team_id"] = g["opponent_team_id"].apply(_safe_team_id)
        g["opp_em"] = g["opponent_team_id"].map(r)
        sos = g.groupby("team_id")["opp_em"].mean().reset_index().rename(columns={"opp_em": "SOS_Power"})
        if "team" in g.columns:
            team_names = (
                g.assign(team_name=g["team"].astype(str).str.strip())
                .groupby("team_id")["team_name"]
                .agg(lambda s: s.value_counts().index[0] if not s.empty else "")
                .reset_index()
                .rename(columns={"team_name": "team"})
            )
            sos = sos.merge(team_names, on="team_id", how="left")
        else:
            sos["team"] = sos["team_id"]
        return sos

    r = ratings.set_index("team")["AdjEM"].to_dict()
    g = games.copy()
    g["opp_em"] = g["opponent"].map(r)
    sos = g.groupby("team")["opp_em"].mean().reset_index().rename(columns={"opp_em":"SOS_Power"})
    return sos


PREDICTION_OUTPUT_COLUMNS = [
    "date",
    "team",
    "opponent",
    "win_prob",
    "projected_spread",
    "projected_score_team",
    "projected_score_opp",
    "vegas_win_prob",
    "vegas_spread",
    "vegas_provider",
]
PREDICTION_LOG_COLUMNS = PREDICTION_OUTPUT_COLUMNS + [
    "actual_result",
    "actual_margin",
    "closing_vegas_win_prob",
    "closing_vegas_spread",
    "closing_vegas_provider",
]
PREDICTION_LOG_DEFAULTS: dict[str, object] = {
    "date": "",
    "team": "",
    "opponent": "",
    "win_prob": np.nan,
    "projected_spread": np.nan,
    "projected_score_team": np.nan,
    "projected_score_opp": np.nan,
    "vegas_win_prob": np.nan,
    "vegas_spread": np.nan,
    "vegas_provider": "",
    "actual_result": pd.NA,
    "actual_margin": np.nan,
    "closing_vegas_win_prob": np.nan,
    "closing_vegas_spread": np.nan,
    "closing_vegas_provider": "",
}
BET_EDGE_THRESHOLD = float(_BETTING_CFG.get("edge_threshold", 0.08))
BET_EDGE_MAX_ABS = float(_BETTING_CFG.get("max_abs_edge", 0.12))
BET_KELLY_SCALE = float(_BETTING_CFG.get("kelly_scale", 0.10))
PREDICTION_CALIBRATION_MIN_ROWS = 200
PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS = 50
PREDICTION_CALIBRATION_RECENT_ROWS = 100
PREDICTION_CALIBRATION_FALLBACK_ROWS = 250
PREDICTION_CALIBRATION_EPS = 1e-6
PREDICTION_ADJUSTMENT_MIN_ROWS = 200
PREDICTION_ADJUSTMENT_VALIDATION_MIN_ROWS = 50
PREDICTION_ADJUSTMENT_CACHE_MAX_AGE_DAYS = 1
PREDICTION_ADJUSTMENT_NEW_ROWS_REFIT_THRESHOLD = 25
_PREDICTION_CALIBRATION_CACHE: dict[tuple[str, int, str], dict[str, object]] = {}
_PREDICTION_ADJUSTMENT_CACHE: dict[tuple[str, int, str], dict[str, object]] = {}
_PREDICTION_ADJUSTMENT_BASE_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_TORVIK_RATINGS_CACHE: dict[int, pd.DataFrame] = {}
_VEGAS_LINES_CACHE: dict[str, pd.DataFrame] = {}


_PREDICTION_NAME_CANONICAL: dict[str, str] = {}
for _canonical_name, _variants in MASTER_TEAM_NAME_OVERRIDES.items():
    _PREDICTION_NAME_CANONICAL[_canonical_name] = _canonical_name
    for _variant in _variants:
        _PREDICTION_NAME_CANONICAL[_variant] = _canonical_name
for _canonical_name, _variants in WHITELIST_ALIAS_NORMALIZATION.items():
    _PREDICTION_NAME_CANONICAL[_canonical_name] = _canonical_name
    for _variant in _variants:
        _PREDICTION_NAME_CANONICAL[_variant] = _canonical_name
_PREDICTION_NAME_CANONICAL.update(
    {
        "mt state marys": "mount state marys",
        "mount saint marys": "mount state marys",
        "mount state marys": "mount state marys",
        "prairie view am": "prairie view",
        "prairie view a and m": "prairie view",
    }
)


def _prediction_logger(logger: logging.Logger | None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger("ncaab_ranker")


def _clip_probability_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.5)
    return np.clip(arr, PREDICTION_CALIBRATION_EPS, 1.0 - PREDICTION_CALIBRATION_EPS)


def _apply_vegas_probability_shrinkage(
    model_probs,
    vegas_probs,
) -> tuple[np.ndarray, dict[str, float | int]]:
    model_arr = _clip_probability_array(model_probs)
    vegas_arr = _clip_probability_array(vegas_probs)
    edge = model_arr - vegas_arr
    abs_edge = np.abs(edge)

    weights = np.ones_like(model_arr, dtype=float)
    medium_mask = (abs_edge >= 0.05) & (abs_edge < 0.15)
    large_mask = abs_edge >= 0.15
    weights[medium_mask] = 0.7
    weights[large_mask] = 0.5

    shrunk = weights * model_arr + (1.0 - weights) * vegas_arr
    compressed = 0.5 + (shrunk - 0.5) * 0.9
    final_probs = _clip_probability_array(compressed)

    stats: dict[str, float | int] = {
        "matched_rows": int(len(model_arr)),
        "medium_edge_count": int(medium_mask.sum()),
        "large_edge_count": int(large_mask.sum()),
        "avg_shrink_applied": float(np.mean(np.abs(final_probs - model_arr))),
        "avg_edge": float(np.mean(edge)),
        "avg_abs_edge": float(np.mean(abs_edge)),
        "avg_compression_applied": float(np.mean(np.abs(final_probs - shrunk))),
    }
    return final_probs, stats


def _logit(values) -> np.ndarray:
    probs = _clip_probability_array(values)
    return np.log(probs / (1.0 - probs))


def _sigmoid(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def _identity_calibration_model(rows_used: int = 0) -> dict[str, object]:
    return {
        "method": "identity",
        "rows_used": int(rows_used),
    }


def _fit_isotonic_calibration(confidence, favorite_won) -> dict[str, object] | None:
    probs = _clip_probability_array(confidence)
    outcomes = np.asarray(favorite_won, dtype=float)
    if len(probs) < PREDICTION_CALIBRATION_MIN_ROWS:
        return None
    if len(np.unique(outcomes)) < 2:
        return None

    order = np.argsort(probs, kind="mergesort")
    probs = probs[order]
    outcomes = outcomes[order]
    uniq_probs, inv = np.unique(probs, return_inverse=True)
    weights = np.bincount(inv).astype(float)
    outcome_sums = np.bincount(inv, weights=outcomes).astype(float)
    avg_outcomes = outcome_sums / np.maximum(weights, 1.0)

    blocks: list[dict[str, float | int]] = []
    for idx, (x_value, y_value, weight_value) in enumerate(
        zip(uniq_probs, avg_outcomes, weights)
    ):
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": float(weight_value),
                "value": float(y_value),
                "x_min": float(x_value),
                "x_max": float(x_value),
            }
        )

    block_idx = 0
    while block_idx < len(blocks) - 1:
        if float(blocks[block_idx]["value"]) <= float(blocks[block_idx + 1]["value"]):
            block_idx += 1
            continue
        left = blocks[block_idx]
        right = blocks[block_idx + 1]
        merged_weight = float(left["weight"]) + float(right["weight"])
        merged_value = (
            float(left["value"]) * float(left["weight"])
            + float(right["value"]) * float(right["weight"])
        ) / max(merged_weight, 1.0)
        blocks[block_idx : block_idx + 2] = [
            {
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": merged_weight,
                "value": merged_value,
                "x_min": float(left["x_min"]),
                "x_max": float(right["x_max"]),
            }
        ]
        if block_idx > 0:
            block_idx -= 1

    fitted = np.empty_like(uniq_probs, dtype=float)
    for block in blocks:
        fitted[int(block["start"]) : int(block["end"]) + 1] = float(block["value"])

    return {
        "method": "isotonic",
        "rows_used": int(len(probs)),
        "x_thresholds": uniq_probs.astype(float).tolist(),
        "y_thresholds": fitted.astype(float).tolist(),
    }


def _fit_logistic_calibration(confidence, favorite_won) -> dict[str, object] | None:
    probs = _clip_probability_array(confidence)
    outcomes = np.asarray(favorite_won, dtype=float)
    if len(probs) < PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS:
        return None
    if len(np.unique(outcomes)) < 2:
        return None

    logits = _logit(probs)
    X = np.column_stack([np.ones(len(logits), dtype=float), logits])
    beta = np.zeros(2, dtype=float)
    ridge = 1e-6 * np.eye(2, dtype=float)

    for _ in range(100):
        eta = X @ beta
        mu = _sigmoid(eta)
        W = np.clip(mu * (1.0 - mu), 1e-8, None)
        XtWX = X.T @ (X * W[:, None]) + ridge
        grad = X.T @ (outcomes - mu)
        try:
            step = np.linalg.solve(XtWX, grad)
        except Exception:
            return None
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-8:
            break

    return {
        "method": "logistic",
        "rows_used": int(len(probs)),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
    }


def apply_prediction_calibration(win_prob, calibration_model: dict[str, object] | None):
    probs = _clip_probability_array(win_prob)
    if not calibration_model:
        return probs

    favorite_mask = probs >= 0.5
    confidence = np.where(favorite_mask, probs, 1.0 - probs)
    method = str(calibration_model.get("method", "identity")).strip().lower()
    if method == "isotonic":
        x_thresholds = np.asarray(
            calibration_model.get("x_thresholds", []),
            dtype=float,
        )
        y_thresholds = np.asarray(
            calibration_model.get("y_thresholds", []),
            dtype=float,
        )
        if len(x_thresholds) == 0 or len(y_thresholds) == 0:
            return probs
        calibrated_conf = np.interp(
            confidence,
            x_thresholds,
            y_thresholds,
            left=float(y_thresholds[0]),
            right=float(y_thresholds[-1]),
        )
        calibrated = np.where(favorite_mask, calibrated_conf, 1.0 - calibrated_conf)
        return _clip_probability_array(calibrated)
    if method == "logistic":
        intercept = float(calibration_model.get("intercept", 0.0))
        slope = float(calibration_model.get("slope", 1.0))
        calibrated_conf = _sigmoid(intercept + slope * _logit(confidence))
        calibrated = np.where(favorite_mask, calibrated_conf, 1.0 - calibrated_conf)
        return _clip_probability_array(calibrated)
    return probs


def _prediction_adjustment_metrics(probs, outcomes) -> dict[str, float]:
    clipped = _clip_probability_array(probs)
    actuals = np.asarray(outcomes, dtype=float)
    return {
        "brier": float(np.mean((clipped - actuals) ** 2)),
        "log_loss": float(
            (-(actuals * np.log(clipped) + (1.0 - actuals) * np.log(1.0 - clipped))).mean()
        ),
    }


def _probability_adjustment_features(
    model_prob,
    vegas_prob,
    projected_spread,
) -> np.ndarray:
    model_arr = _clip_probability_array(model_prob)
    vegas_arr = _clip_probability_array(vegas_prob)
    spread_arr = np.asarray(projected_spread, dtype=float)
    spread_arr = np.where(np.isfinite(spread_arr), spread_arr, 0.0)
    edge = model_arr - vegas_arr
    abs_edge = np.abs(edge)
    abs_spread = np.abs(spread_arr)
    return np.column_stack(
        [
            model_arr,
            vegas_arr,
            edge,
            abs_edge,
            spread_arr,
            abs_spread,
        ]
    )


def _fit_binary_logistic_regression(
    features,
    outcomes,
    ridge_lambda: float = 1e-2,
    max_iter: int = 100,
) -> dict[str, object] | None:
    X_raw = np.asarray(features, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if X_raw.ndim != 2 or len(X_raw) == 0:
        return None
    if len(np.unique(y)) < 2:
        return None

    feature_means = np.nanmean(X_raw, axis=0)
    feature_means = np.where(np.isfinite(feature_means), feature_means, 0.0)
    centered = X_raw - feature_means
    feature_scales = np.nanstd(centered, axis=0)
    feature_scales = np.where(np.isfinite(feature_scales) & (feature_scales > 1e-8), feature_scales, 1.0)
    X_scaled = centered / feature_scales
    X = np.column_stack([np.ones(len(X_scaled), dtype=float), X_scaled])

    beta = np.zeros(X.shape[1], dtype=float)
    ridge = np.eye(X.shape[1], dtype=float) * float(ridge_lambda)
    ridge[0, 0] = 1e-8

    for _ in range(max_iter):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            eta = np.clip(X @ beta, -30.0, 30.0)
            mu = _sigmoid(eta)
            W = np.clip(mu * (1.0 - mu), 1e-8, None)
            XtWX = X.T @ (X * W[:, None]) + ridge
            grad = X.T @ (y - mu) - ridge @ beta
        grad[0] += ridge[0, 0] * beta[0]
        try:
            step = np.linalg.solve(XtWX, grad)
        except Exception:
            return None
        if not np.all(np.isfinite(step)):
            return None
        beta = np.clip(beta + step, -25.0, 25.0)
        if float(np.max(np.abs(step))) < 1e-8:
            break

    return {
        "method": "logistic",
        "rows_used": int(len(X_raw)),
        "feature_names": [
            "model_prob",
            "vegas_prob",
            "edge",
            "abs_edge",
            "projected_spread",
            "abs_projected_spread",
        ],
        "intercept": float(beta[0]),
        "coefficients": beta[1:].astype(float).tolist(),
        "feature_means": feature_means.astype(float).tolist(),
        "feature_scales": feature_scales.astype(float).tolist(),
    }


def _predict_binary_logistic_regression(
    features,
    model: dict[str, object] | None,
) -> np.ndarray:
    X_raw = np.asarray(features, dtype=float)
    if model is None or X_raw.size == 0:
        return _clip_probability_array(np.full(len(X_raw), 0.5, dtype=float))

    coefs = np.asarray(model.get("coefficients", []), dtype=float)
    means = np.asarray(model.get("feature_means", []), dtype=float)
    scales = np.asarray(model.get("feature_scales", []), dtype=float)
    intercept = float(model.get("intercept", 0.0))
    if (
        X_raw.ndim != 2
        or len(coefs) != X_raw.shape[1]
        or len(means) != X_raw.shape[1]
        or len(scales) != X_raw.shape[1]
    ):
        return _clip_probability_array(np.full(len(X_raw), 0.5, dtype=float))

    safe_scales = np.where(np.abs(scales) > 1e-8, scales, 1.0)
    X_scaled = (X_raw - means) / safe_scales
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eta = np.clip(intercept + X_scaled @ coefs, -30.0, 30.0)
        return _clip_probability_array(_sigmoid(eta))


def _load_prediction_adjustment_base_rows(
    predictions_log_path: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = _prediction_logger(logger)
    resolved_path = os.path.abspath(predictions_log_path)
    if not os.path.exists(resolved_path):
        return pd.DataFrame()
    try:
        mtime_ns = os.stat(resolved_path).st_mtime_ns
    except Exception:
        mtime_ns = 0
    cache_key = (resolved_path, int(mtime_ns))
    cached = _PREDICTION_ADJUSTMENT_BASE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    pred_log_df = _ensure_predictions_log_schema(resolved_path, logger)
    if pred_log_df is None or pred_log_df.empty:
        empty = pd.DataFrame()
        _PREDICTION_ADJUSTMENT_BASE_CACHE[cache_key] = empty
        return empty.copy()

    base_df = pred_log_df.copy()
    base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce").dt.date
    base_df["win_prob"] = pd.to_numeric(base_df["win_prob"], errors="coerce")
    base_df["actual_result"] = pd.to_numeric(base_df["actual_result"], errors="coerce")
    base_df["projected_spread"] = pd.to_numeric(base_df["projected_spread"], errors="coerce")
    base_df = base_df[
        base_df["date"].notna()
        & base_df["team"].notna()
        & base_df["opponent"].notna()
        & base_df["win_prob"].notna()
        & base_df["actual_result"].notna()
        & base_df["projected_spread"].notna()
    ].copy()
    if base_df.empty:
        _PREDICTION_ADJUSTMENT_BASE_CACHE[cache_key] = base_df
        return base_df.copy()

    base_df = base_df.drop_duplicates(
        subset=["date", "team", "opponent"],
        keep="last",
    ).sort_values(["date", "team", "opponent"]).reset_index(drop=True)
    base_df["team_name_clean"] = base_df["team"].fillna("").astype(str).apply(clean_team_name)
    base_df["opponent_name_clean"] = (
        base_df["opponent"].fillna("").astype(str).apply(clean_team_name)
    )
    base_df = base_df.rename(columns={"win_prob": "model_prob"})
    _PREDICTION_ADJUSTMENT_BASE_CACHE[cache_key] = base_df.copy()
    return base_df


def _build_prediction_adjustment_training_rows(
    predictions_log_path: str,
    logger: logging.Logger | None = None,
    asof_date: date | None = None,
) -> pd.DataFrame:
    logger = _prediction_logger(logger)
    cutoff_day = asof_date or datetime.now(tz=NY).date()
    base_df = _load_prediction_adjustment_base_rows(predictions_log_path, logger)
    if base_df.empty:
        return pd.DataFrame()

    train_df = base_df.loc[base_df["date"] < cutoff_day].copy()
    if train_df.empty:
        return pd.DataFrame()

    vegas_frames: list[pd.DataFrame] = []
    for day_value in sorted(train_df["date"].dropna().unique()):
        try:
            vegas_df = fetch_vegas_lines(day_value, logger=logger)
        except Exception as ex:
            logger.info(
                "PREDICTIONS adjustment_vegas_failed "
                f"day={day_value} error={type(ex).__name__}: {ex}"
            )
            continue
        if vegas_df is None or vegas_df.empty:
            continue
        merged_day = vegas_df[
            ["date", "team_name_clean", "opponent_name_clean", "vegas_win_prob"]
        ].copy()
        merged_day["date"] = pd.to_datetime(merged_day["date"], errors="coerce").dt.date
        merged_day["vegas_win_prob"] = pd.to_numeric(
            merged_day["vegas_win_prob"], errors="coerce"
        )
        merged_day = merged_day.dropna(subset=["date", "vegas_win_prob"]).copy()
        vegas_frames.append(merged_day)

    if not vegas_frames:
        return pd.DataFrame()

    vegas_all = (
        pd.concat(vegas_frames, ignore_index=True)
        .drop_duplicates(
            subset=["date", "team_name_clean", "opponent_name_clean"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    train_df = train_df.merge(
        vegas_all,
        on=["date", "team_name_clean", "opponent_name_clean"],
        how="left",
    )
    train_df["vegas_win_prob"] = pd.to_numeric(train_df["vegas_win_prob"], errors="coerce")
    train_df = train_df.dropna(subset=["vegas_win_prob"]).copy()
    if train_df.empty:
        return train_df

    train_df["model_prob"] = _clip_probability_array(train_df["model_prob"].to_numpy(dtype=float))
    train_df["vegas_win_prob"] = _clip_probability_array(
        train_df["vegas_win_prob"].to_numpy(dtype=float)
    )
    train_df["edge"] = train_df["model_prob"] - train_df["vegas_win_prob"]
    train_df["abs_edge"] = train_df["edge"].abs()
    train_df["abs_projected_spread"] = train_df["projected_spread"].abs()
    return train_df.reset_index(drop=True)


def _prediction_adjustment_model_path(project_root: str | None = None) -> str:
    root = project_root or _default_project_root()
    return os.path.join(root, "outputs", "calibration_model.pkl")


def _load_prediction_adjustment_model_from_disk(
    model_path: str,
    logger: logging.Logger | None = None,
) -> dict[str, object] | None:
    logger = _prediction_logger(logger)
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as ex:
        logger.info(
            "PREDICTIONS adjustment_cache_load_failed "
            f"path={os.path.abspath(model_path)} error={type(ex).__name__}: {ex}"
        )
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if not isinstance(model, dict):
        return None
    payload = dict(payload)
    payload["model"] = dict(model)
    return payload


def _save_prediction_adjustment_model_to_disk(
    model_path: str,
    payload: dict[str, object],
    logger: logging.Logger | None = None,
) -> None:
    logger = _prediction_logger(logger)
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as fh:
            pickle.dump(payload, fh)
    except Exception as ex:
        logger.info(
            "PREDICTIONS adjustment_cache_save_failed "
            f"path={os.path.abspath(model_path)} error={type(ex).__name__}: {ex}"
        )
        return
    logger.info(
        "PREDICTIONS adjustment_cache_saved "
        f"path={os.path.abspath(model_path)} "
        f"trained_asof_date={payload.get('trained_asof_date')} "
        f"training_rows={payload.get('training_rows')}"
    )


def apply_prediction_probability_adjustment(
    model_prob,
    vegas_prob,
    projected_spread,
    adjustment_model: dict[str, object] | None,
):
    model_arr = _clip_probability_array(model_prob)
    vegas_arr = _clip_probability_array(vegas_prob)
    spread_arr = np.asarray(projected_spread, dtype=float)
    if not adjustment_model:
        return model_arr

    method = str(adjustment_model.get("method", "identity")).strip().lower()
    if method == "logistic":
        features = _probability_adjustment_features(model_arr, vegas_arr, spread_arr)
        return _predict_binary_logistic_regression(features, adjustment_model)
    if method == "shrinkage":
        adjusted, _ = _apply_vegas_probability_shrinkage(model_arr, vegas_arr)
        return adjusted
    return model_arr


def fit_prediction_probability_adjustment_model(
    predictions_log_path: str | None = None,
    logger: logging.Logger | None = None,
    asof_date: date | None = None,
    cache_path: str | None = None,
    force_refit: bool = False,
) -> dict[str, object]:
    logger = _prediction_logger(logger)
    project_root = _default_project_root()
    resolved_path = predictions_log_path or os.path.join(
        project_root,
        "outputs",
        "predictions_log.csv",
    )
    resolved_path = os.path.abspath(resolved_path)
    cutoff_day = asof_date or datetime.now(tz=NY).date()
    resolved_cache_path = os.path.abspath(
        cache_path or _prediction_adjustment_model_path(project_root)
    )

    if not os.path.exists(resolved_path):
        logger.info(
            "PREDICTIONS adjustment_skipped "
            f"reason=missing_predictions_log path={resolved_path}"
        )
        return {"method": "identity", "rows_used": 0}

    base_rows = _load_prediction_adjustment_base_rows(resolved_path, logger)
    eligible_rows = (
        base_rows.loc[base_rows["date"] < cutoff_day].copy()
        if not base_rows.empty
        else pd.DataFrame()
    )
    current_settled_rows = int(len(eligible_rows))

    try:
        mtime_ns = os.stat(resolved_path).st_mtime_ns
    except Exception:
        mtime_ns = 0
    cache_key = (resolved_path, int(mtime_ns), cutoff_day.isoformat(), resolved_cache_path)
    cached = _PREDICTION_ADJUSTMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not force_refit:
        disk_payload = _load_prediction_adjustment_model_from_disk(
            resolved_cache_path,
            logger=logger,
        )
        if disk_payload is not None:
            disk_model = disk_payload.get("model")
            trained_asof_raw = disk_payload.get("trained_asof_date")
            trained_rows = int(disk_payload.get("training_rows", 0) or 0)
            cached_settled_rows = int(
                disk_payload.get("settled_rows", trained_rows) or trained_rows
            )
            trained_asof = pd.to_datetime(trained_asof_raw, errors="coerce")
            trained_asof_date = trained_asof.date() if pd.notna(trained_asof) else None
            age_days = (
                (cutoff_day - trained_asof_date).days
                if trained_asof_date is not None
                else PREDICTION_ADJUSTMENT_CACHE_MAX_AGE_DAYS + 1
            )
            new_rows = max(current_settled_rows - cached_settled_rows, 0)
            if (
                isinstance(disk_model, dict)
                and trained_asof_date is not None
                and trained_asof_date <= cutoff_day
                and age_days <= PREDICTION_ADJUSTMENT_CACHE_MAX_AGE_DAYS
                and new_rows <= PREDICTION_ADJUSTMENT_NEW_ROWS_REFIT_THRESHOLD
            ):
                logger.info(
                    "PREDICTIONS adjustment_cache_loaded "
                    f"path={resolved_cache_path} trained_asof_date={trained_asof_date} "
                    f"training_rows={trained_rows} age_days={age_days} new_rows={new_rows}"
                )
                cached_model = dict(disk_model)
                _PREDICTION_ADJUSTMENT_CACHE[cache_key] = cached_model
                return cached_model

    train_df = _build_prediction_adjustment_training_rows(
        resolved_path,
        logger=logger,
        asof_date=cutoff_day,
    )
    if len(train_df) < PREDICTION_ADJUSTMENT_MIN_ROWS or train_df["actual_result"].nunique() < 2:
        model = {"method": "identity", "rows_used": int(len(train_df))}
        logger.info(
            "PREDICTIONS adjustment_skipped "
            f"reason=insufficient_training_rows rows={len(train_df)} asof={cutoff_day}"
        )
        _PREDICTION_ADJUSTMENT_CACHE[cache_key] = model
        return model

    train_df = train_df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)
    validation_size = 0
    if len(train_df) >= 2 * PREDICTION_ADJUSTMENT_VALIDATION_MIN_ROWS:
        validation_size = max(
            int(len(train_df) * 0.2),
            PREDICTION_ADJUSTMENT_VALIDATION_MIN_ROWS,
        )
        validation_size = min(validation_size, len(train_df) // 2)

    if validation_size > 0:
        fit_df = train_df.iloc[:-validation_size].copy()
        valid_df = train_df.iloc[-validation_size:].copy()
    else:
        fit_df = train_df.copy()
        valid_df = train_df.copy()

    fit_features = _probability_adjustment_features(
        fit_df["model_prob"].to_numpy(dtype=float),
        fit_df["vegas_win_prob"].to_numpy(dtype=float),
        fit_df["projected_spread"].to_numpy(dtype=float),
    )
    logistic_model = _fit_binary_logistic_regression(
        fit_features,
        fit_df["actual_result"].to_numpy(dtype=float),
    )

    valid_actuals = valid_df["actual_result"].to_numpy(dtype=float)
    identity_metrics = _prediction_adjustment_metrics(
        valid_df["model_prob"].to_numpy(dtype=float),
        valid_actuals,
    )
    shrinkage_probs, _ = _apply_vegas_probability_shrinkage(
        valid_df["model_prob"].to_numpy(dtype=float),
        valid_df["vegas_win_prob"].to_numpy(dtype=float),
    )
    shrinkage_metrics = _prediction_adjustment_metrics(
        shrinkage_probs,
        valid_actuals,
    )
    validation_scores = {
        "identity": identity_metrics,
        "shrinkage": shrinkage_metrics,
    }

    best_method = min(
        validation_scores,
        key=lambda key: (validation_scores[key]["brier"], validation_scores[key]["log_loss"]),
    )
    selected_model: dict[str, object] = {
        "method": best_method,
        "rows_used": int(len(train_df)),
    }

    if logistic_model is not None:
        logistic_valid = apply_prediction_probability_adjustment(
            valid_df["model_prob"].to_numpy(dtype=float),
            valid_df["vegas_win_prob"].to_numpy(dtype=float),
            valid_df["projected_spread"].to_numpy(dtype=float),
            logistic_model,
        )
        logistic_metrics = _prediction_adjustment_metrics(logistic_valid, valid_actuals)
        validation_scores["logistic"] = logistic_metrics
        logistic_best = min(
            validation_scores,
            key=lambda key: (validation_scores[key]["brier"], validation_scores[key]["log_loss"]),
        )
        if logistic_best == "logistic":
            selected_model = dict(logistic_model)
            selected_model["rows_used"] = int(len(train_df))
            selected_model["validation_brier"] = float(logistic_metrics["brier"])
            selected_model["validation_log_loss"] = float(logistic_metrics["log_loss"])
        else:
            selected_model["validation_brier"] = float(validation_scores[best_method]["brier"])
            selected_model["validation_log_loss"] = float(validation_scores[best_method]["log_loss"])
    else:
        selected_model["validation_brier"] = float(validation_scores[best_method]["brier"])
        selected_model["validation_log_loss"] = float(validation_scores[best_method]["log_loss"])

    logger.info(
        "PREDICTIONS adjustment_fit "
        f"asof={cutoff_day} rows={len(train_df)} validation_rows={len(valid_df)} "
        f"scores={validation_scores} chosen={selected_model.get('method', 'identity')}"
    )
    cache_payload = {
        "trained_asof_date": cutoff_day.isoformat(),
        "training_rows": int(len(train_df)),
        "settled_rows": int(len(eligible_rows)),
        "newest_training_date": (
            str(train_df["date"].max()) if not train_df.empty else None
        ),
        "predictions_log_path": resolved_path,
        "model": dict(selected_model),
    }
    _save_prediction_adjustment_model_to_disk(
        resolved_cache_path,
        cache_payload,
        logger=logger,
    )
    _PREDICTION_ADJUSTMENT_CACHE[cache_key] = selected_model
    return selected_model


def fit_prediction_calibration_model(
    predictions_log_path: str | None = None,
    logger: logging.Logger | None = None,
    asof_date: date | None = None,
) -> dict[str, object]:
    logger = _prediction_logger(logger)
    project_root = _default_project_root()
    resolved_path = predictions_log_path or os.path.join(
        project_root,
        "outputs",
        "predictions_log.csv",
    )
    resolved_path = os.path.abspath(resolved_path)
    cutoff_day = asof_date or datetime.now(tz=NY).date()

    if not os.path.exists(resolved_path):
        logger.info(
            "PREDICTIONS calibration_skipped "
            f"reason=missing_predictions_log path={resolved_path}"
        )
        return _identity_calibration_model()

    try:
        mtime_ns = os.stat(resolved_path).st_mtime_ns
    except Exception:
        mtime_ns = 0
    cache_key = (resolved_path, int(mtime_ns), "settled_global")
    cached = _PREDICTION_CALIBRATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pred_log_df = _ensure_predictions_log_schema(resolved_path, logger)
    if pred_log_df is None or pred_log_df.empty:
        model = _identity_calibration_model()
        _PREDICTION_CALIBRATION_CACHE[cache_key] = model
        return model

    train_df = pred_log_df.copy()
    train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce").dt.date
    train_df["win_prob"] = pd.to_numeric(train_df["win_prob"], errors="coerce")
    train_df["actual_result"] = pd.to_numeric(train_df["actual_result"], errors="coerce")
    train_df = train_df[
        train_df["date"].notna()
        & train_df["win_prob"].notna()
        & train_df["actual_result"].notna()
    ].copy()
    train_df = train_df.drop_duplicates(
        subset=["date", "team", "opponent"],
        keep="last",
    )
    train_df = train_df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)
    settled_through = train_df["date"].max() if not train_df.empty else None

    if train_df.empty or train_df["actual_result"].nunique() < 2:
        model = _identity_calibration_model(rows_used=len(train_df))
        logger.info(
            "PREDICTIONS calibration_skipped "
            f"reason=insufficient_training_rows rows={len(train_df)} "
            f"asof_request={cutoff_day} settled_through={settled_through}"
        )
        _PREDICTION_CALIBRATION_CACHE[cache_key] = model
        return model

    calibration_df = train_df.tail(min(len(train_df), PREDICTION_CALIBRATION_RECENT_ROWS)).copy()
    calibration_window = len(calibration_df)
    if (
        len(calibration_df) < PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS
        or calibration_df["actual_result"].nunique() < 2
    ):
        calibration_df = train_df.tail(
            min(len(train_df), PREDICTION_CALIBRATION_FALLBACK_ROWS)
        ).copy()
        calibration_window = len(calibration_df)
    if (
        len(calibration_df) < PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS
        or calibration_df["actual_result"].nunique() < 2
    ):
        calibration_df = train_df.copy()
        calibration_window = len(calibration_df)

    if (
        calibration_df.empty
        or len(calibration_df) < PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS
        or calibration_df["actual_result"].nunique() < 2
    ):
        model = _identity_calibration_model(rows_used=len(calibration_df))
        logger.info(
            "PREDICTIONS calibration_skipped "
            f"reason=insufficient_recent_rows total_rows={len(train_df)} "
            f"recent_rows={len(calibration_df)} asof_request={cutoff_day} "
            f"settled_through={settled_through}"
        )
        _PREDICTION_CALIBRATION_CACHE[cache_key] = model
        return model

    def _calibration_arrays(frame: pd.DataFrame):
        raw_probs_local = _clip_probability_array(frame["win_prob"].to_numpy(dtype=float))
        actuals_local = frame["actual_result"].to_numpy(dtype=float)
        favorite_mask_local = raw_probs_local >= 0.5
        confidence_local = np.where(
            favorite_mask_local,
            raw_probs_local,
            1.0 - raw_probs_local,
        )
        favorite_won_local = np.where(
            favorite_mask_local,
            actuals_local,
            1.0 - actuals_local,
        )
        return raw_probs_local, actuals_local, confidence_local, favorite_won_local

    raw_probs, actuals, confidence, favorite_won = _calibration_arrays(calibration_df)
    chosen_method = "identity"
    validation_scores = {"identity": float(np.mean((raw_probs - actuals) ** 2))}
    validation_size = 0

    if len(calibration_df) >= 2 * PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS:
        validation_size = max(
            int(len(calibration_df) * 0.2),
            PREDICTION_CALIBRATION_FALLBACK_MIN_ROWS,
        )
        validation_size = min(validation_size, len(calibration_df) // 2)
        fit_df = calibration_df.iloc[:-validation_size].copy()
        valid_df = calibration_df.iloc[-validation_size:].copy()
        _, _, confidence_fit, favorite_won_fit = _calibration_arrays(fit_df)
        raw_holdout = _clip_probability_array(valid_df["win_prob"].to_numpy(dtype=float))
        actuals_holdout = valid_df["actual_result"].to_numpy(dtype=float)
        baseline_brier = float(np.mean((raw_holdout - actuals_holdout) ** 2))
        validation_scores["identity"] = baseline_brier

        iso_holdout_model = _fit_isotonic_calibration(confidence_fit, favorite_won_fit)
        if iso_holdout_model is not None:
            validation_scores["isotonic"] = float(
                np.mean(
                    (
                        apply_prediction_calibration(raw_holdout, iso_holdout_model)
                        - actuals_holdout
                    )
                    ** 2
                )
            )

        logistic_holdout_model = _fit_logistic_calibration(confidence_fit, favorite_won_fit)
        if logistic_holdout_model is not None:
            validation_scores["logistic"] = float(
                np.mean(
                    (
                        apply_prediction_calibration(raw_holdout, logistic_holdout_model)
                        - actuals_holdout
                    )
                    ** 2
                )
            )

        best_method = min(validation_scores, key=validation_scores.get)
        if (
            best_method != "identity"
            and validation_scores[best_method] < baseline_brier - 1e-4
        ):
            chosen_method = best_method
        logger.info(
            "PREDICTIONS calibration_select "
            f"asof_request={cutoff_day} settled_through={settled_through} "
            f"total_rows={len(train_df)} "
            f"recent_rows={len(calibration_df)} validation_rows={validation_size} "
            f"scores={validation_scores} chosen={chosen_method}"
        )

    if chosen_method == "isotonic":
        model = _fit_isotonic_calibration(confidence, favorite_won)
    elif chosen_method == "logistic":
        model = _fit_logistic_calibration(confidence, favorite_won)
    else:
        model = _identity_calibration_model(rows_used=len(train_df))
    if model is None:
        model = _identity_calibration_model(rows_used=len(train_df))

    calibrated_probs = apply_prediction_calibration(raw_probs, model)
    brier_before = float(np.mean((raw_probs - actuals) ** 2))
    brier_after = float(np.mean((calibrated_probs - actuals) ** 2))
    logger.info(
        "PREDICTIONS calibration_fit "
        f"method={model.get('method', 'identity')} total_rows={len(train_df)} "
        f"recent_rows={len(calibration_df)} window_rows={calibration_window} "
        f"asof_request={cutoff_day} settled_through={settled_through} "
        f"brier_before={brier_before:.4f} "
        f"brier_after={brier_after:.4f}"
    )
    _PREDICTION_CALIBRATION_CACHE[cache_key] = model
    return model


_TEAM_NAME_REMOVE_WORDS = sorted(
    {
        "mountain hawks",
        "golden bears",
        "golden flashes",
        "golden hurricane",
        "demon deacons",
        "wolf pack",
        "wolfpack",
        "red hawks",
        "redhawks",
        "redbirds",
        "bearkats",
        "revolutionaries",
        "midshipmen",
        "longhorns",
        "cowboys",
        "patriots",
        "bison",
        "shockers",
        "anteaters",
        "seahawks",
        "jaguars",
        "lumberjacks",
        "retrievers",
        "tommies",
        "mustangs",
        "lobos",
        "flames",
        "flyers",
        "braves",
        "rams",
        "wolverines",
        "racers",
        "wildcats",
        "tigers",
        "bulldogs",
        "eagles",
        "hawks",
        "panthers",
        "cougars",
        "bruins",
        "rebels",
        "knights",
        "aggress",
        "aggies",
        "bearcats",
        "terriers",
        "wolves",
        "raiders",
    },
    key=len,
    reverse=True,
)


_PREDICTION_CONTAINMENT_TOKEN_ALIASES = {
    "nc": "north carolina",
    "sc": "south carolina",
    "am": "a and m",
}


def clean_team_name(name) -> str:
    if not isinstance(name, str):
        return ""

    cleaned = normalize_team_name(name)
    for word in _TEAM_NAME_REMOVE_WORDS:
        cleaned = re.sub(rf"\b{re.escape(word)}\b", " ", cleaned)
    cleaned = normalize_team_name(cleaned)
    cleaned = apply_name_alias(cleaned)
    cleaned = {
        "prairie view a and m": "prairie view",
        "state thomasminnesota": "state thomas mn",
    }.get(cleaned, cleaned)
    return cleaned


def _prediction_containment_key(name) -> str:
    cleaned = clean_team_name(name)
    if not cleaned:
        return ""
    tokens: list[str] = []
    for token in cleaned.split():
        expanded = _PREDICTION_CONTAINMENT_TOKEN_ALIASES.get(token, token)
        tokens.extend(str(expanded).split())
    return " ".join(tokens)


def _prediction_containment_tokens(name) -> tuple[str, ...]:
    key = _prediction_containment_key(name)
    if not key:
        return ()
    return tuple(token for token in key.split() if token)


def _resolve_containment_name_matches(
    raw_names,
    candidate_names,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    candidates = [
        str(name).strip()
        for name in candidate_names
        if str(name).strip()
    ]
    candidate_key_map = {
        candidate: _prediction_containment_key(candidate)
        for candidate in candidates
    }
    candidate_token_map = {
        candidate: _prediction_containment_tokens(candidate)
        for candidate in candidates
    }
    resolved: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}

    for raw_name in sorted({str(name).strip() for name in raw_names if str(name).strip()}):
        raw_key = _prediction_containment_key(raw_name)
        raw_tokens = _prediction_containment_tokens(raw_name)
        if not raw_key:
            continue
        if len(raw_tokens) < 2:
            continue

        scored_matches: list[tuple[int, int, str]] = []
        raw_token_set = set(raw_tokens)
        for candidate, candidate_key in candidate_key_map.items():
            if not candidate_key:
                continue
            candidate_tokens = candidate_token_map.get(candidate, ())
            if len(candidate_tokens) < 2:
                continue
            candidate_token_set = set(candidate_tokens)
            shorter_len = min(len(raw_token_set), len(candidate_token_set))
            if shorter_len < 2:
                continue
            if not (
                raw_token_set.issubset(candidate_token_set)
                or candidate_token_set.issubset(raw_token_set)
            ):
                continue
            token_overlap = len(raw_token_set & candidate_token_set)
            token_gap = abs(len(raw_token_set) - len(candidate_token_set))
            scored_matches.append((token_overlap, -token_gap, candidate))

        if not scored_matches:
            continue
        scored_matches.sort(reverse=True)
        best_overlap, best_gap, best_candidate = scored_matches[0]
        tied = [
            candidate
            for overlap, gap, candidate in scored_matches
            if overlap == best_overlap and gap == best_gap
        ]
        if len(tied) == 1:
            resolved[raw_name] = best_candidate
        else:
            ambiguous[raw_name] = sorted(tied)[:10]

    return resolved, ambiguous


def _prediction_name_key(raw_name) -> str:
    norm = apply_name_alias(normalize_team_name(clean_team_name(raw_name)))
    if not norm:
        return ""
    norm = re.sub(r"\bmt\b", "mount", norm)
    norm = normalize_team_name(norm)
    norm = apply_name_alias(norm)
    canonical = _PREDICTION_NAME_CANONICAL.get(norm)
    if canonical:
        return canonical
    return norm


def _normalized_colname(col) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col or "").strip().lower())


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lookup = {_normalized_colname(col): col for col in df.columns}
    candidate_keys = [_normalized_colname(c) for c in candidates if _normalized_colname(c)]
    for key in candidate_keys:
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(
            f"Missing expected column. candidates={candidates} available={list(df.columns)}"
        )
    return None


def _empty_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_OUTPUT_COLUMNS)


class _TorvikTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.skip_depth = 0
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "table" and not self.in_table:
            self.in_table = True
            return
        if not self.in_table:
            return
        if tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"td", "th"} and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "table" and self.in_table:
            self.in_table = False
            return
        if not self.in_table:
            return
        if tag in {"td", "th"} and self.in_cell:
            text = " ".join(" ".join(self.current_cell).split())
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if any(cell.strip() for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = []
            self.in_row = False

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_cell:
            self.current_cell.append(data)


def _parse_leading_float(raw_value) -> float | None:
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", str(raw_value or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _clean_torvik_team_name(raw_name) -> str:
    name = str(raw_name or "").strip()
    if not name:
        return ""
    name = re.sub(r"\s+\d+\s*seed.*$", "", name, flags=re.I)
    name = re.sub(r"\s*[✅❌].*$", "", name).strip()
    if "," in name:
        name = name.split(",", 1)[0].strip()
    return name


def _empty_torvik_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team_name_norm",
            "team_name_clean",
            "torvik_team",
            "AdjOE_torvik",
            "AdjDE_torvik",
            "AdjEM_torvik",
        ]
    )


def fetch_torvik_ratings(
    year: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = _prediction_logger(logger)
    season = int(year or date.today().year)
    cached = _TORVIK_RATINGS_CACHE.get(season)
    if cached is not None:
        logger.info(
            f"PREDICTIONS torvik_loaded rows={len(cached)} season={season} source=cache"
        )
        return cached.copy()
    if requests is None:
        logger.warning("PREDICTIONS torvik skipped reason=requests_unavailable")
        return _empty_torvik_df()

    csv_url = f"https://barttorvik.com/getadvstats.php?year={season}&csv=1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(
            csv_url,
            headers=headers,
            timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
        )
        response.raise_for_status()
        payload = response.text.strip()
        if not payload:
            raise ValueError("empty response")
        if payload.lstrip().startswith("<"):
            raise ValueError("unexpected HTML response")
        raw = pd.read_csv(io.StringIO(payload))
    except Exception as ex:
        logger.info(
            "PREDICTIONS torvik fetch_failed "
            f"url={csv_url} error={type(ex).__name__}: {ex}"
        )
        raw = None

    if raw is not None:
        try:
            team_col = _find_column(raw, ["team", "team_name", "teamname", "school"])
            adjoe_col = _find_column(raw, ["AdjOE", "adj_oe", "oe", "offense"])
            adjde_col = _find_column(raw, ["AdjDE", "adj_de", "de", "defense"])
            adjem_col = _find_column(raw, ["AdjEM", "adj_em", "barthag"], required=False)

            torvik = pd.DataFrame(
                {
                    "torvik_team": raw[team_col].fillna("").astype(str).str.strip(),
                    "AdjOE_torvik": pd.to_numeric(raw[adjoe_col], errors="coerce"),
                    "AdjDE_torvik": pd.to_numeric(raw[adjde_col], errors="coerce"),
                }
            )
            if adjem_col is not None and _normalized_colname(adjem_col) != "barthag":
                torvik["AdjEM_torvik"] = pd.to_numeric(raw[adjem_col], errors="coerce")
            else:
                torvik["AdjEM_torvik"] = torvik["AdjOE_torvik"] - torvik["AdjDE_torvik"]

            torvik["team_name_clean"] = torvik["torvik_team"].apply(clean_team_name)
            torvik["team_name_norm"] = torvik["team_name_clean"].apply(_prediction_name_key)
            torvik = torvik[torvik["team_name_norm"] != ""].copy()
            torvik = torvik.dropna(subset=["AdjOE_torvik", "AdjDE_torvik"], how="any")
            torvik = (
                torvik.sort_values(["team_name_norm", "AdjEM_torvik"], ascending=[True, False])
                .drop_duplicates(subset=["team_name_norm"], keep="first")
                .reset_index(drop=True)
            )
            logger.info(
                f"PREDICTIONS torvik_loaded rows={len(torvik)} season={season} source=csv_endpoint"
            )
            _TORVIK_RATINGS_CACHE[season] = torvik.copy()
            return torvik
        except Exception as ex:
            logger.info(
                "PREDICTIONS torvik csv_schema_unrecognized "
                f"error={type(ex).__name__}: {ex}"
            )

    page_url = f"https://barttorvik.com/trank.php?year={season}&sort=&top=0&conlimit=All"
    try:
        session = requests.Session()
        page_response = session.post(
            page_url,
            headers=headers,
            data={"js_test_submitted": "1"},
            timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
        )
        page_response.raise_for_status()
        html = page_response.text
        if "Verifying Browser" in html:
            raise ValueError("js verification was not bypassed")
        if "<table" not in html.lower():
            raise ValueError("table markup missing from trank page")

        parser = _TorvikTableParser()
        parser.feed(html)
        header = None
        data_rows: list[list[str]] = []
        for idx, row in enumerate(parser.rows):
            if row[:7] == ["Rk", "Team", "Conf", "G", "Rec", "AdjOE", "AdjDE"]:
                header = row
                for candidate in parser.rows[idx + 1 :]:
                    if len(candidate) >= len(header) and str(candidate[0]).isdigit():
                        data_rows.append(candidate[: len(header)])
                break

        if header is None or not data_rows:
            raise ValueError("unable to locate Torvik ranking table rows")

        table = pd.DataFrame(data_rows, columns=header)
        torvik = pd.DataFrame(
            {
                "torvik_team": table["Team"].map(_clean_torvik_team_name),
                "AdjOE_torvik": table["AdjOE"].map(_parse_leading_float),
                "AdjDE_torvik": table["AdjDE"].map(_parse_leading_float),
            }
        )
        torvik["AdjEM_torvik"] = torvik["AdjOE_torvik"] - torvik["AdjDE_torvik"]
        torvik["team_name_clean"] = torvik["torvik_team"].apply(clean_team_name)
        torvik["team_name_norm"] = torvik["team_name_clean"].apply(_prediction_name_key)
        torvik = torvik[torvik["team_name_norm"] != ""].copy()
        torvik = torvik.dropna(subset=["AdjOE_torvik", "AdjDE_torvik"], how="any")
        torvik = (
            torvik.sort_values(["team_name_norm", "AdjEM_torvik"], ascending=[True, False])
            .drop_duplicates(subset=["team_name_norm"], keep="first")
            .reset_index(drop=True)
        )
        logger.info(
            f"PREDICTIONS torvik_loaded rows={len(torvik)} season={season} source=trank_page"
        )
        _TORVIK_RATINGS_CACHE[season] = torvik.copy()
        return torvik
    except Exception as ex:
        logger.warning(
            "PREDICTIONS torvik fallback_failed "
            f"url={page_url} error={type(ex).__name__}: {ex}"
        )
        empty = _empty_torvik_df()
        _TORVIK_RATINGS_CACHE[season] = empty.copy()
        return empty


def _extract_scoreboard_team_payload(game_obj: dict, is_home: bool) -> dict:
    side = game_obj.get("home" if is_home else "away")
    if isinstance(side, dict) and side:
        return side

    teams = game_obj.get("teams", [])
    if isinstance(teams, list):
        for item in teams:
            if not isinstance(item, dict):
                continue
            if bool(item.get("isHome")) == bool(is_home):
                return item
    return {}


def _extract_scoreboard_team_name(side: dict) -> str:
    if not isinstance(side, dict):
        return ""
    names = side.get("names", {}) if isinstance(side.get("names"), dict) else {}
    return str(
        side.get("nameShort")
        or names.get("short")
        or side.get("nameFull")
        or names.get("full")
        or side.get("name")
        or side.get("school")
        or ""
    ).strip()


def _extract_scoreboard_team_id(side: dict) -> str:
    if not isinstance(side, dict):
        return ""
    return (
        canonical_team_id(side.get("teamId") or side.get("teamID") or side.get("id"))
        or _safe_team_id(side.get("teamId") or side.get("teamID") or side.get("id"))
    )


def _scoreboard_game_date(game_obj: dict, fallback_day: date) -> str:
    raw_candidates = [
        game_obj.get("gameDate"),
        game_obj.get("startDate"),
        game_obj.get("date"),
        game_obj.get("startTime"),
    ]
    for raw in raw_candidates:
        if raw is None:
            continue
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
        if pd.notna(ts):
            try:
                return ts.tz_convert(NY).date().isoformat()
            except Exception:
                try:
                    return ts.date().isoformat()
                except Exception:
                    pass
    return fallback_day.isoformat()


def _scoreboard_is_neutral(game_obj: dict) -> bool:
    for key in ["neutralSite", "neutral", "isNeutralSite"]:
        if key in game_obj:
            return bool(game_obj.get(key))
    site = game_obj.get("site")
    if isinstance(site, dict):
        for key in ["neutralSite", "neutral", "isNeutralSite"]:
            if key in site:
                return bool(site.get(key))
    return False


def fetch_prediction_matchups(
    season: int | None = None,
    logger: logging.Logger | None = None,
    base_url: str | None = None,
    day: date | None = None,
    include_completed: bool = False,
) -> pd.DataFrame:
    logger = _prediction_logger(logger)
    target_day = day or datetime.now(tz=NY).date()
    src_base = str(base_url or os.environ.get("NCAA_API_BASE_URL", "")).strip()
    out_cols = [
        "date",
        "team_id",
        "opponent_team_id",
        "team",
        "opponent",
        "team_name_clean",
        "opponent_name_clean",
        "team_name_norm",
        "opponent_name_norm",
        "neutral_site",
    ]

    def _empty_schedule_df() -> pd.DataFrame:
        return pd.DataFrame(columns=out_cols)

    def _finalize_rows(rows: list[dict], source: str, detail: str = "") -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=out_cols) if rows else _empty_schedule_df()
        if out.empty:
            logger.info(
                f"PREDICTIONS schedule_loaded source={source} rows=0 "
                f"day={target_day}{detail}"
            )
            return out
        out = out.drop_duplicates(
            subset=["date", "team", "opponent"],
            keep="first",
        ).reset_index(drop=True)
        team_id_map = ti.load_team_id_map()
        before_aliases = dict(TEAM_NAME_ALIASES)
        before_team_id_map = dict(team_id_map)
        matched = 0
        total = 0
        if team_id_map:
            out["team_id"] = out["team_id"].fillna("").apply(_safe_team_id)
            out["opponent_team_id"] = out["opponent_team_id"].fillna("").apply(_safe_team_id)
            for col_name, id_col, scope_label in (
                ("team", "team_id", "schedule_team"),
                ("opponent", "opponent_team_id", "schedule_opponent"),
            ):
                missing_mask = out[id_col] == ""
                if not missing_mask.any():
                    continue
                total += int(missing_mask.sum())
                for idx in out.index[missing_mask]:
                    match = _resolve_team_match(
                        str(out.at[idx, col_name]),
                        team_id_map=team_id_map,
                        logger=logger,
                        source=scope_label,
                        remember=True,
                        allow_fuzzy=True,
                    )
                    if match.team_id:
                        out.at[idx, id_col] = match.team_id
                        matched += 1
        if total:
            ti.log_team_match_coverage(
                logger,
                scope="schedule_ingestion",
                matched=matched,
                total=total,
            )
        if team_id_map != before_team_id_map:
            ti.save_team_id_map(team_id_map)
            logger.info("TEAM_ID_MAP updated context=schedule_ingestion names=%s", len(team_id_map))
        _save_team_aliases_if_changed(
            before_aliases=before_aliases,
            after_aliases=TEAM_NAME_ALIASES,
            logger=logger,
            context="schedule_ingestion",
        )
        logger.info(
            f"PREDICTIONS schedule_loaded source={source} rows={len(out)} "
            f"day={target_day}{detail}"
        )
        return out

    def _parse_api_scoreboard(scoreboard: dict) -> list[dict]:
        rows: list[dict] = []
        for item in scoreboard.get("games", []) or []:
            game = unwrap_scoreboard_game(item)
            state = str(game.get("gameState", "")).strip().lower()
            if state in {"cancelled", "canceled", "postponed"}:
                continue
            if (not include_completed) and state.startswith("final"):
                continue

            home = _extract_scoreboard_team_payload(game, is_home=True)
            away = _extract_scoreboard_team_payload(game, is_home=False)
            home_name = _extract_scoreboard_team_name(home)
            away_name = _extract_scoreboard_team_name(away)
            if not home_name or not away_name:
                continue
            home_name_clean = clean_team_name(home_name)
            away_name_clean = clean_team_name(away_name)

            rows.append(
                {
                    "date": _scoreboard_game_date(game, target_day),
                    "team_id": _extract_scoreboard_team_id(home),
                    "opponent_team_id": _extract_scoreboard_team_id(away),
                    "team": home_name,
                    "opponent": away_name,
                    "team_name_clean": home_name_clean,
                    "opponent_name_clean": away_name_clean,
                    "team_name_norm": _prediction_name_key(home_name_clean),
                    "opponent_name_norm": _prediction_name_key(away_name_clean),
                    "neutral_site": _scoreboard_is_neutral(game),
                }
            )
        return rows

    def _parse_espn_scoreboard(payload: dict) -> list[dict]:
        rows: list[dict] = []
        for event in payload.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            competitions = event.get("competitions", []) or []
            competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
            status = competition.get("status") if isinstance(competition.get("status"), dict) else {}
            status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
            if (not include_completed) and bool(status_type.get("completed")):
                continue
            status_state = str(status_type.get("state", "")).strip().lower()
            status_desc = str(status_type.get("description", "")).strip().lower()
            if status_desc in {"postponed", "canceled", "cancelled"}:
                continue
            if (not include_completed) and status_state in {"post", "final"}:
                continue

            competitors = competition.get("competitors", []) or []
            if not isinstance(competitors, list) or len(competitors) < 2:
                continue

            home = None
            away = None
            for comp in competitors:
                if not isinstance(comp, dict):
                    continue
                side = str(comp.get("homeAway", "")).strip().lower()
                if side == "home" and home is None:
                    home = comp
                elif side == "away" and away is None:
                    away = comp
            if home is None or away is None:
                ordered = [comp for comp in competitors if isinstance(comp, dict)]
                if len(ordered) >= 2:
                    away = away or ordered[0]
                    home = home or ordered[1]

            if not isinstance(home, dict) or not isinstance(away, dict):
                continue

            home_team = home.get("team", {}) if isinstance(home.get("team"), dict) else {}
            away_team = away.get("team", {}) if isinstance(away.get("team"), dict) else {}

            home_name = str(
                home_team.get("displayName")
                or home_team.get("shortDisplayName")
                or home_team.get("location")
                or home_team.get("name")
                or ""
            ).strip()
            away_name = str(
                away_team.get("displayName")
                or away_team.get("shortDisplayName")
                or away_team.get("location")
                or away_team.get("name")
                or ""
            ).strip()
            if not home_name or not away_name:
                continue
            home_name_clean = clean_team_name(home_name)
            away_name_clean = clean_team_name(away_name)

            home_team_id = (
                canonical_team_id(home_team.get("id") or home.get("id"))
                or _safe_team_id(home_team.get("id") or home.get("id"))
            )
            away_team_id = (
                canonical_team_id(away_team.get("id") or away.get("id"))
                or _safe_team_id(away_team.get("id") or away.get("id"))
            )

            rows.append(
                {
                    "date": _scoreboard_game_date(competition or event, target_day),
                    "team_id": home_team_id,
                    "opponent_team_id": away_team_id,
                    "team": home_name,
                    "opponent": away_name,
                    "team_name_clean": home_name_clean,
                    "opponent_name_clean": away_name_clean,
                    "team_name_norm": _prediction_name_key(home_name_clean),
                    "opponent_name_norm": _prediction_name_key(away_name_clean),
                    "neutral_site": bool(competition.get("neutralSite")),
                }
            )
        return rows

    errors: list[str] = []

    if src_base:
        client = SourceClient(base_url=src_base)
        paths = [
            f"/scoreboard/basketball-men/d1/{target_day.year}/{target_day.month:02d}/{target_day.day:02d}"
        ]
        if target_day == datetime.now(tz=NY).date():
            paths.append("/scoreboard/basketball-men/d1")

        scoreboard = None
        for path in paths:
            scoreboard = get_json_with_retry(
                client=client,
                path=path,
                logger=logger,
                phase="PREDICTIONS_SCHEDULE",
                item_key=path,
                max_attempts=1,
            )
            if isinstance(scoreboard, dict):
                break

        if isinstance(scoreboard, dict):
            try:
                return _finalize_rows(
                    _parse_api_scoreboard(scoreboard),
                    source="api",
                    detail=f" base_url={src_base}",
                )
            except Exception as ex:
                errors.append(f"api_parse:{type(ex).__name__}: {ex}")
                logger.warning(
                    "PREDICTIONS schedule api_parse_failed "
                    f"base_url={src_base} day={target_day} "
                    f"error={type(ex).__name__}: {ex}"
                )
        else:
            errors.append(f"api_fetch:unavailable:{src_base}")
            logger.info(
                f"PREDICTIONS schedule api_unavailable base_url={src_base} day={target_day}"
            )

    espn_url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/mens-college-basketball/scoreboard"
    )
    if requests is None:
        errors.append("espn_fetch:requests_unavailable")
    else:
        try:
            response = requests.get(
                espn_url,
                params={
                    "dates": target_day.strftime("%Y%m%d"),
                    "groups": "50",
                    "limit": "400",
                },
                timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(
                    f"unexpected payload type={type(payload).__name__}"
                )
            return _finalize_rows(
                _parse_espn_scoreboard(payload),
                source="espn",
                detail=f" url={espn_url}",
            )
        except Exception as ex:
            errors.append(f"espn_fetch:{type(ex).__name__}: {ex}")
            logger.warning(
                "PREDICTIONS schedule espn_failed "
                f"day={target_day} error={type(ex).__name__}: {ex}"
            )

    logger.warning(
        "PREDICTIONS schedule unavailable "
        f"day={target_day} api_base_url={src_base or 'unset'} errors={errors}"
    )
    return _empty_schedule_df()


def _empty_vegas_lines_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "team_id",
            "opponent_team_id",
            "team",
            "opponent",
            "team_name_clean",
            "opponent_name_clean",
            "vegas_spread",
            "vegas_win_prob",
            "vegas_provider",
        ]
    )


def _american_odds_to_probability(raw_value) -> float | None:
    try:
        odds = float(raw_value)
    except Exception:
        return None
    if not np.isfinite(odds) or odds == 0:
        return None
    if odds > 0:
        return float(100.0 / (odds + 100.0))
    return float((-odds) / ((-odds) + 100.0))


def _safe_float(raw_value) -> float | None:
    try:
        value = float(raw_value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _select_pickcenter_entry(entries) -> dict:
    if not isinstance(entries, list):
        return {}
    ranked_entries: list[tuple[int, dict]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        home_odds = entry.get("homeTeamOdds") if isinstance(entry.get("homeTeamOdds"), dict) else {}
        away_odds = entry.get("awayTeamOdds") if isinstance(entry.get("awayTeamOdds"), dict) else {}
        has_spread = _safe_float(entry.get("spread")) is not None
        has_moneyline = (
            _american_odds_to_probability(home_odds.get("moneyLine")) is not None
            and _american_odds_to_probability(away_odds.get("moneyLine")) is not None
        )
        if not has_spread and not has_moneyline:
            continue
        provider = entry.get("provider") if isinstance(entry.get("provider"), dict) else {}
        try:
            priority = int(provider.get("priority", 9999))
        except Exception:
            priority = 9999
        ranked_entries.append((priority, entry))
    if not ranked_entries:
        return {}
    ranked_entries.sort(key=lambda item: item[0])
    return ranked_entries[0][1]


def fetch_vegas_lines(
    day: date,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = _prediction_logger(logger)
    target_day = pd.Timestamp(day).date()
    cache_key = target_day.isoformat()
    cached = _VEGAS_LINES_CACHE.get(cache_key)
    if cached is not None:
        logger.info(
            f"PREDICTIONS vegas_loaded rows={len(cached)} day={target_day} source=cache"
        )
        return cached.copy()
    if requests is None:
        logger.warning(
            f"PREDICTIONS vegas unavailable day={target_day} reason=requests_unavailable"
        )
        return _empty_vegas_lines_df()

    scoreboard_url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/mens-college-basketball/scoreboard"
    )
    summary_url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        "basketball/mens-college-basketball/summary"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
    }

    try:
        scoreboard_response = requests.get(
            scoreboard_url,
            params={
                "dates": target_day.strftime("%Y%m%d"),
                "groups": "50",
                "limit": "400",
            },
            headers=headers,
            timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
        )
        scoreboard_response.raise_for_status()
        scoreboard_payload = scoreboard_response.json()
        if not isinstance(scoreboard_payload, dict):
            raise ValueError(
                f"unexpected payload type={type(scoreboard_payload).__name__}"
            )
    except Exception as ex:
        logger.warning(
            "PREDICTIONS vegas unavailable "
            f"day={target_day} error={type(ex).__name__}: {ex}"
        )
        empty = _empty_vegas_lines_df()
        _VEGAS_LINES_CACHE[cache_key] = empty.copy()
        return empty

    event_items: list[dict] = []
    for event in scoreboard_payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions", []) or []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        competitors = competition.get("competitors", []) or []
        if not isinstance(competitors, list) or len(competitors) < 2:
            continue

        home = None
        away = None
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            side = str(comp.get("homeAway", "")).strip().lower()
            if side == "home" and home is None:
                home = comp
            elif side == "away" and away is None:
                away = comp
        if home is None or away is None:
            ordered = [comp for comp in competitors if isinstance(comp, dict)]
            if len(ordered) >= 2:
                away = away or ordered[0]
                home = home or ordered[1]
        if not isinstance(home, dict) or not isinstance(away, dict):
            continue

        home_team = home.get("team", {}) if isinstance(home.get("team"), dict) else {}
        away_team = away.get("team", {}) if isinstance(away.get("team"), dict) else {}
        home_name = str(
            home_team.get("displayName")
            or home_team.get("shortDisplayName")
            or home_team.get("location")
            or home_team.get("name")
            or ""
        ).strip()
        away_name = str(
            away_team.get("displayName")
            or away_team.get("shortDisplayName")
            or away_team.get("location")
            or away_team.get("name")
            or ""
        ).strip()
        if not home_name or not away_name:
            continue

        event_id = str(event.get("id") or competition.get("id") or "").strip()
        if not event_id:
            continue

        event_items.append(
            {
                "event": event,
                "competition": competition,
                "home": home,
                "away": away,
                "home_team": home_team,
                "away_team": away_team,
                "home_name": home_name,
                "away_name": away_name,
                "event_id": event_id,
            }
        )

    def _fetch_event_lines(event_item: dict) -> dict[str, object]:
        event = event_item["event"]
        competition = event_item["competition"]
        home = event_item["home"]
        away = event_item["away"]
        home_team = event_item["home_team"]
        away_team = event_item["away_team"]
        home_name = event_item["home_name"]
        away_name = event_item["away_name"]
        event_id = str(event_item["event_id"])

        try:
            summary_response = requests.get(
                summary_url,
                params={"event": event_id},
                headers=headers,
                timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
            )
            summary_response.raise_for_status()
            summary_payload = summary_response.json()
            if not isinstance(summary_payload, dict):
                raise ValueError(
                    f"unexpected summary payload type={type(summary_payload).__name__}"
                )
        except Exception:
            return {
                "rows": [],
                "summary_failures": 1,
                "missing_pickcenter": 0,
            }

        pickcenter = _select_pickcenter_entry(summary_payload.get("pickcenter"))
        if not pickcenter:
            return {
                "rows": [],
                "summary_failures": 0,
                "missing_pickcenter": 1,
            }

        provider = pickcenter.get("provider") if isinstance(pickcenter.get("provider"), dict) else {}
        provider_name = str(provider.get("name") or "").strip()
        home_team_odds = (
            pickcenter.get("homeTeamOdds")
            if isinstance(pickcenter.get("homeTeamOdds"), dict)
            else {}
        )
        away_team_odds = (
            pickcenter.get("awayTeamOdds")
            if isinstance(pickcenter.get("awayTeamOdds"), dict)
            else {}
        )

        spread_raw = _safe_float(pickcenter.get("spread"))
        spread_abs = abs(spread_raw) if spread_raw is not None else None
        home_prob_raw = _american_odds_to_probability(home_team_odds.get("moneyLine"))
        away_prob_raw = _american_odds_to_probability(away_team_odds.get("moneyLine"))
        home_prob = None
        away_prob = None
        if home_prob_raw is not None and away_prob_raw is not None:
            prob_total = home_prob_raw + away_prob_raw
            if prob_total > 0:
                home_prob = float(home_prob_raw / prob_total)
                away_prob = float(away_prob_raw / prob_total)

        home_favorite = bool(home_team_odds.get("favorite"))
        away_favorite = bool(away_team_odds.get("favorite"))
        if not home_favorite and not away_favorite and home_prob is not None and away_prob is not None:
            if home_prob > away_prob:
                home_favorite = True
            elif away_prob > home_prob:
                away_favorite = True

        home_spread = None
        if spread_abs is not None:
            if home_favorite and not away_favorite:
                home_spread = float(spread_abs)
            elif away_favorite and not home_favorite:
                home_spread = float(-spread_abs)
            elif spread_abs == 0:
                home_spread = 0.0

        if home_prob is None and home_spread is not None:
            home_prob = float(1.0 / (1.0 + np.exp(-home_spread / 6.0)))
            away_prob = float(1.0 - home_prob)

        if home_spread is None or home_prob is None or away_prob is None:
            return {
                "rows": [],
                "summary_failures": 0,
                "missing_pickcenter": 1,
            }

        home_team_id = (
            canonical_team_id(home_team.get("id") or home.get("id"))
            or _safe_team_id(home_team.get("id") or home.get("id"))
        )
        away_team_id = (
            canonical_team_id(away_team.get("id") or away.get("id"))
            or _safe_team_id(away_team.get("id") or away.get("id"))
        )
        game_date = _scoreboard_game_date(competition or event, target_day)
        home_name_clean = clean_team_name(home_name)
        away_name_clean = clean_team_name(away_name)

        return {
            "rows": [
                {
                    "date": game_date,
                    "team_id": home_team_id,
                    "opponent_team_id": away_team_id,
                    "team": home_name,
                    "opponent": away_name,
                    "team_name_clean": home_name_clean,
                    "opponent_name_clean": away_name_clean,
                    "vegas_spread": float(home_spread),
                    "vegas_win_prob": float(home_prob),
                    "vegas_provider": provider_name,
                },
                {
                    "date": game_date,
                    "team_id": away_team_id,
                    "opponent_team_id": home_team_id,
                    "team": away_name,
                    "opponent": home_name,
                    "team_name_clean": away_name_clean,
                    "opponent_name_clean": home_name_clean,
                    "vegas_spread": float(-home_spread),
                    "vegas_win_prob": float(away_prob),
                    "vegas_provider": provider_name,
                },
            ],
            "summary_failures": 0,
            "missing_pickcenter": 0,
        }

    rows: list[dict] = []
    summary_failures = 0
    missing_pickcenter = 0
    max_workers = min(8, max(len(event_items), 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_event_lines, event_item): event_item["event_id"]
            for event_item in event_items
        }
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception:
                summary_failures += 1
                continue
            summary_failures += int(result.get("summary_failures", 0))
            missing_pickcenter += int(result.get("missing_pickcenter", 0))
            rows.extend(result.get("rows", []))

    out = pd.DataFrame(rows, columns=_empty_vegas_lines_df().columns) if rows else _empty_vegas_lines_df()
    if not out.empty:
        out = out.drop_duplicates(
            subset=["date", "team_id", "opponent_team_id"],
            keep="first",
        ).reset_index(drop=True)
    logger.info(
        "PREDICTIONS vegas_loaded "
        f"rows={len(out)} day={target_day} "
        f"events={len(scoreboard_payload.get('events', []) or [])} "
        f"summary_failures={summary_failures} "
        f"missing_pickcenter={missing_pickcenter}"
    )
    _VEGAS_LINES_CACHE[cache_key] = out.copy()
    return out


def build_game_predictions(
    teams_df: pd.DataFrame,
    season: int | None = None,
    logger: logging.Logger | None = None,
    base_url: str | None = None,
    day: date | None = None,
    use_external_blend: bool = True,
    use_vegas_blend: bool = True,
    include_completed: bool = False,
    schedule_df: pd.DataFrame | None = None,
    hca_points_per_100: float = 3.0,
    apply_calibration: bool = True,
    probability_adjustment_model: dict[str, object] | None = None,
) -> pd.DataFrame:
    logger = _prediction_logger(logger)
    if teams_df is None or teams_df.empty:
        logger.warning("PREDICTIONS skipped reason=empty_teams_df")
        return _empty_predictions_df()

    try:
        team_col = _find_column(teams_df, ["Team", "team", "team_name"])
        adjem_col = _find_column(teams_df, ["AdjEM", "adj_em", "em"])
        adjem_adj_col = _find_column(
            teams_df,
            ["AdjEM_adj", "adj_em_adj", "adjusted_adjem"],
            required=False,
        )
        adjoe_col = _find_column(teams_df, ["AdjOE", "adj_oe", "oe"])
        adjde_col = _find_column(teams_df, ["AdjDE", "adj_de", "de"])
        pace_col = _find_column(teams_df, ["Tempo", "tempo", "Pace", "pace", "AdjT", "adj_t"])
        team_id_col = _find_column(teams_df, ["team_id", "teamid"], required=False)
    except Exception as ex:
        logger.warning(
            "PREDICTIONS skipped reason=missing_rating_columns "
            f"error={type(ex).__name__}: {ex}"
        )
        return _empty_predictions_df()

    internal = pd.DataFrame(
        {
            "team": teams_df[team_col].fillna("").astype(str).str.strip(),
            "AdjEM_internal": pd.to_numeric(teams_df[adjem_col], errors="coerce"),
            "AdjOE_internal": pd.to_numeric(teams_df[adjoe_col], errors="coerce"),
            "AdjDE_internal": pd.to_numeric(teams_df[adjde_col], errors="coerce"),
            "Tempo_internal": pd.to_numeric(teams_df[pace_col], errors="coerce"),
        }
    )
    if adjem_adj_col is not None:
        adjusted_adjem = pd.to_numeric(teams_df[adjem_adj_col], errors="coerce")
        internal["AdjEM_internal"] = adjusted_adjem.where(
            adjusted_adjem.notna(), internal["AdjEM_internal"]
        )
    if team_id_col is not None:
        internal["team_id"] = teams_df[team_id_col].apply(_safe_team_id)
    else:
        internal["team_id"] = ""

    internal["team_name_clean"] = internal["team"].apply(clean_team_name)
    internal["team_name_norm"] = internal["team_name_clean"].apply(_prediction_name_key)
    internal = internal[internal["team_name_clean"] != ""].copy()
    if internal.empty:
        logger.warning("PREDICTIONS skipped reason=no_valid_internal_teams")
        return _empty_predictions_df()
    logger.info(
        f"PREDICTIONS ratings_loaded rows={len(internal)} "
        f"using_adjusted_adjem={adjem_adj_col is not None}"
    )

    for col in ["AdjEM_internal", "AdjOE_internal", "AdjDE_internal", "Tempo_internal"]:
        internal[col] = internal[col].replace([np.inf, -np.inf], np.nan)

    league_tempo = internal["Tempo_internal"].dropna().median()
    if not np.isfinite(league_tempo):
        league_tempo = 70.0
    internal["Tempo_internal"] = internal["Tempo_internal"].fillna(float(league_tempo))

    if use_external_blend:
        torvik = fetch_torvik_ratings(year=season, logger=logger)
    else:
        logger.info("PREDICTIONS torvik skipped reason=external_blend_disabled")
        torvik = _empty_torvik_df()
    if torvik.empty:
        internal["AdjOE_torvik"] = np.nan
        internal["AdjDE_torvik"] = np.nan
        internal["AdjEM_torvik"] = np.nan
    else:
        internal = internal.merge(
            torvik[
                [
                    "team_name_norm",
                    "AdjOE_torvik",
                    "AdjDE_torvik",
                    "AdjEM_torvik",
                ]
            ],
            on="team_name_norm",
            how="left",
        )
    torvik_matches = int(internal["AdjEM_torvik"].notna().sum())
    logger.info(
        "PREDICTIONS torvik_merge "
        f"matched_rows={torvik_matches} total_rows={len(internal)}"
    )

    internal["AdjEM_blend"] = np.where(
        internal["AdjEM_torvik"].notna(),
        0.7 * internal["AdjEM_internal"] + 0.3 * internal["AdjEM_torvik"],
        internal["AdjEM_internal"],
    )

    if schedule_df is not None:
        schedule = schedule_df.copy()
        if "date" not in schedule.columns:
            schedule["date"] = (day or datetime.now(tz=NY).date()).isoformat()
        if "team_id" not in schedule.columns:
            schedule["team_id"] = ""
        if "opponent_team_id" not in schedule.columns:
            schedule["opponent_team_id"] = ""
        if "neutral_site" not in schedule.columns:
            schedule["neutral_site"] = False
        if "team_name_clean" not in schedule.columns:
            schedule["team_name_clean"] = schedule.get("team", pd.Series("", index=schedule.index)).apply(clean_team_name)
        if "opponent_name_clean" not in schedule.columns:
            schedule["opponent_name_clean"] = schedule.get("opponent", pd.Series("", index=schedule.index)).apply(clean_team_name)
        if "team_name_norm" not in schedule.columns:
            schedule["team_name_norm"] = schedule["team_name_clean"].apply(_prediction_name_key)
        if "opponent_name_norm" not in schedule.columns:
            schedule["opponent_name_norm"] = schedule["opponent_name_clean"].apply(_prediction_name_key)
    else:
        schedule = fetch_prediction_matchups(
            season=season,
            logger=logger,
            base_url=base_url,
            day=day,
            include_completed=include_completed,
        )
    if schedule.empty:
        return _empty_predictions_df()

    vegas_lines = _empty_vegas_lines_df()
    if use_vegas_blend:
        vegas_frames: list[pd.DataFrame] = []
        unique_schedule_days = sorted(
            {
                parsed_day.date()
                for parsed_day in pd.to_datetime(schedule["date"], errors="coerce")
                if pd.notna(parsed_day)
            }
        )
        for schedule_day in unique_schedule_days:
            try:
                day_lines = fetch_vegas_lines(schedule_day, logger=logger)
            except Exception as ex:
                logger.warning(
                    "PREDICTIONS vegas_failed "
                    f"day={schedule_day} error={type(ex).__name__}: {ex}"
                )
                continue
            if day_lines is not None and not day_lines.empty:
                vegas_frames.append(day_lines)
        if vegas_frames:
            vegas_lines = (
                pd.concat(vegas_frames, ignore_index=True)
                .drop_duplicates(
                    subset=["date", "team_id", "opponent_team_id"],
                    keep="first",
                )
                .reset_index(drop=True)
            )
    else:
        logger.info("PREDICTIONS vegas skipped reason=vegas_blend_disabled")

    internal_by_name = (
        internal.sort_values(["team_name_clean", "AdjEM_internal"], ascending=[True, False])
        .drop_duplicates(subset=["team_name_clean"], keep="first")
        .set_index("team_name_clean")
    )
    internal_by_id = (
        internal[internal["team_id"] != ""]
        .sort_values(["team_id", "AdjEM_internal"], ascending=[True, False])
        .drop_duplicates(subset=["team_id"], keep="first")
        .set_index("team_id")
    )
    rating_id_by_clean_name = (
        internal[internal["team_id"] != ""]
        .sort_values(["team_name_clean", "AdjEM_internal"], ascending=[True, False])
        .drop_duplicates(subset=["team_name_clean"], keep="first")
        .set_index("team_name_clean")["team_id"]
        .to_dict()
    )
    valid_rating_ids = {
        str(team_id).strip()
        for team_id in internal_by_id.index
        if str(team_id).strip()
    }
    prediction_out_dir = str(ti.TEAM_ID_MAP_PATH.parent)
    prediction_team_dir = load_team_directory_cache(prediction_out_dir)
    prediction_team_id_map = _build_team_identity_map(
        out_dir=prediction_out_dir,
        team_meta=prediction_team_dir,
        ratings_df=teams_df,
    )
    before_prediction_team_id_map = dict(prediction_team_id_map)
    before_prediction_aliases = dict(TEAM_NAME_ALIASES)
    before_id_gate = len(schedule)
    schedule["team_id"] = schedule["team_id"].fillna("").apply(_safe_team_id)
    schedule["opponent_team_id"] = schedule["opponent_team_id"].fillna("").apply(_safe_team_id)

    recovered_team_ids = 0
    recovered_opponent_ids = 0
    resolver_attempts = 0
    resolver_matches = 0
    invalid_team_mask = (schedule["team_id"] == "") | (~schedule["team_id"].isin(valid_rating_ids))
    if invalid_team_mask.any():
        resolver_attempts += int(invalid_team_mask.sum())
        for idx in schedule.index[invalid_team_mask]:
            match = _resolve_team_match(
                str(schedule.at[idx, "team"]),
                team_id_map=prediction_team_id_map,
                logger=logger,
                source="prediction_team",
                remember=True,
                allow_fuzzy=True,
            )
            if match.team_id and match.team_id in valid_rating_ids:
                schedule.at[idx, "team_id"] = match.team_id
                recovered_team_ids += 1
                resolver_matches += 1
        remaining_team_mask = (schedule["team_id"] == "") | (~schedule["team_id"].isin(valid_rating_ids))
        if remaining_team_mask.any():
            recovered_team_series = (
                schedule.loc[remaining_team_mask, "team_name_clean"]
                .map(rating_id_by_clean_name)
                .fillna("")
                .apply(_safe_team_id)
            )
            recovered_team_fill_mask = recovered_team_series != ""
            if recovered_team_fill_mask.any():
                recovered_idx = recovered_team_series.index[recovered_team_fill_mask]
                schedule.loc[recovered_idx, "team_id"] = recovered_team_series.loc[recovered_idx]
                recovered_team_ids += int(recovered_team_fill_mask.sum())

    invalid_opponent_mask = (
        (schedule["opponent_team_id"] == "")
        | (~schedule["opponent_team_id"].isin(valid_rating_ids))
    )
    if invalid_opponent_mask.any():
        resolver_attempts += int(invalid_opponent_mask.sum())
        for idx in schedule.index[invalid_opponent_mask]:
            match = _resolve_team_match(
                str(schedule.at[idx, "opponent"]),
                team_id_map=prediction_team_id_map,
                logger=logger,
                source="prediction_opponent",
                remember=True,
                allow_fuzzy=True,
            )
            if match.team_id and match.team_id in valid_rating_ids:
                schedule.at[idx, "opponent_team_id"] = match.team_id
                recovered_opponent_ids += 1
                resolver_matches += 1
        remaining_opponent_mask = (
            (schedule["opponent_team_id"] == "")
            | (~schedule["opponent_team_id"].isin(valid_rating_ids))
        )
        if remaining_opponent_mask.any():
            recovered_opp_series = (
                schedule.loc[remaining_opponent_mask, "opponent_name_clean"]
                .map(rating_id_by_clean_name)
                .fillna("")
                .apply(_safe_team_id)
            )
            recovered_opp_fill_mask = recovered_opp_series != ""
            if recovered_opp_fill_mask.any():
                recovered_idx = recovered_opp_series.index[recovered_opp_fill_mask]
                schedule.loc[recovered_idx, "opponent_team_id"] = recovered_opp_series.loc[recovered_idx]
                recovered_opponent_ids += int(recovered_opp_fill_mask.sum())

    if recovered_team_ids or recovered_opponent_ids:
        logger.info(
            "PREDICTIONS id_recovered "
            f"team_id={recovered_team_ids} opponent_team_id={recovered_opponent_ids}"
        )
    if resolver_attempts:
        ti.log_team_match_coverage(
            logger,
            scope="prediction_pipeline",
            matched=resolver_matches,
            total=resolver_attempts,
        )
    if prediction_team_id_map != before_prediction_team_id_map:
        ti.save_team_id_map(prediction_team_id_map, Path(prediction_out_dir) / "team_id_map.json")
        logger.info(
            "TEAM_ID_MAP updated context=prediction_pipeline names=%s",
            len(prediction_team_id_map),
        )
    _save_team_aliases_if_changed(
        before_aliases=before_prediction_aliases,
        after_aliases=TEAM_NAME_ALIASES,
        logger=logger,
        context="prediction_pipeline",
    )

    valid_id_mask = (
        (schedule["team_id"] != "")
        & (schedule["opponent_team_id"] != "")
        & schedule["team_id"].isin(valid_rating_ids)
        & schedule["opponent_team_id"].isin(valid_rating_ids)
    )
    invalid_id_rows = int((~valid_id_mask).sum())
    if invalid_id_rows:
        skipped = schedule.loc[
            ~valid_id_mask,
            ["date", "team", "opponent", "team_id", "opponent_team_id"],
        ].head(20).to_dict(orient="records")
        logger.warning(
            "PREDICTIONS skipped_matchups_missing_valid_ids "
            f"rows={invalid_id_rows} sample={skipped}"
        )
    schedule = schedule.loc[valid_id_mask].copy()
    logger.info(
        "PREDICTIONS id_gate "
        f"rows_before={before_id_gate} rows_after={len(schedule)} "
        f"valid_rating_ids={len(valid_rating_ids)}"
    )
    if schedule.empty:
        logger.warning("PREDICTIONS skipped reason=no_matchups_after_id_gate")
        return _empty_predictions_df()

    ratings_name_set = {
        str(name).strip()
        for name in internal_by_name.index
        if str(name).strip()
    }
    schedule_team_name_set = {
        str(name).strip()
        for name in schedule["team_name_clean"].fillna("").astype(str)
        if str(name).strip()
    }
    schedule_opponent_name_set = {
        str(name).strip()
        for name in schedule["opponent_name_clean"].fillna("").astype(str)
        if str(name).strip()
    }
    schedule_name_set = schedule_team_name_set | schedule_opponent_name_set
    missing_schedule_team_names = sorted(schedule_team_name_set - ratings_name_set)
    missing_schedule_opponent_names = sorted(schedule_opponent_name_set - ratings_name_set)
    ratings_not_in_schedule = sorted(ratings_name_set - schedule_name_set)

    logger.debug(
        "PREDICTIONS merge_keys counts "
        f"schedule_team_name_clean={len(schedule_team_name_set)} "
        f"schedule_opponent_name_clean={len(schedule_opponent_name_set)} "
        f"ratings_team_name_clean={len(ratings_name_set)}"
    )
    logger.debug(
        "PREDICTIONS merge_keys schedule_team_name_clean_sample="
        f"{sorted(schedule_team_name_set)[:25]}"
    )
    logger.debug(
        "PREDICTIONS merge_keys ratings_team_name_clean_sample="
        f"{sorted(ratings_name_set)[:25]}"
    )
    if missing_schedule_team_names:
        logger.warning(
            "PREDICTIONS merge_keys missing_in_ratings_team_name_clean_sample="
            f"{missing_schedule_team_names[:25]}"
        )
    if missing_schedule_opponent_names:
        logger.warning(
            "PREDICTIONS merge_keys missing_in_ratings_opponent_name_clean_sample="
            f"{missing_schedule_opponent_names[:25]}"
        )
    logger.debug(
        "PREDICTIONS merge_keys ratings_not_in_schedule_sample="
        f"{ratings_not_in_schedule[:25]}"
    )

    missing_team_pairs = (
        schedule.loc[
            ~schedule["team_name_clean"].isin(ratings_name_set),
            ["team", "team_name_clean"],
        ]
        .drop_duplicates()
        .sort_values(["team_name_clean", "team"])
    )
    if not missing_team_pairs.empty:
        logger.warning(
            "PREDICTIONS merge_keys missing_team_examples="
            f"{missing_team_pairs.head(25).to_dict(orient='records')}"
        )

    missing_opponent_pairs = (
        schedule.loc[
            ~schedule["opponent_name_clean"].isin(ratings_name_set),
            ["opponent", "opponent_name_clean"],
        ]
        .drop_duplicates()
        .sort_values(["opponent_name_clean", "opponent"])
    )
    if not missing_opponent_pairs.empty:
        logger.warning(
            "PREDICTIONS merge_keys missing_opponent_examples="
            f"{missing_opponent_pairs.head(25).to_dict(orient='records')}"
        )

    def fill_metric(
        frame: pd.DataFrame,
        output_col: str,
        id_col: str,
        name_col: str,
        source_col: str,
    ) -> None:
        frame[output_col] = np.nan
        if source_col in internal_by_id.columns:
            frame[output_col] = frame[id_col].map(internal_by_id[source_col])
        missing = frame[output_col].isna()
        if source_col in internal_by_name.columns:
            frame.loc[missing, output_col] = frame.loc[missing, name_col].map(
                internal_by_name[source_col]
            )

    def fill_metric_fuzzy(
        frame: pd.DataFrame,
        output_col: str,
        name_col: str,
        source_col: str,
        resolved_name_map: dict[str, str],
    ) -> int:
        if not resolved_name_map or source_col not in internal_by_name.columns:
            return 0
        missing = frame[output_col].isna()
        if not missing.any():
            return 0
        missing_idx = frame.index[missing]
        resolved_names = frame.loc[missing_idx, name_col].map(resolved_name_map)
        resolved_values = resolved_names.map(internal_by_name[source_col])
        fillable = resolved_values.notna()
        if not fillable.any():
            return 0
        fill_idx = resolved_values.index[fillable]
        frame.loc[fill_idx, output_col] = resolved_values.loc[fill_idx]
        return int(fillable.sum())

    preds = schedule.copy()
    fill_metric(preds, "team_AdjEM_blend", "team_id", "team_name_clean", "AdjEM_blend")
    fill_metric(
        preds,
        "opponent_AdjEM_blend",
        "opponent_team_id",
        "opponent_name_clean",
        "AdjEM_blend",
    )
    fill_metric(preds, "team_AdjOE", "team_id", "team_name_clean", "AdjOE_internal")
    fill_metric(
        preds,
        "opponent_AdjOE",
        "opponent_team_id",
        "opponent_name_clean",
        "AdjOE_internal",
    )
    fill_metric(preds, "team_Tempo", "team_id", "team_name_clean", "Tempo_internal")
    fill_metric(
        preds,
        "opponent_Tempo",
        "opponent_team_id",
        "opponent_name_clean",
        "Tempo_internal",
    )

    team_metric_cols = ["team_AdjEM_blend", "team_AdjOE", "team_Tempo"]
    opponent_metric_cols = ["opponent_AdjEM_blend", "opponent_AdjOE", "opponent_Tempo"]
    team_unmatched_names = sorted(
        {
            str(name).strip()
            for name in preds.loc[
                preds[team_metric_cols].isna().any(axis=1),
                "team_name_clean",
            ].fillna("").astype(str)
            if str(name).strip()
        }
    )
    opponent_unmatched_names = sorted(
        {
            str(name).strip()
            for name in preds.loc[
                preds[opponent_metric_cols].isna().any(axis=1),
                "opponent_name_clean",
            ].fillna("").astype(str)
            if str(name).strip()
        }
    )
    team_fuzzy_matches, team_fuzzy_ambiguous = _resolve_containment_name_matches(
        team_unmatched_names,
        internal_by_name.index.tolist(),
    )
    opponent_fuzzy_matches, opponent_fuzzy_ambiguous = _resolve_containment_name_matches(
        opponent_unmatched_names,
        internal_by_name.index.tolist(),
    )

    if team_fuzzy_matches:
        logger.info(
            "PREDICTIONS merge_fuzzy team_name_matches="
            f"{len(team_fuzzy_matches)} sample="
            f"{[{'schedule_name_clean': k, 'ratings_name_clean': v} for k, v in list(team_fuzzy_matches.items())[:10]]}"
        )
    if opponent_fuzzy_matches:
        logger.info(
            "PREDICTIONS merge_fuzzy opponent_name_matches="
            f"{len(opponent_fuzzy_matches)} sample="
            f"{[{'schedule_name_clean': k, 'ratings_name_clean': v} for k, v in list(opponent_fuzzy_matches.items())[:10]]}"
        )
    if team_fuzzy_ambiguous:
        logger.warning(
            "PREDICTIONS merge_fuzzy ambiguous_team_names="
            f"{dict(list(team_fuzzy_ambiguous.items())[:10])}"
        )
    if opponent_fuzzy_ambiguous:
        logger.warning(
            "PREDICTIONS merge_fuzzy ambiguous_opponent_names="
            f"{dict(list(opponent_fuzzy_ambiguous.items())[:10])}"
        )

    fill_metric_fuzzy(
        preds,
        "team_AdjEM_blend",
        "team_name_clean",
        "AdjEM_blend",
        team_fuzzy_matches,
    )
    fill_metric_fuzzy(
        preds,
        "opponent_AdjEM_blend",
        "opponent_name_clean",
        "AdjEM_blend",
        opponent_fuzzy_matches,
    )
    fill_metric_fuzzy(
        preds,
        "team_AdjOE",
        "team_name_clean",
        "AdjOE_internal",
        team_fuzzy_matches,
    )
    fill_metric_fuzzy(
        preds,
        "opponent_AdjOE",
        "opponent_name_clean",
        "AdjOE_internal",
        opponent_fuzzy_matches,
    )
    fill_metric_fuzzy(
        preds,
        "team_Tempo",
        "team_name_clean",
        "Tempo_internal",
        team_fuzzy_matches,
    )
    fill_metric_fuzzy(
        preds,
        "opponent_Tempo",
        "opponent_name_clean",
        "Tempo_internal",
        opponent_fuzzy_matches,
    )

    required_metrics = [
        "team_AdjEM_blend",
        "opponent_AdjEM_blend",
        "team_AdjOE",
        "opponent_AdjOE",
        "team_Tempo",
        "opponent_Tempo",
    ]
    matched_rows = int((~preds[required_metrics].isna().any(axis=1)).sum())
    logger.info(
        f"PREDICTIONS merge matched_rows={matched_rows} total_rows={len(preds)}"
    )
    missing_mask = preds[required_metrics].isna().any(axis=1)
    missing_rows = int(missing_mask.sum())
    if missing_rows:
        skipped = preds.loc[missing_mask, ["date", "team", "opponent"]].to_dict(orient="records")
        logger.warning(
            "PREDICTIONS skipped_matchups_missing_ratings "
            f"rows={missing_rows} sample={skipped[:10]}"
        )
        preds = preds.loc[~missing_mask].copy()

    if preds.empty:
        logger.warning("PREDICTIONS skipped reason=no_matchups_after_rating_merge")
        return _empty_predictions_df()

    hca = np.where(
        preds["neutral_site"].fillna(False),
        0.0,
        float(hca_points_per_100),
    )
    preds["projected_spread"] = (
        preds["team_AdjEM_blend"] - preds["opponent_AdjEM_blend"] + hca
    )
    preds["win_prob"] = 1.0 / (1.0 + np.exp(-preds["projected_spread"] / 6.0))
    preds["win_prob"] = _clip_probability_array(preds["win_prob"].to_numpy(dtype=float))
    if apply_calibration:
        calibration_asof = day or datetime.now(tz=NY).date()
        calibration_model = fit_prediction_calibration_model(
            logger=logger,
            asof_date=calibration_asof,
        )
        raw_probs = preds["win_prob"].to_numpy(dtype=float)
        calibrated_probs = apply_prediction_calibration(raw_probs, calibration_model)
        preds["win_prob"] = calibrated_probs
        logger.info(
            "PREDICTIONS calibration_applied "
            f"method={calibration_model.get('method', 'identity')} "
            f"rows={len(preds)} asof={calibration_asof} "
            f"mean_abs_delta={float(np.mean(np.abs(calibrated_probs - raw_probs))):.4f}"
        )
    if not vegas_lines.empty:
        preds = preds.merge(
            vegas_lines[
                [
                    "date",
                    "team_id",
                    "opponent_team_id",
                    "vegas_spread",
                    "vegas_win_prob",
                    "vegas_provider",
                ]
            ],
            on=["date", "team_id", "opponent_team_id"],
            how="left",
        )
        vegas_mask = (
            preds["vegas_spread"].notna()
            & preds["vegas_win_prob"].notna()
        )
        vegas_matches = int(vegas_mask.sum())
        if vegas_matches:
            model_prob_before = preds.loc[vegas_mask, "win_prob"].to_numpy(dtype=float)
            vegas_prob = preds.loc[vegas_mask, "vegas_win_prob"].to_numpy(dtype=float)
            selected_adjustment_model = probability_adjustment_model
            if selected_adjustment_model is None:
                selected_adjustment_model = fit_prediction_probability_adjustment_model(
                    logger=logger,
                    asof_date=day or datetime.now(tz=NY).date(),
                )
            adjusted_prob = apply_prediction_probability_adjustment(
                model_prob_before,
                vegas_prob,
                preds.loc[vegas_mask, "projected_spread"].to_numpy(dtype=float),
                selected_adjustment_model,
            )
            edge = model_prob_before - vegas_prob
            abs_edge = np.abs(edge)
            preds.loc[vegas_mask, "win_prob"] = adjusted_prob
            preds["win_prob"] = _clip_probability_array(preds["win_prob"].to_numpy(dtype=float))
            logger.info(
                "PREDICTIONS vegas_applied "
                f"matched_rows={vegas_matches} total_rows={len(preds)} "
                f"method={selected_adjustment_model.get('method', 'identity')} "
                f"provider_sample={sorted(set(preds.loc[vegas_mask, 'vegas_provider'].fillna('').astype(str)))[:5]} "
                f"mean_abs_prob_delta={float(np.mean(np.abs(preds.loc[vegas_mask, 'win_prob'].to_numpy(dtype=float) - model_prob_before))):.4f} "
                f"avg_abs_edge={float(np.mean(abs_edge)):.4f} "
                f"medium_edge_count={int(((abs_edge >= 0.05) & (abs_edge < 0.15)).sum())} "
                f"large_edge_count={int((abs_edge >= 0.15).sum())}"
            )
        else:
            logger.info(
                f"PREDICTIONS vegas_applied matched_rows=0 total_rows={len(preds)}"
            )
    poss = (preds["team_Tempo"] + preds["opponent_Tempo"]) / 2.0
    preds["projected_score_team"] = poss * (preds["team_AdjOE"] / 100.0)
    preds["projected_score_opp"] = poss * (preds["opponent_AdjOE"] / 100.0)
    if "vegas_win_prob" not in preds.columns:
        preds["vegas_win_prob"] = np.nan
    if "vegas_spread" not in preds.columns:
        preds["vegas_spread"] = np.nan
    if "vegas_provider" not in preds.columns:
        preds["vegas_provider"] = ""

    out = preds[PREDICTION_OUTPUT_COLUMNS].copy()
    out["win_prob"] = pd.to_numeric(out["win_prob"], errors="coerce").round(4)
    out["projected_spread"] = pd.to_numeric(out["projected_spread"], errors="coerce").round(2)
    out["projected_score_team"] = pd.to_numeric(out["projected_score_team"], errors="coerce").round(1)
    out["projected_score_opp"] = pd.to_numeric(out["projected_score_opp"], errors="coerce").round(1)
    out["vegas_win_prob"] = pd.to_numeric(out["vegas_win_prob"], errors="coerce").round(4)
    out["vegas_spread"] = pd.to_numeric(out["vegas_spread"], errors="coerce").round(2)
    out["vegas_provider"] = out["vegas_provider"].fillna("").astype(str)
    out = out.sort_values(["date", "team", "opponent"]).reset_index(drop=True)
    logger.info(f"PREDICTIONS built rows={len(out)}")
    return out


def refresh_live_predictions(
    cfg_path: str,
    base_url: str | None = None,
    day: date | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    logger = _prediction_logger(logger)
    cfg = load_config(cfg_path)
    project_root = _resolve_project_root_from_cfg(cfg_path)
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    target_day = day or datetime.now(tz=NY).date()
    src_base = str(base_url or os.environ.get("NCAA_API_BASE_URL", "")).strip()
    season = int(cfg.get("season", date.today().year))
    predictions_path = os.path.join(out_dir, "game_predictions.csv")
    predictions_log_path = os.path.join(out_dir, "predictions_log.csv")
    debug_path = os.path.join(out_dir, "debug_today_schedule.json")

    ratings_df = pd.DataFrame()
    ratings_xlsx_path = os.path.join(out_dir, "teams_power_full.xlsx")
    games_used_path = os.path.join(out_dir, "games_used_for_ratings.csv")

    if os.path.exists(ratings_xlsx_path):
        try:
            ratings_df = pd.read_excel(ratings_xlsx_path)
            if ratings_df is not None and not ratings_df.empty:
                logger.info(
                    "PREDICTIONS live_ratings_loaded "
                    f"source=teams_power_full rows={len(ratings_df)} "
                    f"path={os.path.abspath(ratings_xlsx_path)}"
                )
        except Exception as ex:
            logger.warning(
                "PREDICTIONS live_ratings_failed "
                f"source=teams_power_full path={os.path.abspath(ratings_xlsx_path)} "
                f"error={type(ex).__name__}: {ex}"
            )
            ratings_df = pd.DataFrame()

    if ratings_df.empty and os.path.exists(games_used_path):
        try:
            games_used_df = pd.read_csv(games_used_path)
            games_used_df = validate_games_df(games_used_df, logger)
            if games_used_df is not None and not games_used_df.empty:
                ratings_df = compute_ratings(
                    games_used_df,
                    cfg,
                    logger,
                    asof=pd.Timestamp(target_day, tz=NY),
                )
                logger.info(
                    "PREDICTIONS live_ratings_loaded "
                    f"source=games_used_for_ratings rows={len(ratings_df)} "
                    f"path={os.path.abspath(games_used_path)}"
                )
        except Exception as ex:
            logger.warning(
                "PREDICTIONS live_ratings_failed "
                f"source=games_used_for_ratings path={os.path.abspath(games_used_path)} "
                f"error={type(ex).__name__}: {ex}"
            )
            ratings_df = pd.DataFrame()

    if ratings_df is None or ratings_df.empty:
        logger.error(
            "PREDICTIONS ERROR no_ratings_available_for_live "
            f"day={target_day} cfg={os.path.abspath(cfg_path)}"
        )
        empty_df = _empty_predictions_df()
        empty_df.to_csv(predictions_path, index=False)
        debug_payload = {
            "date": target_day.isoformat(),
            "source": "live_refresh",
            "reason": "no_ratings_available",
            "base_url": src_base or "",
            "ratings_xlsx_path": os.path.abspath(ratings_xlsx_path),
            "games_used_path": os.path.abspath(games_used_path),
        }
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2)
        return {
            "rows": 0,
            "schedule_rows": 0,
            "path": predictions_path,
            "debug_path": debug_path,
        }

    schedule_df = fetch_prediction_matchups(
        season=season,
        logger=logger,
        base_url=src_base,
        day=target_day,
        include_completed=False,
    )
    if not src_base:
        logger.info(
            f"PREDICTIONS source=espn_schedule rows={len(schedule_df)} day={target_day}"
        )

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    try:
        hca_points_per_100 = float(model_cfg.get("hca_points_per_100", 3.0))
    except Exception:
        hca_points_per_100 = 3.0

    try:
        pred_df = build_game_predictions(
            ratings_df,
            season=season,
            logger=logger,
            base_url=src_base,
            day=target_day,
            schedule_df=schedule_df,
            hca_points_per_100=hca_points_per_100,
        )
    except Exception as ex:
        logger.error(
            "PREDICTIONS live_build_failed "
            f"day={target_day} error={type(ex).__name__}: {ex}"
        )
        pred_df = _empty_predictions_df()

    pred_df.to_csv(predictions_path, index=False)
    logger.info(
        f"PREDICTIONS live_rows={len(pred_df)} path={os.path.abspath(predictions_path)}"
    )

    if pred_df.empty:
        debug_payload = {
            "date": target_day.isoformat(),
            "base_url": src_base or "",
            "schedule_rows": int(len(schedule_df)),
            "schedule_columns": list(schedule_df.columns),
            "schedule_sample": schedule_df.head(50).to_dict(orient="records"),
            "ratings_rows": int(len(ratings_df)),
            "ratings_columns": list(ratings_df.columns),
            "ratings_sample": ratings_df.head(20).to_dict(orient="records"),
        }
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2, default=str)
        logger.error(
            "PREDICTIONS ERROR no_games_found_for_today "
            f"day={target_day} debug_path={os.path.abspath(debug_path)}"
        )
    else:
        pred_log_new = pred_df.copy()
        pred_log_new["actual_result"] = None
        pred_log_new["actual_margin"] = None
        pred_log_new = pred_log_new.reindex(columns=PREDICTION_LOG_COLUMNS)
        pred_log_existing = _ensure_predictions_log_schema(predictions_log_path, logger)
        if pred_log_existing is None:
            pred_log_existing = pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)
        for col in PREDICTION_LOG_COLUMNS:
            if col not in pred_log_existing.columns:
                pred_log_existing[col] = PREDICTION_LOG_DEFAULTS.get(col, np.nan)
        replace_keys = {
            (
                str(row["date"]).strip(),
                str(row["team"]).strip(),
                str(row["opponent"]).strip(),
            )
            for _, row in pred_log_new[["date", "team", "opponent"]].iterrows()
            if str(row["date"]).strip()
            and str(row["team"]).strip()
            and str(row["opponent"]).strip()
        }
        if replace_keys:
            keep_mask = ~pred_log_existing.apply(
                lambda row: (
                    str(row.get("date", "")).strip(),
                    str(row.get("team", "")).strip(),
                    str(row.get("opponent", "")).strip(),
                )
                in replace_keys,
                axis=1,
            )
            pred_log_existing = pred_log_existing.loc[keep_mask].copy()
        pred_log_combined = pd.concat(
            [pred_log_existing.reindex(columns=PREDICTION_LOG_COLUMNS), pred_log_new],
            ignore_index=True,
        )
        pred_log_combined.to_csv(predictions_log_path, index=False)
        logger.info(
            "PREDICTIONS live_log_upserted "
            f"rows={len(pred_log_new)} path={os.path.abspath(predictions_log_path)}"
        )

    return {
        "rows": int(len(pred_df)),
        "schedule_rows": int(len(schedule_df)),
        "path": predictions_path,
        "debug_path": debug_path if pred_df.empty else "",
    }


def _ensure_predictions_log_schema(
    predictions_log_path: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame | None:
    logger = _prediction_logger(logger)
    if not os.path.exists(predictions_log_path):
        return None

    try:
        pred_log_df = pd.read_csv(predictions_log_path)
    except Exception as ex:
        logger.warning(
            "PREDICTIONS log_schema_failed "
            f"path={os.path.abspath(predictions_log_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        return None

    changed = False
    for col in PREDICTION_LOG_COLUMNS:
        if col not in pred_log_df.columns:
            pred_log_df[col] = PREDICTION_LOG_DEFAULTS.get(col, np.nan)
            changed = True

    for col in ["vegas_provider", "closing_vegas_provider"]:
        if col in pred_log_df.columns:
            pred_log_df[col] = pred_log_df[col].fillna("").astype(str)
    for col in [
        "vegas_win_prob",
        "vegas_spread",
        "closing_vegas_win_prob",
        "closing_vegas_spread",
    ]:
        if col in pred_log_df.columns:
            pred_log_df[col] = pd.to_numeric(pred_log_df[col], errors="coerce")
    if "actual_result" in pred_log_df.columns:
        pred_log_df["actual_result"] = pd.to_numeric(
            pred_log_df["actual_result"], errors="coerce"
        ).astype("Int64")
    if "actual_margin" in pred_log_df.columns:
        pred_log_df["actual_margin"] = pd.to_numeric(
            pred_log_df["actual_margin"], errors="coerce"
        )

    ordered_cols = [col for col in PREDICTION_LOG_COLUMNS if col in pred_log_df.columns]
    extra_cols = [col for col in pred_log_df.columns if col not in ordered_cols]
    reordered = pred_log_df[ordered_cols + extra_cols].copy()
    if list(reordered.columns) != list(pred_log_df.columns):
        pred_log_df = reordered
        changed = True
    else:
        pred_log_df = reordered

    if changed:
        pred_log_df.to_csv(predictions_log_path, index=False)
        logger.info(
            f"PREDICTIONS log_schema_updated path={os.path.abspath(predictions_log_path)}"
        )
    return pred_log_df


def update_prediction_results(logger: logging.Logger | None = None) -> int:
    logger = _prediction_logger(logger)
    today = datetime.now(tz=NY).date()
    project_root = _default_project_root()
    predictions_log_path = os.path.join(project_root, "outputs", "predictions_log.csv")
    updated_rows = 0
    dates_processed = 0

    def _log_backfill_summary() -> None:
        logger.info(
            "PREDICTIONS backfill_updated "
            f"rows={updated_rows} dates_processed={dates_processed} "
            f"path={os.path.abspath(predictions_log_path)}"
        )

    if not os.path.exists(predictions_log_path):
        logger.info(
            f"PREDICTIONS results_updated rows=0 path={os.path.abspath(predictions_log_path)}"
        )
        _log_backfill_summary()
        return 0

    pred_log_df = _ensure_predictions_log_schema(predictions_log_path, logger)
    if pred_log_df is None:
        logger.info(
            f"PREDICTIONS results_updated rows=0 path={os.path.abspath(predictions_log_path)}"
        )
        _log_backfill_summary()
        return 0

    required_cols = {"date", "team", "opponent"}
    missing_cols = required_cols - set(pred_log_df.columns)
    if missing_cols:
        logger.warning(
            "PREDICTIONS results_update_failed "
            f"path={os.path.abspath(predictions_log_path)} missing_columns={sorted(missing_cols)}"
        )
        logger.info(
            f"PREDICTIONS results_updated rows=0 path={os.path.abspath(predictions_log_path)}"
        )
        _log_backfill_summary()
        return 0

    if "actual_result" not in pred_log_df.columns:
        pred_log_df["actual_result"] = pd.NA
    if "actual_margin" not in pred_log_df.columns:
        pred_log_df["actual_margin"] = np.nan
    for col in [
        "vegas_win_prob",
        "vegas_spread",
        "closing_vegas_win_prob",
        "closing_vegas_spread",
    ]:
        if col not in pred_log_df.columns:
            pred_log_df[col] = np.nan
    for col in ["vegas_provider", "closing_vegas_provider"]:
        if col not in pred_log_df.columns:
            pred_log_df[col] = ""

    pred_log_df["actual_result"] = pd.to_numeric(
        pred_log_df["actual_result"],
        errors="coerce",
    ).astype("Int64")
    pred_log_df["actual_margin"] = pd.to_numeric(
        pred_log_df["actual_margin"],
        errors="coerce",
    )
    pred_log_df["vegas_win_prob"] = pd.to_numeric(
        pred_log_df["vegas_win_prob"],
        errors="coerce",
    )
    pred_log_df["vegas_spread"] = pd.to_numeric(
        pred_log_df["vegas_spread"],
        errors="coerce",
    )
    pred_log_df["closing_vegas_win_prob"] = pd.to_numeric(
        pred_log_df["closing_vegas_win_prob"],
        errors="coerce",
    )
    pred_log_df["closing_vegas_spread"] = pd.to_numeric(
        pred_log_df["closing_vegas_spread"],
        errors="coerce",
    )
    pred_log_df["vegas_provider"] = pred_log_df["vegas_provider"].fillna("").astype(str)
    pred_log_df["closing_vegas_provider"] = (
        pred_log_df["closing_vegas_provider"].fillna("").astype(str)
    )
    pred_log_df["_prediction_date"] = pd.to_datetime(
        pred_log_df["date"],
        errors="coerce",
    ).dt.date
    pred_log_df["_team_name_clean"] = pred_log_df["team"].apply(clean_team_name)
    pred_log_df["_opponent_name_clean"] = pred_log_df["opponent"].apply(clean_team_name)

    needs_result_mask = (
        pred_log_df["_prediction_date"].notna()
        & (pred_log_df["_prediction_date"] < today)
        & pred_log_df["actual_result"].isna()
    )
    needs_line_mask = (
        pred_log_df["_prediction_date"].notna()
        & (pred_log_df["_prediction_date"] < today)
        & (
            pred_log_df["vegas_win_prob"].isna()
            | pred_log_df["closing_vegas_win_prob"].isna()
        )
    )
    needs_update_mask = needs_result_mask | needs_line_mask
    if not needs_update_mask.any():
        pred_log_df = pred_log_df.drop(
            columns=["_prediction_date", "_team_name_clean", "_opponent_name_clean"]
        )
        logger.info(
            f"PREDICTIONS results_updated rows=0 path={os.path.abspath(predictions_log_path)}"
        )
        _log_backfill_summary()
        return 0

    pending_dates = sorted({d for d in pred_log_df.loc[needs_update_mask, "_prediction_date"] if d})
    results_lookup: dict[tuple[str, str, str], tuple[int, float]] = {}
    line_lookup: dict[tuple[str, str, str], tuple[float, float, str]] = {}

    for game_day in pending_dates:
        dates_processed += 1
        try:
            games_df = fetch_espn_games_for_day(game_day, logger)
        except Exception as ex:
            logger.warning(
                "PREDICTIONS results_fetch_failed "
                f"day={game_day} error={type(ex).__name__}: {ex}"
            )
            continue
        if games_df is None or games_df.empty:
            continue

        games_df = games_df.copy()
        games_df["_team_name_clean"] = games_df["team"].apply(clean_team_name)
        games_df["_opponent_name_clean"] = games_df["opponent"].apply(clean_team_name)
        games_df["_actual_result"] = (
            pd.to_numeric(games_df["pts_for"], errors="coerce")
            > pd.to_numeric(games_df["pts_against"], errors="coerce")
        ).astype(int)
        games_df["_actual_margin"] = (
            pd.to_numeric(games_df["pts_for"], errors="coerce")
            - pd.to_numeric(games_df["pts_against"], errors="coerce")
        )

        for _, game_row in games_df.iterrows():
            date_key = str(game_row.get("game_date") or "").strip()
            team_key = str(game_row.get("_team_name_clean") or "").strip()
            opponent_key = str(game_row.get("_opponent_name_clean") or "").strip()
            if not date_key or not team_key or not opponent_key:
                continue
            key = (date_key, team_key, opponent_key)
            if key not in results_lookup:
                results_lookup[key] = (
                    int(game_row["_actual_result"]),
                    float(game_row["_actual_margin"]),
                )
        try:
            closing_lines_df = fetch_vegas_lines(game_day, logger)
        except Exception as ex:
            logger.warning(
                "PREDICTIONS closing_lines_fetch_failed "
                f"day={game_day} error={type(ex).__name__}: {ex}"
            )
            continue
        if closing_lines_df is None or closing_lines_df.empty:
            continue

        closing_lines_df = closing_lines_df.copy()
        closing_lines_df["_team_name_clean"] = closing_lines_df["team"].apply(clean_team_name)
        closing_lines_df["_opponent_name_clean"] = closing_lines_df["opponent"].apply(clean_team_name)
        closing_lines_df["vegas_win_prob"] = pd.to_numeric(
            closing_lines_df["vegas_win_prob"], errors="coerce"
        )
        closing_lines_df["vegas_spread"] = pd.to_numeric(
            closing_lines_df["vegas_spread"], errors="coerce"
        )
        for _, line_row in closing_lines_df.iterrows():
            date_key = str(line_row.get("date") or "").strip()
            team_key = str(line_row.get("_team_name_clean") or "").strip()
            opponent_key = str(line_row.get("_opponent_name_clean") or "").strip()
            if not date_key or not team_key or not opponent_key:
                continue
            key = (date_key, team_key, opponent_key)
            if key not in line_lookup:
                line_lookup[key] = (
                    float(line_row.get("vegas_win_prob"))
                    if pd.notna(line_row.get("vegas_win_prob"))
                    else np.nan,
                    float(line_row.get("vegas_spread"))
                    if pd.notna(line_row.get("vegas_spread"))
                    else np.nan,
                    str(line_row.get("vegas_provider") or "").strip(),
                )

    if results_lookup or line_lookup:
        for idx in pred_log_df.index[needs_update_mask]:
            date_key = pred_log_df.at[idx, "_prediction_date"]
            team_key = str(pred_log_df.at[idx, "_team_name_clean"] or "").strip()
            opponent_key = str(pred_log_df.at[idx, "_opponent_name_clean"] or "").strip()
            if not date_key or not team_key or not opponent_key:
                continue
            resolved = results_lookup.get((date_key.isoformat(), team_key, opponent_key))
            if resolved is None:
                pass
            else:
                actual_result, actual_margin = resolved
                if pd.isna(pred_log_df.at[idx, "actual_result"]):
                    pred_log_df.at[idx, "actual_result"] = int(actual_result)
                    updated_rows += 1
                pred_log_df.at[idx, "actual_margin"] = float(actual_margin)
            line_resolved = line_lookup.get((date_key.isoformat(), team_key, opponent_key))
            if line_resolved is not None:
                line_prob, line_spread, line_provider = line_resolved
                if pd.isna(pred_log_df.at[idx, "vegas_win_prob"]):
                    pred_log_df.at[idx, "vegas_win_prob"] = line_prob
                if pd.isna(pred_log_df.at[idx, "vegas_spread"]):
                    pred_log_df.at[idx, "vegas_spread"] = line_spread
                if not str(pred_log_df.at[idx, "vegas_provider"] or "").strip():
                    pred_log_df.at[idx, "vegas_provider"] = line_provider
                pred_log_df.at[idx, "closing_vegas_win_prob"] = line_prob
                pred_log_df.at[idx, "closing_vegas_spread"] = line_spread
                pred_log_df.at[idx, "closing_vegas_provider"] = line_provider

    pred_log_df = pred_log_df.drop(
        columns=["_prediction_date", "_team_name_clean", "_opponent_name_clean"]
    )
    pred_log_df.to_csv(predictions_log_path, index=False)
    logger.info(
        f"PREDICTIONS results_updated rows={updated_rows} path={os.path.abspath(predictions_log_path)}"
    )
    _log_backfill_summary()
    return updated_rows


def log_prediction_metrics(logger: logging.Logger | None = None) -> dict[str, float | int | None]:
    logger = _prediction_logger(logger)
    project_root = _default_project_root()
    predictions_log_path = os.path.join(project_root, "outputs", "predictions_log.csv")
    metrics = {
        "rows_used": 0,
        "brier_score": None,
        "log_loss": None,
        "spread_mae": None,
    }

    if not os.path.exists(predictions_log_path):
        logger.info(
            "MODEL_METRICS "
            "rows_used=0 brier_score=nan log_loss=nan spread_mae=nan "
            f"path={os.path.abspath(predictions_log_path)}"
        )
        return metrics

    pred_log_df = _ensure_predictions_log_schema(predictions_log_path, logger)
    if pred_log_df is None:
        logger.info(
            "MODEL_METRICS "
            "rows_used=0 brier_score=nan log_loss=nan spread_mae=nan "
            f"path={os.path.abspath(predictions_log_path)}"
        )
        return metrics

    required_cols = {"win_prob", "actual_result", "projected_spread"}
    missing_cols = required_cols - set(pred_log_df.columns)
    if missing_cols:
        logger.warning(
            "MODEL_METRICS failed "
            f"path={os.path.abspath(predictions_log_path)} missing_columns={sorted(missing_cols)}"
        )
        logger.info(
            "MODEL_METRICS "
            "rows_used=0 brier_score=nan log_loss=nan spread_mae=nan "
            f"path={os.path.abspath(predictions_log_path)}"
        )
        return metrics

    if "actual_margin" not in pred_log_df.columns:
        pred_log_df["actual_margin"] = np.nan

    eval_df = pred_log_df.copy()
    eval_df["win_prob"] = pd.to_numeric(eval_df["win_prob"], errors="coerce")
    eval_df["actual_result"] = pd.to_numeric(eval_df["actual_result"], errors="coerce")
    eval_df["projected_spread"] = pd.to_numeric(eval_df["projected_spread"], errors="coerce")
    eval_df["actual_margin"] = pd.to_numeric(eval_df["actual_margin"], errors="coerce")
    eval_df = eval_df.loc[
        eval_df["win_prob"].notna()
        & eval_df["actual_result"].notna()
        & eval_df["projected_spread"].notna()
    ].copy()

    rows_used = int(len(eval_df))
    metrics["rows_used"] = rows_used
    if rows_used == 0:
        logger.info(
            "MODEL_METRICS "
            "rows_used=0 brier_score=nan log_loss=nan spread_mae=nan "
            f"path={os.path.abspath(predictions_log_path)}"
        )
        return metrics

    probs = eval_df["win_prob"].clip(lower=1e-15, upper=1 - 1e-15)
    actuals = eval_df["actual_result"]
    brier_score = float(((probs - actuals) ** 2).mean())
    log_loss = float((-(actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs))).mean())
    spread_df = eval_df.loc[eval_df["actual_margin"].notna()].copy()
    spread_mae = (
        float((spread_df["projected_spread"] - spread_df["actual_margin"]).abs().mean())
        if not spread_df.empty
        else None
    )

    metrics["brier_score"] = brier_score
    metrics["log_loss"] = log_loss
    metrics["spread_mae"] = spread_mae
    spread_mae_str = "nan" if spread_mae is None else f"{spread_mae:.4f}"
    logger.info(
        "MODEL_METRICS "
        f"rows_used={rows_used} "
        f"brier_score={brier_score:.4f} "
        f"log_loss={log_loss:.4f} "
        f"spread_mae={spread_mae_str} "
        f"path={os.path.abspath(predictions_log_path)}"
    )
    return metrics


def backfill_historical_predictions(
    start_date: date,
    end_date: date,
    logger: logging.Logger | None = None,
) -> int:
    logger = _prediction_logger(logger)
    project_root = _default_project_root()
    out_dir = os.path.join(project_root, "outputs")
    cfg_path = _default_config_path()
    games_history_path = os.path.join(out_dir, "games_history.csv")
    predictions_log_path = os.path.join(out_dir, "predictions_log.csv")

    try:
        cfg = load_config(cfg_path)
    except Exception as ex:
        logger.warning(
            "PREDICTIONS backfill_failed "
            f"reason=config_load_failed path={os.path.abspath(cfg_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        logger.info("PREDICTIONS backfill_generated rows=0 days_processed=0")
        return 0

    if not os.path.exists(games_history_path):
        logger.warning(
            "PREDICTIONS backfill_failed "
            f"reason=missing_games_history path={os.path.abspath(games_history_path)}"
        )
        logger.info("PREDICTIONS backfill_generated rows=0 days_processed=0")
        return 0

    try:
        games_history_df = pd.read_csv(games_history_path)
    except Exception as ex:
        logger.warning(
            "PREDICTIONS backfill_failed "
            f"reason=games_history_read_failed path={os.path.abspath(games_history_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        logger.info("PREDICTIONS backfill_generated rows=0 days_processed=0")
        return 0

    games_history_df = validate_games_df(games_history_df, logger)
    if games_history_df.empty:
        logger.info("PREDICTIONS backfill_generated rows=0 days_processed=0")
        return 0

    if "is_d1_team" in games_history_df.columns and "is_d1_opponent" in games_history_df.columns:
        games_history_df = games_history_df[
            games_history_df["is_d1_team"].fillna(False)
            & games_history_df["is_d1_opponent"].fillna(False)
        ].copy()
    if games_history_df.empty:
        logger.info("PREDICTIONS backfill_generated rows=0 days_processed=0")
        return 0

    games_history_df["game_date"] = pd.to_datetime(games_history_df["game_date"], errors="coerce")
    games_history_df = games_history_df.dropna(subset=["game_date"]).copy()
    if games_history_df.empty:
        logger.info("PREDICTIONS backfill_generated rows=0 days_processed=0")
        return 0

    season = int(cfg.get("season", date.today().year))
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    try:
        hca_points_per_100 = float(model_cfg.get("hca_points_per_100", 3.0))
    except Exception:
        hca_points_per_100 = 3.0
    existing_keys: set[tuple[str, str, str]] = set()
    if os.path.exists(predictions_log_path):
        existing_df = _ensure_predictions_log_schema(predictions_log_path, logger)
        if existing_df is not None and {"date", "team", "opponent"}.issubset(existing_df.columns):
            existing_keys = {
                (
                    str(row["date"]).strip(),
                    str(row["team"]).strip(),
                    str(row["opponent"]).strip(),
                )
                for _, row in existing_df[["date", "team", "opponent"]].dropna().iterrows()
                if str(row["date"]).strip()
                and str(row["team"]).strip()
                and str(row["opponent"]).strip()
            }

    days_processed = 0
    total_rows = 0
    pending_frames: list[pd.DataFrame] = []

    for day_value in daterange(start_date, end_date):
        days_processed += 1
        logger.info(f"PREDICTIONS backfill_day start day={day_value}")

        games_subset = games_history_df.loc[
            games_history_df["game_date"] < pd.Timestamp(day_value)
        ].copy()
        if games_subset.empty:
            continue

        try:
            ratings_df = compute_ratings(
                games_subset,
                cfg,
                logger,
                asof=pd.Timestamp(day_value, tz=NY),
            )
        except Exception as ex:
            logger.warning(
                "PREDICTIONS backfill_day ratings_failed "
                f"day={day_value} error={type(ex).__name__}: {ex}"
            )
            continue
        if ratings_df is None or ratings_df.empty:
            continue

        try:
            pred_df = build_game_predictions(
                ratings_df,
                season=season,
                logger=logger,
                base_url=None,
                day=day_value,
                use_external_blend=False,
                include_completed=True,
                hca_points_per_100=hca_points_per_100,
            )
        except Exception as ex:
            logger.warning(
                "PREDICTIONS backfill_day predictions_failed "
                f"day={day_value} error={type(ex).__name__}: {ex}"
            )
            continue
        if pred_df is None or pred_df.empty:
            continue

        pred_df = pred_df.copy()
        pred_df["actual_result"] = None
        pred_df["actual_margin"] = None
        pred_df = pred_df.reindex(columns=PREDICTION_LOG_COLUMNS)
        dedupe_mask = pred_df.apply(
            lambda row: (
                str(row.get("date", "")).strip(),
                str(row.get("team", "")).strip(),
                str(row.get("opponent", "")).strip(),
            )
            not in existing_keys,
            axis=1,
        )
        pred_df = pred_df.loc[dedupe_mask].copy()
        if pred_df.empty:
            continue

        new_keys = {
            (
                str(row["date"]).strip(),
                str(row["team"]).strip(),
                str(row["opponent"]).strip(),
            )
            for _, row in pred_df[["date", "team", "opponent"]].iterrows()
            if str(row["date"]).strip()
            and str(row["team"]).strip()
            and str(row["opponent"]).strip()
        }
        existing_keys.update(new_keys)
        pending_frames.append(pred_df)
        total_rows += int(len(pred_df))

    if pending_frames:
        backfill_df = pd.concat(pending_frames, ignore_index=True)
        backfill_df.to_csv(
            predictions_log_path,
            mode="a",
            header=not os.path.exists(predictions_log_path),
            index=False,
        )

    logger.info(
        f"PREDICTIONS backfill_generated rows={total_rows} days_processed={days_processed}"
    )
    if total_rows > 0:
        update_prediction_results(logger)
        log_prediction_metrics(logger)
    return total_rows


def write_outputs(
    ratings: pd.DataFrame,
    sos: pd.DataFrame,
    out_dir: str,
    logger: logging.Logger,
    export_format: str = "xlsx",
    cleanup_csv_outputs: bool = False,
    conf_map: dict[str, str] | None = None,
):
    if ratings is None or ratings.empty:
        raise RuntimeError("ratings is empty at write_outputs()")

    os.makedirs(out_dir, exist_ok=True)
    ratings_out = ratings.copy()
    logger.info(f"EXPORT ratings rows={len(ratings_out)} cols={list(ratings_out.columns)}")
    if len(ratings_out) == 0:
        raise RuntimeError("EXPORT would write empty ratings; aborting")

    if "team_id" in ratings_out.columns and "team_id" in sos.columns:
        full = ratings_out.merge(sos[["team_id", "SOS_Power"]], on="team_id", how="left")
    else:
        full = ratings_out.merge(sos, on="team", how="left")
    if "SOS_Power" not in full.columns and "AvgOppEM" in full.columns:
        full = full.rename(columns={"AvgOppEM": "SOS_Power"})
    if full.empty:
        raise RuntimeError("EXPORT would write empty merged output; aborting")

    logger.info(
        f"EXPORT full rows={len(full)} cols={list(full.columns)} "
        f"null_SOS={full['SOS_Power'].isna().sum() if 'SOS_Power' in full else 'missing'}"
    )

    if "Conference" in full.columns:
        full = full.drop(columns=["Conference"])
    if conf_map and "team_id" in full.columns:
        full["Conference"] = full["team_id"].apply(_safe_team_id).map(conf_map).fillna("")
    else:
        full["Conference"] = ""

    # Conference-level summary (from full export frame).
    conf_df = full.copy()
    for col in ["AdjEM", "AdjOE", "AdjDE", "SOS_Power"]:
        if col in conf_df.columns:
            conf_df[col] = pd.to_numeric(conf_df[col], errors="coerce")
        else:
            conf_df[col] = np.nan
    conf_df["Conference"] = conf_df["Conference"].fillna("").astype(str)
    conf_df["team"] = conf_df["team"].fillna("").astype(str)
    conf_df = conf_df[conf_df["Conference"].astype(str).str.strip() != ""].copy()

    grouped = conf_df.groupby("Conference", dropna=False)
    conf_summary = grouped.agg(
        Teams=("team", "size"),
        Avg_AdjEM=("AdjEM", "mean"),
        Median_AdjEM=("AdjEM", "median"),
        Avg_AdjOE=("AdjOE", "mean"),
        Avg_AdjDE=("AdjDE", "mean"),
        Avg_SOS_Power=("SOS_Power", "mean"),
    ).reset_index()

    top_rows = (
        conf_df.sort_values(["Conference", "AdjEM", "team"], ascending=[True, False, True], na_position="last")
        .groupby("Conference", as_index=False)
        .first()[["Conference", "team", "AdjEM"]]
        .rename(columns={"team": "Top_Team", "AdjEM": "Top_Team_AdjEM"})
    )

    conf_summary = conf_summary.merge(top_rows, on="Conference", how="left")
    conf_summary = conf_summary[
        [
            "Conference",
            "Teams",
            "Avg_AdjEM",
            "Median_AdjEM",
            "Avg_AdjOE",
            "Avg_AdjDE",
            "Avg_SOS_Power",
            "Top_Team",
            "Top_Team_AdjEM",
        ]
    ]
    conf_summary = conf_summary.round(
        {
            "Avg_AdjEM": 3,
            "Median_AdjEM": 3,
            "Avg_AdjOE": 3,
            "Avg_AdjDE": 3,
            "Avg_SOS_Power": 3,
            "Top_Team_AdjEM": 3,
        }
    )
    conf_summary = conf_summary.sort_values(
        ["Avg_AdjEM", "Conference"], ascending=[False, True]
    ).reset_index(drop=True)
    conf_summary.insert(0, "Conf_Rank", np.arange(1, len(conf_summary) + 1))
    conf_summary.insert(
        2,
        "Conf_Tier",
        np.select(
            [
                conf_summary["Conf_Rank"] <= 5,
                conf_summary["Conf_Rank"] <= 16,
            ],
            ["Power", "Mid"],
            default="Low",
        ),
    )
    conf_summary = conf_summary[
        [
            "Conf_Rank",
            "Conference",
            "Conf_Tier",
            "Teams",
            "Avg_AdjEM",
            "Median_AdjEM",
            "Avg_AdjOE",
            "Avg_AdjDE",
            "Avg_SOS_Power",
            "Top_Team",
            "Top_Team_AdjEM",
        ]
    ]

    conf_rank_map = (
        conf_summary.set_index("Conference")["Conf_Rank"].to_dict()
        if not conf_summary.empty
        else {}
    )
    conf_avg_em_map = (
        conf_summary.set_index("Conference")["Avg_AdjEM"].to_dict()
        if not conf_summary.empty
        else {}
    )
    conf_tier_map = (
        conf_summary.set_index("Conference")["Conf_Tier"].to_dict()
        if not conf_summary.empty
        else {}
    )
    if "Conf_Rank" in full.columns:
        full = full.drop(columns=["Conf_Rank"])
    if "Conf_Avg_AdjEM" in full.columns:
        full = full.drop(columns=["Conf_Avg_AdjEM"])
    if "Conf_Tier" in full.columns:
        full = full.drop(columns=["Conf_Tier"])
    full["Conf_Rank"] = full["Conference"].map(conf_rank_map)
    full["Conf_Avg_AdjEM"] = full["Conference"].map(conf_avg_em_map)
    full["Conf_Tier"] = full["Conference"].map(conf_tier_map).fillna("")

    conf_csv_path = os.path.join(out_dir, "conference_power.csv")
    conf_summary.to_csv(conf_csv_path, index=False)
    conf_xlsx_path = os.path.join(out_dir, "conference_power.xlsx")
    try:
        with pd.ExcelWriter(conf_xlsx_path, engine="openpyxl") as w:
            conf_summary.to_excel(w, sheet_name="data", index=False)
    except Exception as ex:
        logger.warning(
            f"CONFERENCE_EXPORT xlsx_write_failed error={type(ex).__name__}: {ex}"
        )
    logger.info(
        f"CONFERENCE_EXPORT csv={os.path.abspath(conf_csv_path)} "
        f"xlsx={os.path.abspath(conf_xlsx_path)} rows={len(conf_summary)}"
    )

    fmt = str(export_format or "xlsx").strip().lower()
    if fmt not in {"xlsx", "csv"}:
        logger.warning(f"Unknown export_format={export_format!r}; defaulting to xlsx")
        fmt = "xlsx"

    if fmt == "csv":
        logger.info("EXPORT_FORMAT=csv csv_outputs_skipped=false")
        csv_paths = [
            (os.path.join(out_dir, "teams_power_full.csv"), full),
            (os.path.join(out_dir, "teams_power_top25.csv"), full.head(25)),
            (os.path.join(out_dir, "teams_power_top100.csv"), full.head(100)),
        ]
        for path, df_out in csv_paths:
            abs_path = os.path.abspath(path)
            logger.info(
                f"EXPORT_WRITE format=csv path={abs_path} rows={len(df_out)} cols={list(df_out.columns)}"
            )
            df_out.to_csv(path, index=False)
        return

    logger.info("EXPORT_FORMAT=xlsx csv_outputs_skipped=true")
    logger.info(f"EXPORT_XLSX rows={len(full)} cols={list(full.columns)}")
    xlsx_paths = [
        (os.path.join(out_dir, "teams_power_full.xlsx"), full),
        (os.path.join(out_dir, "teams_power_top25.xlsx"), full.head(25)),
        (os.path.join(out_dir, "teams_power_top100.xlsx"), full.head(100)),
    ]
    for path, df_out in xlsx_paths:
        abs_path = os.path.abspath(path)
        logger.info(
            f"EXPORT_WRITE format=xlsx path={abs_path} rows={len(df_out)} cols={list(df_out.columns)}"
        )
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_out.to_excel(w, sheet_name="data", index=False)
        logger.info(f"EXPORT_XLSX wrote {abs_path}")

    if cleanup_csv_outputs:
        legacy_csv = [
            os.path.join(out_dir, "teams_power_full.csv"),
            os.path.join(out_dir, "teams_power_top25.csv"),
            os.path.join(out_dir, "teams_power_top100.csv"),
        ]
        removed = []
        failed = []
        for csv_path in legacy_csv:
            if not os.path.exists(csv_path):
                continue
            try:
                os.remove(csv_path)
                removed.append(os.path.abspath(csv_path))
            except Exception as ex:
                failed.append(f"{os.path.abspath(csv_path)} ({type(ex).__name__}: {ex})")
        logger.info(
            f"EXPORT_CLEANUP cleanup_outputs=true removed_csv={len(removed)}"
        )
        for path in removed:
            logger.info(f"EXPORT_CLEANUP removed {path}")
        for msg in failed:
            logger.warning(f"EXPORT_CLEANUP failed {msg}")

def upsert_games_history(new_games: pd.DataFrame, out_dir: str, logger: logging.Logger) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "games_history.csv")

    # If no new rows, just return existing history (if any)
    if new_games is None or new_games.empty:
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            logger.info(f"Loaded existing games_history.csv rows={len(hist)}")
            return hist
        return pd.DataFrame()

    # Load existing history
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
    else:
        hist = pd.DataFrame(columns=new_games.columns)

    combined = pd.concat([hist, new_games], ignore_index=True)
    combined = validate_games_df(combined, logger)
    combined = filter_strict_d1_games(combined, logger, label="GAMES_HISTORY_UPSERT")

    # Dedup: unique by (game_id, team_id) when available, else fallback to team name.
    if "game_id" in combined.columns and "team_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["game_id", "team_id"], keep="last")
    elif "game_id" in combined.columns and "team" in combined.columns:
        combined = combined.drop_duplicates(subset=["game_id", "team"], keep="last")

    # Sort (optional but nice)
    if "game_date" in combined.columns:
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined = combined.sort_values(["game_date", "game_id", "team"], ascending=[False, True, True])

    combined.to_csv(hist_path, index=False)
    logger.info(f"Upserted games_history.csv rows={len(combined)} path={hist_path}")
    return combined


def upsert_players_history(
    players_new_df: pd.DataFrame,
    out_dir: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "players_history.csv")

    expected_cols = [
        "game_id",
        "game_date",
        "team_id",
        "opponent_team_id",
        "team",
        "opponent",
        "player_id",
        "player_key",
        "player_name",
        "minutes",
        "points",
        "fgm",
        "fga",
        "ftm",
        "fta",
        "tpm",
        "tpa",
        "orb",
        "drb",
        "trb",
        "ast",
        "tov",
        "stl",
        "blk",
        "pf",
        "starter",
    ]

    if players_new_df is None or players_new_df.empty:
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            logger.info(f"Loaded players_history.csv rows={len(hist)} path={hist_path}")
            return hist
        hist = pd.DataFrame(columns=expected_cols)
        hist.to_csv(hist_path, index=False)
        logger.info(f"Upserted players_history.csv rows=0 path={hist_path}")
        return hist

    new_df = players_new_df.copy()
    for c in expected_cols:
        if c not in new_df.columns:
            new_df[c] = ""
    for c in ["game_id", "game_date", "team_id", "opponent_team_id", "player_id", "player_key", "player_name"]:
        new_df[c] = new_df[c].fillna("").astype(str).str.strip()
    for c in ["team", "opponent"]:
        new_df[c] = new_df[c].fillna("").astype(str)
    new_df["team_id"] = new_df["team_id"].apply(_safe_team_id)
    new_df["opponent_team_id"] = new_df["opponent_team_id"].apply(_safe_team_id)
    new_df["player_id"] = new_df["player_id"].apply(_safe_player_id)
    fallback_key_mask = new_df["player_key"] == ""
    if fallback_key_mask.any():
        new_df.loc[fallback_key_mask, "player_key"] = (
            new_df.loc[fallback_key_mask, "player_name"].map(normalize_name)
            + "|"
            + new_df.loc[fallback_key_mask, "team_id"]
        )
    for c in [
        "minutes",
        "points",
        "fgm",
        "fga",
        "ftm",
        "fta",
        "tpm",
        "tpa",
        "orb",
        "drb",
        "trb",
        "ast",
        "tov",
        "stl",
        "blk",
        "pf",
    ]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0.0)
    if "starter" in new_df.columns:
        new_df["starter"] = new_df["starter"].apply(to_bool_flag)
    else:
        new_df["starter"] = False

    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
    else:
        hist = pd.DataFrame(columns=expected_cols)
    for c in expected_cols:
        if c not in hist.columns:
            hist[c] = ""

    combined = pd.concat([hist[expected_cols], new_df[expected_cols]], ignore_index=True)
    for c in ["game_id", "game_date", "team_id", "opponent_team_id", "player_key"]:
        combined[c] = combined[c].fillna("").astype(str).str.strip()
    combined["team_id"] = combined["team_id"].apply(_safe_team_id)
    combined["opponent_team_id"] = combined["opponent_team_id"].apply(_safe_team_id)
    combined["player_key"] = combined["player_key"].fillna("").astype(str).str.strip()

    game_id_key = combined["game_id"].fillna("").astype(str).str.strip()
    has_game_id = game_id_key != ""
    combined["__upsert_key"] = np.where(
        has_game_id,
        game_id_key + "|" + combined["team_id"] + "|" + combined["player_key"],
        combined["game_date"].fillna("").astype(str).str.strip()
        + "|"
        + combined["team_id"]
        + "|"
        + combined["player_key"]
        + "|"
        + combined["opponent_team_id"],
    )
    combined = combined.drop_duplicates(subset=["__upsert_key"], keep="last").drop(
        columns=["__upsert_key"]
    )

    if "game_date" in combined.columns:
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined = combined.sort_values(
            ["game_date", "game_id", "team_id", "player_key"],
            ascending=[False, True, True, True],
        )
        combined["game_date"] = combined["game_date"].dt.strftime("%Y-%m-%d")

    combined.to_csv(hist_path, index=False)
    logger.info(f"Upserted players_history.csv rows={len(combined)} path={hist_path}")
    return combined


def compute_player_rankings(
    players_history: pd.DataFrame,
    games_for_ratings: pd.DataFrame,
    ratings_df: pd.DataFrame,
    conf_map: dict[str, str],
    cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    out_cols = [
        "Player_Rank",
        "player_key",
        "player_name",
        "team_id",
        "team",
        "Conference",
        "Games",
        "Total_Minutes",
        "Min_per_game",
        "Min_Stability",
        "Adj_GS_per40",
        "Eff",
        "Impact_WAvg",
        "Impact_Shrunk",
        "Eff_WAvg",
        "OppAdj_Mean",
        "Player_Rating",
    ]
    if (
        players_history is None
        or players_history.empty
        or games_for_ratings is None
        or games_for_ratings.empty
        or ratings_df is None
        or ratings_df.empty
    ):
        return pd.DataFrame(columns=out_cols)

    ph = players_history.copy()
    gf = games_for_ratings.copy()
    for c in [
        "team_id",
        "opponent_team_id",
        "game_id",
        "game_date",
        "player_id",
        "player_key",
        "player_name",
    ]:
        if c not in ph.columns:
            ph[c] = ""
    if "team" not in ph.columns:
        ph["team"] = ""
    for c in ["team_id", "opponent_team_id", "game_id", "game_date"]:
        if c not in gf.columns:
            gf[c] = ""
    ph["team_id"] = ph["team_id"].apply(_safe_team_id)
    ph["opponent_team_id"] = ph["opponent_team_id"].apply(_safe_team_id)
    ph["game_id"] = ph["game_id"].fillna("").astype(str).str.strip()
    ph["game_date"] = pd.to_datetime(ph["game_date"], errors="coerce")
    gf["team_id"] = gf["team_id"].apply(_safe_team_id)
    gf["opponent_team_id"] = gf["opponent_team_id"].apply(_safe_team_id)
    gf["game_id"] = gf["game_id"].fillna("").astype(str).str.strip()
    gf["game_date"] = pd.to_datetime(gf["game_date"], errors="coerce")

    ph["__game_key"] = np.where(
        ph["game_id"] != "",
        ph["game_id"] + "|" + ph["team_id"] + "|" + ph["opponent_team_id"],
        ph["game_date"].dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + ph["team_id"]
        + "|"
        + ph["opponent_team_id"],
    )
    gf["__game_key"] = np.where(
        gf["game_id"] != "",
        gf["game_id"] + "|" + gf["team_id"] + "|" + gf["opponent_team_id"],
        gf["game_date"].dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + gf["team_id"]
        + "|"
        + gf["opponent_team_id"],
    )
    valid_keys = set(gf["__game_key"].dropna().astype(str))
    ph = ph[ph["__game_key"].astype(str).isin(valid_keys)].copy()
    if ph.empty:
        logger.info("PLAYER_RATINGS no matching player rows after D1 game-key filter")
        return pd.DataFrame(columns=out_cols)

    # Canonical/stable key: prefer player_id|team_id; fallback name|team_id.
    ph["player_id"] = ph.get("player_id", "").fillna("").apply(_safe_player_id)
    ph["player_key"] = ph.get("player_key", "").fillna("").astype(str).str.strip()
    ph["player_name"] = ph.get("player_name", "").fillna("").astype(str).str.strip()
    ph["__player_name_norm"] = ph["player_name"].map(normalize_name).fillna("")
    team_present = ph["team_id"] != ""
    has_player_id = (ph["player_id"] != "") & team_present
    ph.loc[has_player_id, "player_key"] = (
        ph.loc[has_player_id, "player_id"] + "|" + ph.loc[has_player_id, "team_id"]
    )

    no_id_mask = (~has_player_id) & team_present
    existing_key = ph["player_key"].astype(str)
    existing_has_pipe = existing_key.str.contains("\\|", regex=True)
    existing_left = existing_key.str.rsplit("|", n=1).str[0].fillna("").str.strip()
    existing_right = existing_key.str.rsplit("|", n=1).str[-1].fillna("").str.strip()
    existing_good_no_id = (
        no_id_mask
        & existing_has_pipe
        & (existing_right == ph["team_id"])
        & (existing_left != "")
        & (~existing_left.str.fullmatch(r"\d+", na=False))
    )
    fallback_key = (
        ph["__player_name_norm"] + "|" + ph["team_id"]
    ).where((ph["__player_name_norm"] != "") & team_present, "")
    rebuild_no_id = no_id_mask & (~existing_good_no_id)
    ph.loc[rebuild_no_id, "player_key"] = fallback_key[rebuild_no_id]
    ph = ph[(ph["player_key"] != "") & team_present].copy()
    ph = ph.drop(columns=["__player_name_norm"], errors="ignore")
    if ph.empty:
        return pd.DataFrame(columns=out_cols)

    for c in [
        "minutes",
        "points",
        "fgm",
        "fga",
        "ftm",
        "fta",
        "tpm",
        "tpa",
        "orb",
        "drb",
        "trb",
        "ast",
        "tov",
        "stl",
        "blk",
        "pf",
    ]:
        ph[c] = pd.to_numeric(ph.get(c, 0), errors="coerce").fillna(0.0)
    ph["minutes_float"] = ph["minutes"].apply(_parse_minutes_value)
    ph["team"] = ph.get("team", "").fillna("").astype(str).str.strip()

    defaults = {
        "enabled": True,
        "min_games": 8,
        "min_total_minutes": 160,
        "min_minutes_per_game": 12,
        "max_single_game_weight_minutes": 36,
        "recency_half_life_days": 21,
        "opponent_adjust_strength": 0.5,
        "prior_minutes": 240,
        "prior_rating": 0.0,
        "pace_adjust": True,
        "min_minutes": 5.0,
        "minutes_stability_games": 10,
    }
    raw_players_cfg = cfg.get("players", {}) if isinstance(cfg.get("players", {}), dict) else {}
    players_cfg = dict(defaults)
    players_cfg.update(raw_players_cfg)
    min_minutes = float(players_cfg.get("min_minutes", 5.0))
    min_games_default = 8
    min_total_minutes_default = 160.0
    try:
        min_games = int(players_cfg.get("min_games", min_games_default))
    except Exception:
        min_games = min_games_default
    if min_games < 1:
        min_games = 1
    try:
        min_total_minutes = float(players_cfg.get("min_total_minutes", min_total_minutes_default))
    except Exception:
        min_total_minutes = min_total_minutes_default
    if min_total_minutes < 0:
        min_total_minutes = 0.0
    try:
        min_minutes_per_game = float(players_cfg.get("min_minutes_per_game", 12.0))
    except Exception:
        min_minutes_per_game = 12.0
    if min_minutes_per_game < 0:
        min_minutes_per_game = 0.0
    try:
        max_single_game_weight_minutes = float(
            players_cfg.get("max_single_game_weight_minutes", 36.0)
        )
    except Exception:
        max_single_game_weight_minutes = 36.0
    max_single_game_weight_minutes = max(1.0, max_single_game_weight_minutes)
    try:
        opponent_adjust_strength = float(players_cfg.get("opponent_adjust_strength", 0.5))
    except Exception:
        opponent_adjust_strength = 0.5
    try:
        prior_minutes = float(players_cfg.get("prior_minutes", 240.0))
    except Exception:
        prior_minutes = 240.0
    if prior_minutes < 0:
        prior_minutes = 0.0
    try:
        prior_rating = float(players_cfg.get("prior_rating", 0.0))
    except Exception:
        prior_rating = 0.0
    pace_adjust = to_bool_flag(players_cfg.get("pace_adjust", True))
    stability_window = int(players_cfg.get("minutes_stability_games", 10))
    if stability_window <= 0:
        stability_window = 10
    ph = ph[ph["minutes_float"] >= min_minutes].copy()
    if ph.empty:
        logger.info("PLAYER_RATINGS no rows after min-minutes filter")
        return pd.DataFrame(columns=out_cols)

    ph["usage"] = ph["fga"] + 0.475 * ph["fta"] + ph["tov"]
    ph["pts_per_usage"] = ph["points"] / np.maximum(1.0, ph["usage"])
    ph["gs"] = (
        ph["points"]
        + 0.4 * ph["fgm"]
        - 0.7 * ph["fga"]
        - 0.4 * (ph["fta"] - ph["ftm"])
        + 0.7 * ph["orb"]
        + 0.3 * ph["drb"]
        + ph["stl"]
        + 0.7 * ph["ast"]
        + 0.7 * ph["blk"]
        - 1.0 * ph["tov"]
        - 0.4 * ph["pf"]
    )
    ph["gs_per40"] = ph["gs"] * 40.0 / np.maximum(12.0, ph["minutes_float"])

    ratings_work = ratings_df.copy()
    for c in ["AdjDE", "Pace"]:
        ratings_work[c] = pd.to_numeric(ratings_work.get(c), errors="coerce")
    if "team_id" in ratings_work.columns:
        ratings_work["team_id"] = ratings_work["team_id"].apply(_safe_team_id)
        opp_adjde_map = ratings_work.set_index("team_id")["AdjDE"].to_dict()
        team_pace_map = ratings_work.set_index("team_id")["Pace"].to_dict()
    else:
        opp_adjde_map = ratings_work.set_index("team")["AdjDE"].to_dict()
        team_pace_map = ratings_work.set_index("team")["Pace"].to_dict()

    league_avg_adjde = pd.to_numeric(ratings_work.get("AdjDE"), errors="coerce").mean()
    if not np.isfinite(league_avg_adjde) or league_avg_adjde <= 0:
        league_avg_adjde = pd.to_numeric(ratings_work.get("AdjDE"), errors="coerce").median()
    if not np.isfinite(league_avg_adjde) or league_avg_adjde <= 0:
        league_avg_adjde = 100.0
    league_median_pace = pd.to_numeric(ratings_work.get("Pace"), errors="coerce").median()
    if not np.isfinite(league_median_pace) or league_median_pace <= 0:
        league_median_pace = 70.0

    ph["team_Pace"] = pd.to_numeric(ph["team_id"].map(team_pace_map), errors="coerce")
    ph["team_Pace"] = ph["team_Pace"].replace([np.inf, -np.inf], np.nan).fillna(
        league_median_pace
    )
    if pace_adjust:
        ph["impact"] = ph["gs_per40"] * (ph["team_Pace"] / float(league_median_pace))
    else:
        ph["impact"] = ph["gs_per40"]

    ph["__player_game_key"] = np.where(
        ph["game_id"] != "",
        ph["game_id"],
        ph["game_date"].dt.strftime("%Y-%m-%d").fillna("") + "|" + ph["opponent_team_id"],
    )
    ph["opp_AdjDE"] = ph["opponent_team_id"].map(opp_adjde_map)
    ph["adj_factor"] = pd.to_numeric(ph["opp_AdjDE"], errors="coerce") / float(league_avg_adjde)
    ph["adj_factor"] = 1.0 + opponent_adjust_strength * (ph["adj_factor"] - 1.0)
    ph["adj_factor"] = (
        ph["adj_factor"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.85, upper=1.15)
    )
    ph["impact_adj"] = ph["impact"] * ph["adj_factor"]

    half_life = (
        players_cfg.get("recency_half_life_days")
        or cfg.get("model", {}).get("recency_half_life_days")
        or cfg.get("ratings", {}).get("half_life_days")
        or 21
    )
    try:
        half_life = float(half_life)
    except Exception:
        half_life = 21.0
    if half_life <= 0:
        half_life = 21.0
    now_ny = pd.Timestamp.now(tz=NY).normalize()
    if ph["game_date"].dt.tz is None:
        ph["game_date"] = ph["game_date"].dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    else:
        ph["game_date"] = ph["game_date"].dt.tz_convert(NY)
    ph = ph.dropna(subset=["game_date"])
    if ph.empty:
        return pd.DataFrame(columns=out_cols)
    age_days = (now_ny - ph["game_date"]).dt.total_seconds() / 86400.0
    ph["w_recency"] = np.exp(-np.log(2.0) * (age_days / half_life))
    ph["minutes_cap"] = np.minimum(
        np.maximum(ph["minutes_float"], 0.0), max_single_game_weight_minutes
    )
    ph["w_minutes"] = np.sqrt(ph["minutes_cap"] / 36.0)
    ph["w"] = ph["w_recency"] * ph["w_minutes"]
    ph = ph[ph["w"] > 0].copy()
    if ph.empty:
        return pd.DataFrame(columns=out_cols)

    mode_names = (
        ph.groupby("player_key")["player_name"]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else "")
        .to_dict()
    )
    mode_team_id = (
        ph.groupby("player_key")["team_id"]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else "")
        .to_dict()
    )
    mode_team_name = (
        ph.groupby("player_key")["team"]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else "")
        .to_dict()
    )

    tail = ph.sort_values(["player_key", "game_date"], ascending=[True, False]).groupby("player_key").head(stability_window)
    stability_df = tail.groupby("player_key")["minutes_float"].agg(["mean", "std"]).rename(
        columns={"mean": "minutes_mean", "std": "minutes_std"}
    )
    stability_df["minutes_std"] = stability_df["minutes_std"].fillna(0.0)
    stability_df["Min_Stability"] = 1.0 / (
        1.0 + (stability_df["minutes_std"] / np.maximum(1.0, stability_df["minutes_mean"]))
    )

    def _wavg(group: pd.DataFrame, val_col: str) -> float:
        v = pd.to_numeric(group[val_col], errors="coerce")
        w = pd.to_numeric(group["w"], errors="coerce")
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return 0.0
        vv = v[mask].to_numpy(dtype=float)
        ww = w[mask].to_numpy(dtype=float)
        sw = float(ww.sum())
        return float((vv * ww).sum() / sw) if sw > 0 else 0.0

    agg_rows = []
    for pkey, grp in ph.groupby("player_key", sort=False):
        game_keys = grp["__player_game_key"].replace("", np.nan).dropna()
        games = int(game_keys.nunique())
        total_minutes = float(pd.to_numeric(grp["minutes_float"], errors="coerce").sum())
        min_pg = float(total_minutes / games) if games > 0 else 0.0
        impact_wavg = _wavg(grp, "impact_adj")
        eff_wavg = _wavg(grp, "pts_per_usage")
        opp_adj_mean = float(pd.to_numeric(grp["adj_factor"], errors="coerce").mean())
        agg_rows.append(
            {
                "player_key": pkey,
                "player_name": mode_names.get(pkey, ""),
                "team_id": mode_team_id.get(pkey, ""),
                "team": mode_team_name.get(pkey, ""),
                "Games": games,
                "Total_Minutes": total_minutes,
                "Min_per_game": min_pg,
                "Adj_GS_per40": impact_wavg,
                "Eff": eff_wavg,
                "Impact_WAvg": impact_wavg,
                "Eff_WAvg": eff_wavg,
                "OppAdj_Mean": opp_adj_mean,
            }
        )
    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        return pd.DataFrame(columns=out_cols)
    agg = agg[
        (pd.to_numeric(agg["Games"], errors="coerce").fillna(0) >= min_games)
        & (
            pd.to_numeric(agg["Total_Minutes"], errors="coerce").fillna(0.0)
            >= min_total_minutes
        )
        & (
            pd.to_numeric(agg["Min_per_game"], errors="coerce").fillna(0.0)
            >= min_minutes_per_game
        )
    ].copy()
    if agg.empty:
        logger.info(
            f"PLAYER_RATINGS no rows after player sample guardrails min_games={min_games} "
            f"min_total_minutes={min_total_minutes:.1f} min_minutes_per_game={min_minutes_per_game:.1f}"
        )
        return pd.DataFrame(columns=out_cols)

    agg = agg.merge(stability_df[["Min_Stability"]], how="left", left_on="player_key", right_index=True)
    agg["Min_Stability"] = agg["Min_Stability"].fillna(1.0).clip(lower=0.0, upper=1.0)
    eff_median = float(pd.to_numeric(agg["Eff_WAvg"], errors="coerce").median())
    if not np.isfinite(eff_median):
        eff_median = 0.0
    shrink = pd.to_numeric(agg["Total_Minutes"], errors="coerce").fillna(0.0)
    shrink = shrink / np.maximum(shrink + prior_minutes, 1e-9)
    agg["Impact_Shrunk"] = shrink * agg["Impact_WAvg"] + (1.0 - shrink) * prior_rating
    agg["Player_Rating"] = (
        agg["Impact_Shrunk"]
        + 2.0 * (agg["Eff_WAvg"] - eff_median)
        + 0.5 * agg["Min_Stability"]
    )
    agg["Conference"] = agg["team_id"].apply(_safe_team_id).map(conf_map).fillna("")
    agg = agg.sort_values(
        ["Player_Rating", "Min_per_game", "player_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    agg.insert(0, "Player_Rank", np.arange(1, len(agg) + 1))

    for c in [
        "Total_Minutes",
        "Min_per_game",
        "Min_Stability",
        "Adj_GS_per40",
        "Eff",
        "Impact_WAvg",
        "Impact_Shrunk",
        "Eff_WAvg",
        "OppAdj_Mean",
        "Player_Rating",
    ]:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").round(3)
    agg["Games"] = pd.to_numeric(agg["Games"], errors="coerce").fillna(0).astype(int)
    agg["team_id"] = agg["team_id"].apply(_safe_team_id)
    return agg[out_cols]


def write_player_rankings_outputs(
    player_rankings: pd.DataFrame,
    out_dir: str,
    logger: logging.Logger,
    export_format: str = "xlsx",
):
    if player_rankings is None or player_rankings.empty:
        logger.info("PLAYER_EXPORT skipped empty player_rankings")
        return
    os.makedirs(out_dir, exist_ok=True)
    full_xlsx_path = os.path.join(out_dir, "player_rankings_full.xlsx")
    top100_xlsx_path = os.path.join(out_dir, "player_rankings_top100.xlsx")
    top100 = player_rankings.head(100).copy()

    try:
        with pd.ExcelWriter(full_xlsx_path, engine="openpyxl") as w:
            player_rankings.to_excel(w, sheet_name="data", index=False)
    except Exception as ex:
        logger.warning(
            f"PLAYER_EXPORT xlsx_write_failed file={os.path.abspath(full_xlsx_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
    try:
        with pd.ExcelWriter(top100_xlsx_path, engine="openpyxl") as w:
            top100.to_excel(w, sheet_name="data", index=False)
    except Exception as ex:
        logger.warning(
            f"PLAYER_EXPORT xlsx_write_failed file={os.path.abspath(top100_xlsx_path)} "
            f"error={type(ex).__name__}: {ex}"
        )

    logger.info(
        f"PLAYER_EXPORT xlsx_full={os.path.abspath(full_xlsx_path)} rows={len(player_rankings)} "
        f"xlsx_top100={os.path.abspath(top100_xlsx_path)} rows_top100={len(top100)}"
    )

    if str(export_format or "").strip().lower() == "csv":
        full_csv_path = os.path.join(out_dir, "player_rankings_full.csv")
        top100_csv_path = os.path.join(out_dir, "player_rankings_top100.csv")
        player_rankings.to_csv(full_csv_path, index=False)
        top100.to_csv(top100_csv_path, index=False)
        logger.info(
            f"PLAYER_EXPORT csv_full={os.path.abspath(full_csv_path)} "
            f"csv_top100={os.path.abspath(top100_csv_path)}"
        )


def _clean_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _clean_html_text(v) -> str:
    s = _clean_text(v)
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _standardize_player_status(raw_status, injury_text="", notes_text="") -> str:
    parts = [_clean_text(raw_status), _clean_text(injury_text), _clean_text(notes_text)]
    text = " ".join(x for x in parts if x)
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "available"

    out_phrases = [
        "out for season",
        "out indefinitely",
        "season ending",
        "season end",
        "season over",
        "miss the remainder",
        "will miss the rest",
        "ruled out",
        "not expected to play",
        "not playing",
        "left team",
        "quit the team",
        "redshirt",
        "suspension",
        "suspended",
        "inactive",
        "unavailable",
    ]
    questionable_phrases = [
        "questionable",
        "game time decision",
        "gametime decision",
        "day to day",
        "dtd",
        "gtd",
    ]
    probable_phrases = [
        "probable",
        "expected to play",
        "will play",
        "should play",
        "likely to play",
    ]
    available_phrases = [
        "available",
        "active",
        "cleared",
        "healthy",
    ]

    if any(phrase in text for phrase in out_phrases) or re.search(r"\bout\b", text):
        return "out"
    if "doubtful" in text:
        return "doubtful"
    if any(phrase in text for phrase in questionable_phrases):
        return "questionable"
    if any(phrase in text for phrase in probable_phrases):
        return "probable"
    if any(phrase in text for phrase in available_phrases):
        return "available"
    return "questionable"


def _empty_injury_data_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "team",
            "player",
            "position",
            "injury",
            "notes",
            "status_raw",
            "status",
            "report_date",
            "fetched_at_utc",
            "team_key",
            "player_key",
        ]
    )


def _empty_player_impact_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "player", "impact", "team_key", "player_key"])


def _empty_manual_override_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "team",
            "player",
            "status_override",
            "impact_override",
            "team_key",
            "player_key",
        ]
    )


def build_player_impact_table(
    players_history: pd.DataFrame | str | None,
    output_path: str | None = None,
    logger: logging.Logger | None = None,
    stability_window: int = 8,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    out_cols = ["team", "player", "impact"]
    empty_out = _empty_player_impact_df()[out_cols].copy()

    source_desc = "dataframe"
    source_loaded = False
    if isinstance(players_history, pd.DataFrame):
        hist = players_history.copy()
        source_loaded = True
    else:
        path = os.fspath(players_history or "")
        source_desc = os.path.abspath(path) if path else ""
        if not path:
            logger.warning("PLAYER_IMPACT_BUILD skipped reason=missing_players_history")
            return empty_out
        if not os.path.exists(path):
            logger.warning(
                "PLAYER_IMPACT_BUILD skipped "
                f"reason=players_history_missing path={os.path.abspath(path)}"
            )
            return empty_out
        try:
            hist = pd.read_csv(path)
            source_loaded = True
        except Exception as ex:
            logger.warning(
                "PLAYER_IMPACT_BUILD skipped "
                f"reason=players_history_read_failed path={os.path.abspath(path)} "
                f"error={type(ex).__name__}: {ex}"
            )
            return empty_out

    def _write_output(df: pd.DataFrame) -> None:
        if not output_path:
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(
            f"PLAYER_IMPACT_BUILD wrote rows={len(df)} path={os.path.abspath(output_path)}"
        )

    if hist is None or hist.empty:
        logger.warning("PLAYER_IMPACT_BUILD skipped reason=empty_players_history")
        if source_loaded:
            _write_output(empty_out)
        return empty_out

    try:
        team_col = _find_column(hist, ["team", "team_name", "school", "teamdisplayname"])
        player_col = _find_column(
            hist,
            ["player_name", "player", "athlete", "player_display_name"],
        )
        minutes_col = _find_column(hist, ["minutes", "minutes_float", "min", "mins"])
    except Exception as ex:
        logger.warning(
            "PLAYER_IMPACT_BUILD skipped "
            f"reason=missing_required_columns error={type(ex).__name__}: {ex}"
        )
        return empty_out

    game_id_col = _find_column(hist, ["game_id", "gameid"], required=False)
    game_date_col = _find_column(hist, ["game_date", "date"], required=False)
    opponent_col = _find_column(hist, ["opponent", "opp", "opponent_name"], required=False)
    opponent_team_id_col = _find_column(
        hist,
        ["opponent_team_id", "opponent_id"],
        required=False,
    )
    starter_col = _find_column(hist, ["starter", "is_starter"], required=False)

    points_col = _find_column(hist, ["points", "pts"], required=False)
    fga_col = _find_column(hist, ["fga", "field_goal_attempts"], required=False)
    fta_col = _find_column(hist, ["fta", "free_throw_attempts"], required=False)
    tov_col = _find_column(hist, ["tov", "turnovers"], required=False)
    ast_col = _find_column(hist, ["ast", "assists"], required=False)
    trb_col = _find_column(hist, ["trb", "rebounds", "total_rebounds"], required=False)
    stl_col = _find_column(hist, ["stl", "steals"], required=False)
    blk_col = _find_column(hist, ["blk", "blocks"], required=False)
    pf_col = _find_column(hist, ["pf", "fouls"], required=False)

    def _numeric_series(col_name: str | None) -> pd.Series:
        if col_name is None:
            return pd.Series(0.0, index=hist.index, dtype=float)
        return pd.to_numeric(hist[col_name], errors="coerce").fillna(0.0)

    def _stable_display(series: pd.Series) -> str:
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned[cleaned != ""]
        if cleaned.empty:
            return ""
        counts = cleaned.value_counts()
        top_count = int(counts.iloc[0])
        candidates = sorted(str(name) for name, count in counts.items() if count == top_count)
        return candidates[0] if candidates else ""

    def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return 0.0
        vv = v[mask].to_numpy(dtype=float)
        ww = w[mask].to_numpy(dtype=float)
        denom = float(ww.sum())
        return float((vv * ww).sum() / denom) if denom > 0 else 0.0

    def _scale_series(series: pd.Series, low_q: float = 0.35, high_q: float = 0.90) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if not s.notna().any():
            return pd.Series(0.0, index=series.index, dtype=float)
        low = float(s.quantile(low_q))
        high = float(s.quantile(high_q))
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        scaled = (s.fillna(low) - low) / (high - low)
        return scaled.clip(lower=0.0, upper=1.0)

    work = pd.DataFrame(
        {
            "team": hist[team_col].fillna("").astype(str).str.strip(),
            "player": hist[player_col].fillna("").astype(str).str.strip(),
            "minutes_float": _numeric_series(minutes_col).clip(lower=0.0),
        }
    )
    work["team_key"] = work["team"].apply(_team_name_key)
    work["player_key"] = work["player"].apply(normalize_player_name)
    work = work[
        (work["team_key"] != "")
        & (work["player_key"] != "")
        & (work["minutes_float"] > 0.0)
    ].copy()
    if work.empty:
        logger.warning("PLAYER_IMPACT_BUILD skipped reason=no_valid_player_minutes")
        if source_loaded:
            _write_output(empty_out)
        return empty_out

    if game_date_col is not None:
        work["game_date"] = pd.to_datetime(
            hist.loc[work.index, game_date_col],
            errors="coerce",
            utc=True,
        )
    else:
        work["game_date"] = pd.NaT

    game_key = (
        hist.loc[work.index, game_id_col].fillna("").astype(str).str.strip()
        if game_id_col is not None
        else pd.Series("", index=work.index, dtype="object")
    )
    if opponent_team_id_col is not None:
        opponent_fallback = hist.loc[work.index, opponent_team_id_col].fillna("").astype(str).str.strip()
    elif opponent_col is not None:
        opponent_fallback = hist.loc[work.index, opponent_col].fillna("").astype(str).str.strip()
    else:
        opponent_fallback = pd.Series("", index=work.index, dtype="object")
    date_fallback = (
        work["game_date"].dt.strftime("%Y-%m-%d").fillna("")
        if work["game_date"].notna().any()
        else pd.Series("", index=work.index, dtype="object")
    )
    work["game_key"] = game_key
    missing_game_key = work["game_key"] == ""
    work.loc[missing_game_key, "game_key"] = (
        date_fallback[missing_game_key] + "|" + opponent_fallback[missing_game_key]
    ).str.strip("|")

    work["starter_flag"] = 0.0
    if starter_col is not None:
        work["starter_flag"] = (
            hist.loc[work.index, starter_col].apply(to_bool_flag).astype(float)
        )

    work["points"] = _numeric_series(points_col).loc[work.index]
    work["fga"] = _numeric_series(fga_col).loc[work.index]
    work["fta"] = _numeric_series(fta_col).loc[work.index]
    work["tov"] = _numeric_series(tov_col).loc[work.index]
    work["ast"] = _numeric_series(ast_col).loc[work.index]
    work["trb"] = _numeric_series(trb_col).loc[work.index]
    work["stl"] = _numeric_series(stl_col).loc[work.index]
    work["blk"] = _numeric_series(blk_col).loc[work.index]
    work["pf"] = _numeric_series(pf_col).loc[work.index]

    rich_stats_available = (
        points_col is not None
        and fga_col is not None
        and fta_col is not None
        and tov_col is not None
        and any(col is not None for col in [ast_col, trb_col, stl_col, blk_col])
    )

    work["usage"] = work["fga"] + 0.475 * work["fta"] + work["tov"]
    work["box_score_prod"] = (
        work["points"]
        + 0.7 * work["ast"]
        + 0.5 * work["trb"]
        + 1.5 * work["stl"]
        + 1.5 * work["blk"]
        - work["tov"]
        - 0.3 * work["pf"]
    )
    work["efficiency_metric"] = np.where(
        work["usage"] > 0.0,
        work["points"] / work["usage"],
        work["points"] / np.maximum(work["minutes_float"], 1.0),
    )
    work["box_score_per40"] = work["box_score_prod"] * 40.0 / np.maximum(
        work["minutes_float"], 12.0
    )
    work["fallback_efficiency"] = work["points"] / np.maximum(work["minutes_float"], 1.0)
    work["fallback_box_per40"] = (
        work["points"]
        + work["ast"]
        + work["trb"]
        + work["stl"]
        + work["blk"]
    ) * 40.0 / np.maximum(work["minutes_float"], 12.0)

    stability_window = max(int(stability_window), 1)
    sort_cols = ["team_key", "player_key", "game_key"]
    ascending = [True, True, False]
    if work["game_date"].notna().any():
        sort_cols = ["team_key", "player_key", "game_date", "game_key"]
        ascending = [True, True, False, False]
    work = work.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    tail = work.groupby(["team_key", "player_key"], sort=False).head(stability_window)
    stability = (
        tail.groupby(["team_key", "player_key"])["minutes_float"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stability["std"] = stability["std"].fillna(0.0)
    stability["minutes_stability"] = 1.0 / (
        1.0 + (stability["std"] / np.maximum(stability["mean"], 1.0))
    )

    rows: list[dict] = []
    for (team_key, player_key), grp in work.groupby(["team_key", "player_key"], sort=False):
        game_keys = grp["game_key"].replace("", np.nan).dropna()
        games = int(game_keys.nunique()) if not game_keys.empty else int(len(grp))
        total_minutes = float(pd.to_numeric(grp["minutes_float"], errors="coerce").sum())
        weights = pd.to_numeric(grp["minutes_float"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "team": _stable_display(grp["team"]),
                "player": _stable_display(grp["player"]),
                "team_key": team_key,
                "player_key": player_key,
                "games": games,
                "total_minutes": total_minutes,
                "usage_total": float(pd.to_numeric(grp["usage"], errors="coerce").sum()),
                "box_total": float(
                    pd.to_numeric(grp["box_score_prod"], errors="coerce")
                    .fillna(0.0)
                    .clip(lower=0.0)
                    .sum()
                ),
                "efficiency_wavg": _weighted_avg(
                    grp["efficiency_metric"] if rich_stats_available else grp["fallback_efficiency"],
                    weights,
                ),
                "box_per40_wavg": _weighted_avg(
                    grp["box_score_per40"] if rich_stats_available else grp["fallback_box_per40"],
                    weights,
                ),
                "starter_rate": float(
                    pd.to_numeric(grp["starter_flag"], errors="coerce").fillna(0.0).mean()
                ),
            }
        )

    agg = pd.DataFrame(rows)
    if agg.empty:
        logger.warning("PLAYER_IMPACT_BUILD skipped reason=no_aggregated_players")
        if source_loaded:
            _write_output(empty_out)
        return empty_out

    agg = agg.merge(
        stability[["team_key", "player_key", "minutes_stability"]],
        on=["team_key", "player_key"],
        how="left",
    )
    agg["minutes_stability"] = (
        pd.to_numeric(agg["minutes_stability"], errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )
    agg["games"] = pd.to_numeric(agg["games"], errors="coerce").fillna(0).astype(int)
    agg["total_minutes"] = pd.to_numeric(agg["total_minutes"], errors="coerce").fillna(0.0)
    agg["usage_total"] = pd.to_numeric(agg["usage_total"], errors="coerce").fillna(0.0)
    agg["box_total"] = pd.to_numeric(agg["box_total"], errors="coerce").fillna(0.0)
    agg["efficiency_wavg"] = pd.to_numeric(agg["efficiency_wavg"], errors="coerce").fillna(0.0)
    agg["box_per40_wavg"] = pd.to_numeric(agg["box_per40_wavg"], errors="coerce").fillna(0.0)
    agg["starter_rate"] = (
        pd.to_numeric(agg["starter_rate"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )

    agg["minutes_share"] = agg["total_minutes"] / agg.groupby("team_key")["total_minutes"].transform("sum").replace(0, np.nan)
    agg["usage_share"] = agg["usage_total"] / agg.groupby("team_key")["usage_total"].transform("sum").replace(0, np.nan)
    agg["production_share"] = agg["box_total"] / agg.groupby("team_key")["box_total"].transform("sum").replace(0, np.nan)
    for col in ["minutes_share", "usage_share", "production_share"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    agg["scaled_minutes_share"] = (agg["minutes_share"] / 0.18).clip(lower=0.0, upper=1.0)
    agg["scaled_efficiency"] = _scale_series(agg["efficiency_wavg"])
    agg["scaled_box_score_production"] = _scale_series(agg["box_per40_wavg"])
    agg["scaled_team_importance"] = (
        (
            0.45 * agg["minutes_share"]
            + 0.30 * agg["usage_share"]
            + 0.25 * agg["production_share"]
        )
        / 0.18
    ).clip(lower=0.0, upper=1.0)
    agg["sample_strength"] = (
        (agg["total_minutes"] / np.maximum(agg["total_minutes"] + 120.0, 1.0))
        * (agg["games"] / 5.0).clip(lower=0.0, upper=1.0)
    ).clip(lower=0.0, upper=1.0)

    if rich_stats_available:
        agg["impact"] = 6.0 * agg["sample_strength"] * (
            0.35 * agg["scaled_minutes_share"]
            + 0.20 * agg["scaled_efficiency"]
            + 0.18 * agg["scaled_box_score_production"]
            + 0.15 * agg["scaled_team_importance"]
            + 0.07 * agg["minutes_stability"]
            + 0.05 * agg["starter_rate"]
        )
        formula_name = "rich"
    else:
        agg["impact"] = 6.0 * agg["sample_strength"] * (
            (
                agg["scaled_minutes_share"]
                + agg["scaled_efficiency"]
                + agg["scaled_box_score_production"]
            )
            / 3.0
        )
        formula_name = "fallback"

    agg["impact"] = (
        pd.to_numeric(agg["impact"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=6.0)
        .round(3)
    )

    out = (
        agg.sort_values(["team", "impact", "player"], ascending=[True, False, True])
        .reset_index(drop=True)[out_cols]
    )
    _write_output(out)

    teams_with_five = int((out.groupby("team")["impact"].apply(lambda s: (s > 0).sum()) >= 5).sum())
    top20 = (
        out.sort_values(["impact", "team", "player"], ascending=[False, True, True])
        .head(20)
        .to_dict(orient="records")
    )
    logger.info(
        "PLAYER_IMPACT_BUILD scored "
        f"players={len(out)} teams={out['team'].nunique()} "
        f"teams_with_at_least_5={teams_with_five} "
        f"formula={formula_name} source={source_desc}"
    )
    logger.info(f"PLAYER_IMPACT_BUILD top20={top20}")
    return out


def _dedupe_injury_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_injury_data_df()

    out = df.copy()
    out["__status_priority"] = (
        out["status"].map(PLAYER_STATUS_PRIORITY).fillna(PLAYER_STATUS_PRIORITY["questionable"])
    )
    out = out.sort_values(
        ["__status_priority", "source", "team", "player"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    out = out.drop_duplicates(subset=["team_key", "player_key"], keep="first").reset_index(
        drop=True
    )
    return out.drop(columns=["__status_priority"])


def fetch_rotowire_injury_data(logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    if requests is None:
        raise RuntimeError("requests package is required for injury ingestion")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }
    params = {
        "team": "other",
        "pos": "ALL",
        "conf": "ALL",
        "site": "other",
        "slateID": "",
    }
    response = requests.get(
        ROTOWIRE_CBB_INJURY_URL,
        params=params,
        headers=headers,
        timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f"Unexpected Rotowire injury payload type={type(payload).__name__}"
        )

    fetched_at_utc = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    rows: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        team = _clean_text(item.get("team"))
        player = _clean_text(item.get("player"))
        if not team or not player:
            continue
        status_raw = _clean_html_text(item.get("status"))
        injury = _clean_html_text(item.get("injury"))
        notes = _clean_html_text(item.get("rDate"))
        rows.append(
            {
                "source": "rotowire",
                "team": team,
                "player": player,
                "position": _clean_text(item.get("position")),
                "injury": injury,
                "notes": notes,
                "status_raw": status_raw,
                "status": _standardize_player_status(status_raw, injury, notes),
                "report_date": notes,
                "fetched_at_utc": fetched_at_utc,
                "team_key": _team_name_key(team),
                "player_key": normalize_player_name(player),
            }
        )

    out = _empty_injury_data_df() if not rows else pd.DataFrame(rows)
    out = _dedupe_injury_rows(out)
    logger.info(f"PLAYER_STATUS_FETCH source=rotowire rows={len(out)}")
    return out


def fetch_espn_injury_data(logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    if requests is None:
        raise RuntimeError("requests package is required for injury ingestion")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
    }
    response = requests.get(
        ESPN_CBB_INJURY_URL,
        headers=headers,
        timeout=(HTTP_TIMEOUT_CONNECT_S, HTTP_TIMEOUT_READ_S),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Unexpected ESPN injury payload type={type(payload).__name__}"
        )

    fetched_at_utc = _clean_text(payload.get("timestamp"))
    if not fetched_at_utc:
        fetched_at_utc = (
            datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        )

    rows: list[dict] = []
    for team_entry in payload.get("injuries", []) or []:
        if not isinstance(team_entry, dict):
            continue
        team_name = _clean_text(team_entry.get("displayName"))
        for injury_entry in team_entry.get("injuries", []) or []:
            if not isinstance(injury_entry, dict):
                continue
            athlete = injury_entry.get("athlete", {})
            athlete = athlete if isinstance(athlete, dict) else {}
            athlete_team = athlete.get("team", {})
            athlete_team = athlete_team if isinstance(athlete_team, dict) else {}
            athlete_position = athlete.get("position", {})
            athlete_position = (
                athlete_position if isinstance(athlete_position, dict) else {}
            )

            player = _clean_text(
                athlete.get("displayName")
                or athlete.get("fullName")
                or athlete.get("shortName")
            )
            if not player:
                continue

            resolved_team = _clean_text(athlete_team.get("displayName")) or team_name
            short_comment = _clean_text(injury_entry.get("shortComment"))
            long_comment = _clean_text(injury_entry.get("longComment"))
            notes = " ".join(x for x in [short_comment, long_comment] if x).strip()
            injury = _clean_text(
                injury_entry.get("type")
                or injury_entry.get("detail")
                or injury_entry.get("displayName")
            )
            status_raw = _clean_text(injury_entry.get("status"))

            rows.append(
                {
                    "source": "espn",
                    "team": resolved_team,
                    "player": player,
                    "position": _clean_text(athlete_position.get("abbreviation")),
                    "injury": injury,
                    "notes": notes,
                    "status_raw": status_raw,
                    "status": _standardize_player_status(status_raw, injury, notes),
                    "report_date": _clean_text(injury_entry.get("date")),
                    "fetched_at_utc": fetched_at_utc,
                    "team_key": _team_name_key(resolved_team),
                    "player_key": normalize_player_name(player),
                }
            )

    out = _empty_injury_data_df() if not rows else pd.DataFrame(rows)
    out = _dedupe_injury_rows(out)
    logger.info(f"PLAYER_STATUS_FETCH source=espn rows={len(out)}")
    return out


def fetch_injury_data(logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    failures: list[str] = []

    for name, fetcher in [
        ("rotowire", fetch_rotowire_injury_data),
        ("espn", fetch_espn_injury_data),
    ]:
        try:
            df = fetcher(logger=logger)
        except Exception as ex:
            failures.append(f"{name}:{type(ex).__name__}: {ex}")
            logger.warning(
                f"PLAYER_STATUS_FETCH source={name} status=ERROR "
                f"error={type(ex).__name__}: {ex}"
            )
            continue
        if not df.empty:
            logger.info(f"PLAYER_STATUS_FETCH source={name} status=OK rows={len(df)}")
            return df
        logger.info(f"PLAYER_STATUS_FETCH source={name} status=EMPTY rows=0")

    if failures:
        logger.warning(
            "PLAYER_STATUS_FETCH all_sources_failed "
            f"errors={failures}"
        )
    return _empty_injury_data_df()


def load_player_impacts(path: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    if not os.path.exists(path):
        logger.warning(f"PLAYER_IMPACT missing path={os.path.abspath(path)}")
        return _empty_player_impact_df()

    df = pd.read_csv(path)
    required = {"team", "player", "impact"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"PLAYER_IMPACT missing required columns={missing} path={os.path.abspath(path)}"
        )

    out = df.copy()
    out["team"] = out["team"].fillna("").astype(str).str.strip()
    out["player"] = out["player"].fillna("").astype(str).str.strip()
    out["impact"] = pd.to_numeric(out["impact"], errors="coerce").fillna(0.0)
    out = out[(out["team"] != "") & (out["player"] != "")].copy()
    out["team_key"] = out["team"].apply(_team_name_key)
    out["player_key"] = out["player"].apply(normalize_player_name)
    out = out[(out["team_key"] != "") & (out["player_key"] != "")].copy()
    out = out.drop_duplicates(subset=["team_key", "player_key"], keep="last").reset_index(
        drop=True
    )
    logger.info(
        f"PLAYER_IMPACT loaded rows={len(out)} path={os.path.abspath(path)}"
    )
    return out[["team", "player", "impact", "team_key", "player_key"]]


def load_manual_overrides(path: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    if not os.path.exists(path):
        logger.info(f"PLAYER_OVERRIDE missing path={os.path.abspath(path)}")
        return _empty_manual_override_df()

    df = pd.read_csv(path)
    required = {"team", "player"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"PLAYER_OVERRIDE missing required columns={missing} path={os.path.abspath(path)}"
        )
    if "status" not in df.columns and "impact" not in df.columns:
        raise RuntimeError(
            "PLAYER_OVERRIDE requires at least one of ['status', 'impact'] "
            f"path={os.path.abspath(path)}"
        )

    out = df.copy()
    out["team"] = out["team"].fillna("").astype(str).str.strip()
    out["player"] = out["player"].fillna("").astype(str).str.strip()
    if "status" in out.columns:
        out["status_override"] = out["status"].fillna("").astype(str).str.strip()
    else:
        out["status_override"] = ""
    if "impact" in out.columns:
        out["impact_override"] = pd.to_numeric(out["impact"], errors="coerce")
    else:
        out["impact_override"] = np.nan

    out = out[
        (out["team"] != "")
        & (out["player"] != "")
        & ((out["status_override"] != "") | out["impact_override"].notna())
    ].copy()
    out["team_key"] = out["team"].apply(_team_name_key)
    out["player_key"] = out["player"].apply(normalize_player_name)
    out = out[(out["team_key"] != "") & (out["player_key"] != "")].copy()
    out = out.drop_duplicates(subset=["team_key", "player_key"], keep="last").reset_index(
        drop=True
    )
    logger.info(
        f"PLAYER_OVERRIDE loaded rows={len(out)} path={os.path.abspath(path)}"
    )
    return out[
        [
            "team",
            "player",
            "status_override",
            "impact_override",
            "team_key",
            "player_key",
        ]
    ]


def build_player_adjustments(
    project_root: str,
    out_dir: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = logger or logging.getLogger("ncaab_ranker")
    data_dir = os.path.join(project_root, "data")
    impacts_path = os.path.join(data_dir, "player_impact.csv")
    overrides_path = os.path.join(data_dir, "manual_overrides.csv")
    output_path = os.path.join(out_dir, "player_adjustments.csv")
    players_history_path = os.path.join(out_dir, "players_history.csv")

    try:
        build_player_impact_table(
            players_history=players_history_path,
            output_path=impacts_path,
            logger=logger,
        )
    except Exception as ex:
        logger.warning(
            "PLAYER_IMPACT_BUILD failed "
            f"path={os.path.abspath(players_history_path)} "
            f"error={type(ex).__name__}: {ex}"
        )

    impacts_df = load_player_impacts(impacts_path, logger)
    overrides_df = load_manual_overrides(overrides_path, logger)
    injuries_df = fetch_injury_data(logger)

    key_frames: list[pd.DataFrame] = []
    if not impacts_df.empty:
        key_frames.append(
            impacts_df[["team_key", "player_key", "team", "player"]].assign(__priority=0)
        )
    if not overrides_df.empty:
        key_frames.append(
            overrides_df[["team_key", "player_key", "team", "player"]].assign(__priority=1)
        )
    if not injuries_df.empty:
        key_frames.append(
            injuries_df[["team_key", "player_key", "team", "player"]].assign(__priority=2)
        )

    if key_frames:
        base = pd.concat(key_frames, ignore_index=True)
        base = base[(base["team_key"] != "") & (base["player_key"] != "")].copy()
        base = (
            base.sort_values(["__priority", "team", "player"], ascending=[True, True, True])
            .drop_duplicates(subset=["team_key", "player_key"], keep="first")
            .drop(columns=["__priority"])
            .reset_index(drop=True)
        )
    else:
        base = pd.DataFrame(columns=["team_key", "player_key", "team", "player"])

    impacts_merge = impacts_df.rename(
        columns={
            "team": "impact_team",
            "player": "impact_player",
            "impact": "impact_base",
        }
    )
    overrides_merge = overrides_df.rename(
        columns={
            "team": "override_team",
            "player": "override_player",
        }
    )
    injuries_merge = injuries_df.rename(
        columns={
            "team": "injury_team",
            "player": "injury_player",
            "source": "injury_source",
        }
    )

    merged = base.merge(
        impacts_merge[["team_key", "player_key", "impact_base"]],
        on=["team_key", "player_key"],
        how="left",
    )
    merged = merged.merge(
        injuries_merge[
            [
                "team_key",
                "player_key",
                "injury_source",
                "position",
                "injury",
                "notes",
                "status_raw",
                "status",
                "report_date",
                "fetched_at_utc",
            ]
        ],
        on=["team_key", "player_key"],
        how="left",
    )
    merged = merged.merge(
        overrides_merge[
            ["team_key", "player_key", "status_override", "impact_override"]
        ],
        on=["team_key", "player_key"],
        how="left",
    )

    merged["impact_base"] = pd.to_numeric(merged["impact_base"], errors="coerce")
    merged["impact_override"] = pd.to_numeric(merged["impact_override"], errors="coerce")

    override_status = merged["status_override"].fillna("").astype(str).str.strip()
    has_override_status = override_status != ""
    has_override_impact = merged["impact_override"].notna()

    merged["status"] = merged["status"].fillna("").astype(str).str.strip()
    merged["status_raw"] = merged["status_raw"].fillna("").astype(str).str.strip()
    merged.loc[has_override_status, "status_raw"] = override_status[has_override_status]
    merged.loc[has_override_status, "status"] = override_status[has_override_status].apply(
        _standardize_player_status
    )
    merged.loc[merged["status"] == "", "status"] = "available"
    merged.loc[merged["status_raw"] == "", "status_raw"] = (
        merged.loc[merged["status_raw"] == "", "status"].str.title()
    )

    merged["impact"] = merged["impact_base"]
    merged.loc[has_override_impact, "impact"] = merged.loc[has_override_impact, "impact_override"]
    merged["impact"] = pd.to_numeric(merged["impact"], errors="coerce").fillna(0.0)
    merged["impact_missing"] = merged["impact_base"].isna() & (~has_override_impact)
    merged["status_weight"] = (
        merged["status"].map(PLAYER_STATUS_WEIGHTS).fillna(PLAYER_STATUS_WEIGHTS["questionable"])
    )
    merged["impact_contribution"] = (
        pd.to_numeric(merged["impact"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged["status_weight"], errors="coerce").fillna(0.0)
    )
    merged["override_applied"] = has_override_status | has_override_impact
    merged["status_source"] = np.where(
        has_override_status,
        "manual_override",
        np.where(
            merged["injury_source"].fillna("").astype(str).str.strip() != "",
            merged["injury_source"].fillna("").astype(str).str.strip(),
            "default_available",
        ),
    )
    merged["impact_source"] = np.where(
        has_override_impact,
        "manual_override",
        np.where(merged["impact_missing"], "missing", "player_impact"),
    )

    if merged.empty:
        team_totals = pd.DataFrame(
            columns=["team_key", "team", "impact_total", "impacted_players"]
        )
    else:
        team_totals = (
            merged.groupby("team_key", as_index=False)
            .agg(
                team=(
                    "team",
                    lambda s: next((_clean_text(v) for v in s if _clean_text(v)), ""),
                ),
                impact_total=("impact_contribution", "sum"),
                impacted_players=(
                    "impact_contribution",
                    lambda s: int(
                        (pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum()
                    ),
                ),
            )
        )
        team_totals["impact_total"] = pd.to_numeric(
            team_totals["impact_total"], errors="coerce"
        ).fillna(0.0)

    team_total_map = (
        team_totals.set_index("team_key")["impact_total"].to_dict()
        if not team_totals.empty
        else {}
    )
    merged["impact_total"] = merged["team_key"].map(team_total_map).fillna(0.0)

    out = merged[
        [
            "team",
            "player",
            "injury_source",
            "status_source",
            "status_raw",
            "status",
            "status_weight",
            "injury",
            "notes",
            "report_date",
            "impact_base",
            "impact",
            "impact_missing",
            "impact_source",
            "override_applied",
            "impact_contribution",
            "impact_total",
        ]
    ].rename(columns={"injury_source": "source"})
    out = out.sort_values(
        ["impact_total", "impact_contribution", "team", "player"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(
        "PLAYER_ADJUSTMENTS wrote "
        f"rows={len(out)} teams={len(team_totals)} "
        f"path={os.path.abspath(output_path)}"
    )
    return out, team_totals


def apply_player_adjustments_to_ratings(
    ratings_df: pd.DataFrame,
    team_adjustments_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger("ncaab_ranker")
    if ratings_df is None or ratings_df.empty:
        return ratings_df

    out = ratings_df.copy()
    out["__team_key"] = out["team"].apply(_team_name_key)

    team_totals = (
        team_adjustments_df[["team_key", "impact_total", "team"]].drop_duplicates("team_key")
        if team_adjustments_df is not None and not team_adjustments_df.empty
        else pd.DataFrame(columns=["team_key", "impact_total", "team"])
    )
    if not team_totals.empty:
        out = out.merge(
            team_totals[["team_key", "impact_total"]],
            left_on="__team_key",
            right_on="team_key",
            how="left",
        )
    else:
        out["impact_total"] = np.nan

    out["impact_total"] = pd.to_numeric(out["impact_total"], errors="coerce").fillna(0.0)
    raw_adjem = pd.to_numeric(out["AdjEM"], errors="coerce").fillna(0.0)
    adjusted_adjem = (raw_adjem - out["impact_total"]).round(3)
    impact_total = out["impact_total"].round(3)

    if "team_key" in out.columns:
        out = out.drop(columns=["team_key"])
    if "impact_total" in out.columns:
        out = out.drop(columns=["impact_total"])
    if "AdjEM_adj" in out.columns:
        out = out.drop(columns=["AdjEM_adj"])

    if "AdjEM" in out.columns:
        insert_at = out.columns.get_loc("AdjEM") + 1
        out.insert(insert_at, "impact_total", impact_total)
        out.insert(insert_at + 1, "AdjEM_adj", adjusted_adjem)
    else:
        out["impact_total"] = impact_total
        out["AdjEM_adj"] = adjusted_adjem

    adjusted_team_count = int((out["impact_total"] > 0).sum())
    unmatched = pd.DataFrame()
    if not team_totals.empty:
        unmatched = team_totals[
            (pd.to_numeric(team_totals["impact_total"], errors="coerce").fillna(0.0) > 0.0)
            & (~team_totals["team_key"].isin(set(out["__team_key"])))
        ]
    if not unmatched.empty:
        sample = unmatched[["team", "impact_total"]].head(10).to_dict(orient="records")
        logger.warning(
            "PLAYER_ADJUSTMENTS unmatched_teams "
            f"count={len(unmatched)} sample={sample}"
        )
    logger.info(
        f"PLAYER_ADJUSTMENTS applied adjusted_team_count={adjusted_team_count} "
        f"ratings_rows={len(out)}"
    )
    return out.drop(columns=["__team_key"])


def validate_games_df(g: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if g is None or g.empty:
        logger.info("VALIDATE: games df empty")
        return g

    def to_bool(x) -> bool:
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

    needed = [
        "game_id","game_date","team","opponent","location",
        "pts_for","pts_against","fga","fta","orb","to",
        "opp_fga","opp_fta","opp_orb","opp_to"
    ]
    missing = [c for c in needed if c not in g.columns]
    if missing:
        logger.info(f"VALIDATE: missing columns: {missing}")

    # Coerce numeric columns
    num_cols = ["pts_for","pts_against","fga","fta","orb","to","opp_fga","opp_fta","opp_orb","opp_to"]
    for c in num_cols:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(int)

    # Parse date
    if "game_date" in g.columns:
        g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    else:
        g["game_date"] = pd.NaT

    if "team_is_d1" in g.columns and "is_d1_team" not in g.columns:
        g["is_d1_team"] = g["team_is_d1"]
    if "opp_is_d1" in g.columns and "is_d1_opponent" not in g.columns:
        g["is_d1_opponent"] = g["opp_is_d1"]

    for c in ["is_d1_team", "is_d1_opponent"]:
        if c in g.columns:
            g[c] = g[c].apply(to_bool)

    for c in ["team_id", "opponent_team_id"]:
        if c in g.columns:
            g[c] = g[c].apply(_safe_team_id)

    # Drop junk rows
    before = len(g)
    g = g.dropna(subset=["game_id","team","opponent","game_date"])
    after = len(g)
    if after != before:
        logger.info(f"VALIDATE: dropped {before-after} rows with missing ids/names/dates")

    # Dedup again for safety
    before = len(g)
    if "team_id" in g.columns:
        g = g.drop_duplicates(subset=["game_id","team_id"], keep="last")
        removed = before - len(g)
        if removed:
            logger.info(f"VALIDATE: removed {removed} duplicate rows by (game_id,team_id)")
    else:
        g = g.drop_duplicates(subset=["game_id","team"], keep="last")
        removed = before - len(g)
        if removed:
            logger.info(f"VALIDATE: removed {removed} duplicate rows by (game_id,team)")

    # Quick stats quality check (zeros)
    if "fga" in g.columns:
        zero_fga = int((g["fga"] == 0).sum())
        logger.info(f"VALIDATE: fga==0 rows={zero_fga}/{len(g)}")

    return g


def filter_strict_d1_games(
    g: pd.DataFrame,
    logger: logging.Logger,
    label: str = "GAMES",
) -> pd.DataFrame:
    if g is None or g.empty:
        return g
    if "is_d1_team" not in g.columns or "is_d1_opponent" not in g.columns:
        logger.warning(f"{label} strict_d1 skipped missing_d1_flags")
        return g
    before = len(g)
    out = g[
        g["is_d1_team"].fillna(False)
        & g["is_d1_opponent"].fillna(False)
    ].copy()
    logger.info(f"{label} strict_d1 rows_before={before} rows_after={len(out)}")
    return out

def games_date_bounds(games_df: pd.DataFrame) -> tuple[date | None, date | None]:
    if games_df is None or games_df.empty or "game_date" not in games_df.columns:
        return None, None
    dates = pd.to_datetime(games_df["game_date"], errors="coerce")
    if not dates.notna().any():
        return None, None
    return dates.min().date(), dates.max().date()

def fetch_games_range(
    client: SourceClient,
    season: int,
    logger: logging.Logger,
    start_day: date,
    end_day: date,
    sleep_s: float,
    d1_team_ids: set[str],
    d1_team_meta: dict,
    label: str,
    max_boxscores_per_day: int | None = None,
    player_rows_sink: list[dict] | None = None,
    debug_out_dir: str | None = None,
) -> pd.DataFrame:
    if start_day > end_day:
        return pd.DataFrame()
    range_days = (end_day - start_day).days + 1
    if label.upper().startswith("INCREMENTAL") and range_days > 14:
        logger.warning(
            f"{label} range unusually large days={range_days} start_day={start_day} end_day={end_day}"
        )

    all_new = []
    d = start_day
    progress = {
        "pbar": None,
        "games_done": 0,
        "games_total_est": 0,
        "d1_team_ids": d1_team_ids,
        "d1_team_meta": d1_team_meta,
    }
    heartbeat = HeartbeatState(logger=logger, phase=label)
    days_looped = 0

    logger.info(f"{label} fetch from {start_day} to {end_day}")
    while d <= end_day:
        days_looped += 1
        heartbeat.touch(day=d, force=True)
        day_df = fetch_games_stub(
            client,
            season,
            logger,
            day=d,
            progress=progress,
            phase=label,
            heartbeat=heartbeat,
            max_boxscores=max_boxscores_per_day,
            player_rows_sink=player_rows_sink,
            debug_out_dir=debug_out_dir,
        )
        logger.info(f"{label} CHECK games rows={len(day_df)}")
        if not day_df.empty and "game_date" in day_df.columns:
            logger.info(
                f"{label} CHECK games date range={day_df['game_date'].min()} to {day_df['game_date'].max()}"
            )
        day_df = validate_games_df(day_df, logger)
        day_rows = 0 if day_df is None else len(day_df)
        day_games = 0 if day_df is None or day_df.empty else int(day_df["game_id"].nunique())
        df_min = None
        df_max = None
        if day_df is not None and not day_df.empty:
            df_dates = pd.to_datetime(day_df["game_date"], errors="coerce")
            if df_dates.notna().any():
                df_min = df_dates.min().date()
                df_max = df_dates.max().date()

        logger.info(
            f"{label}_DAILY_CHECK loop_date={d} df_min={df_min} df_max={df_max} "
            f"rows={day_rows} unique_games={day_games}"
        )
        heartbeat.touch(day=d, processed_inc=1, fetched_inc=1, force=True)
        if day_df is not None and not day_df.empty:
            all_new.append(day_df)
        if sleep_s > 0:
            time.sleep(sleep_s)
        d = (pd.Timestamp(d) + pd.Timedelta(days=1)).date()

    if progress["pbar"] is not None:
        progress["pbar"].close()

    games_new = pd.concat(all_new, ignore_index=True) if all_new else pd.DataFrame()
    games_new = validate_games_df(games_new, logger)
    if PLAYER_DEBUG and player_rows_sink is not None:
        logger.info(
            f"PLAYER_SINK_SUMMARY label={label} rows_in_sink={len(player_rows_sink)} "
            f"days={days_looped}"
        )
    if not games_new.empty:
        logger.info(
            f"{label}_SANITY rows={len(games_new)} unique_games={games_new['game_id'].nunique()} "
            f"date_range={games_new['game_date'].min()}..{games_new['game_date'].max()} "
            f"unique_teams={games_new['team'].nunique()}"
        )
    return games_new


def run_once(
    cfg_path: str,
    base_url: str | None = None,
    force_player_backfill: bool = False,
):
    global PLAYER_DEBUG, PLAYER_DEBUG_DUMP, _PLAYER_DUMPED_ONCE
    cfg = load_config(cfg_path)
    project_root = _resolve_project_root_from_cfg(cfg_path)
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "run.log"))
    PLAYER_DEBUG = to_bool_flag(os.environ.get("NCAAB_PLAYER_DEBUG", False))
    PLAYER_DEBUG_DUMP = to_bool_flag(os.environ.get("NCAAB_PLAYER_DEBUG_DUMP", False))
    _PLAYER_DUMPED_ONCE = False
    if PLAYER_DEBUG:
        logger.info(f"PLAYER_DEBUG enabled=true dump={PLAYER_DEBUG_DUMP}")

    run_id = generate_run_id()
    config_hash = hash_config(cfg)
    logger.info(f"RUN_START run_id={run_id} config_hash={config_hash}")
    if force_player_backfill:
        logger.info("RUN_FLAG backfill_players=true")

    src_base = str(base_url or os.environ.get("NCAA_API_BASE_URL", "")).strip()
    client = SourceClient(base_url=src_base)
    season = int(cfg.get("season", date.today().year))

    logger.info(
        f"Refresh start. season={season} base_url={src_base or 'unset'}"
    )
    hist_path = os.path.join(out_dir, "games_history.csv")
    today = datetime.now(tz=NY).date()
    refresh_cfg = cfg.get("refresh", {})
    backfill_cfg = refresh_cfg.get("backfill", {})
    backfill_start_str = str(backfill_cfg.get("start_date", SEASON_START.isoformat()))
    try:
        configured_start_day = date.fromisoformat(backfill_start_str)
    except Exception:
        configured_start_day = SEASON_START
        logger.warning(
            f"Invalid refresh.backfill.start_date={backfill_start_str!r}; "
            f"using fallback {configured_start_day}"
        )
    season_start_day = SEASON_START
    if configured_start_day != season_start_day:
        logger.info(
            "SEASON_START override "
            f"configured={configured_start_day} enforced={season_start_day}"
        )

    rebuilt_hist_df = rebuild_full_season_games(
        logger=logger,
        start_date=season_start_day,
        end_date=today,
        out_dir=out_dir,
    )

    hist_exists = os.path.exists(hist_path)
    hist_min_day = None
    hist_max_day = None
    if rebuilt_hist_df is not None and not rebuilt_hist_df.empty:
        hist = rebuilt_hist_df
        hist_exists = True
        hist_min_day, hist_max_day = games_date_bounds(hist)
    elif hist_exists:
        hist = pd.read_csv(hist_path)
        hist_min_day, hist_max_day = games_date_bounds(hist)

    needs_season_backfill = (not hist_exists) or (hist_min_day is None) or (hist_min_day > season_start_day)
    if hist_max_day is not None:
        # buffer 2 days back for stat corrections
        start_day = (pd.Timestamp(hist_max_day) - pd.Timedelta(days=2)).date()
    else:
        start_day = season_start_day
    start_day = max(start_day, season_start_day)

    d1_rebuild_mode_raw = str(refresh_cfg.get("rebuild_d1_ids_mode", "season_to_date"))
    d1_rebuild_mode = d1_rebuild_mode_raw.strip().lower()
    if d1_rebuild_mode not in {"incremental", "season_to_date"}:
        logger.warning(
            f"Invalid refresh.rebuild_d1_ids_mode={d1_rebuild_mode_raw!r}; "
            "defaulting to season_to_date"
        )
        d1_rebuild_mode = "season_to_date"

    d1_rebuild_days_limit_raw = refresh_cfg.get("rebuild_d1_ids_days_limit", None)
    d1_rebuild_days_limit = None
    if d1_rebuild_days_limit_raw is not None:
        try:
            d1_rebuild_days_limit = int(d1_rebuild_days_limit_raw)
            if d1_rebuild_days_limit <= 0:
                d1_rebuild_days_limit = None
        except Exception:
            logger.warning(
                f"Invalid refresh.rebuild_d1_ids_days_limit={d1_rebuild_days_limit_raw!r}; ignoring"
            )
            d1_rebuild_days_limit = None

    diagnostic_mode = to_bool_flag(
        os.environ.get("NCAAB_DIAGNOSTIC_MODE", refresh_cfg.get("diagnostic_mode", False))
    )
    verify_d1_mode = to_bool_flag(os.environ.get("NCAAB_VERIFY_D1", False))
    diagnostic_max_boxscores = 5 if diagnostic_mode else None
    diagnostic_day = start_day if start_day <= today else today
    if diagnostic_mode:
        logger.warning(
            f"DIAGNOSTIC_MODE enabled day={diagnostic_day} max_boxscores={diagnostic_max_boxscores}. "
            "Run will exit after quick diagnostic summary."
        )

    d1_source_raw = str(refresh_cfg.get("d1_source", "whitelist")).strip().lower()
    if d1_source_raw not in {"whitelist", "dynamic"}:
        logger.warning(
            f"Invalid refresh.d1_source={d1_source_raw!r}; defaulting to whitelist"
        )
        d1_source_raw = "whitelist"
    d1_source = d1_source_raw
    allow_dynamic_d1_rebuild = to_bool_flag(refresh_cfg.get("allow_dynamic_d1_rebuild", False))

    d1_whitelist_relpath = str(
        refresh_cfg.get("d1_whitelist_path", DEFAULT_D1_WHITELIST_RELPATH)
    ).strip()
    if not d1_whitelist_relpath:
        d1_whitelist_relpath = DEFAULT_D1_WHITELIST_RELPATH
    if os.path.isabs(d1_whitelist_relpath):
        d1_whitelist_path = d1_whitelist_relpath
    else:
        d1_whitelist_path = os.path.join(project_root, d1_whitelist_relpath)

    whitelist_required = d1_source == "whitelist"
    dynamic_rebuild_allowed = (d1_source == "dynamic") or allow_dynamic_d1_rebuild
    master_whitelist_rows: list[dict[str, str]] = []
    if whitelist_required:
        master_whitelist_rows = load_d1_master_whitelist(
            d1_whitelist_path, logger, expected_season=season
        )
        if len(master_whitelist_rows) < 300:
            raise RuntimeError(
                "D1 whitelist required but file is missing/invalid/too small. "
                f"path={d1_whitelist_path} loaded_rows={len(master_whitelist_rows)} expected>=300. "
                "To use dynamic IDs instead, set refresh.d1_source='dynamic'."
            )

    logger.info(
        "LOOP_PLAN "
        f"season_start_day={season_start_day} hist_min_day={hist_min_day} hist_max_day={hist_max_day} "
        f"incremental_start_day={start_day} d1_rebuild_mode={d1_rebuild_mode} "
        f"d1_rebuild_days_limit={d1_rebuild_days_limit} diagnostic_mode={diagnostic_mode} "
        f"d1_source={d1_source} allow_dynamic_d1_rebuild={allow_dynamic_d1_rebuild} "
        f"dynamic_rebuild_allowed={dynamic_rebuild_allowed} d1_whitelist_path={d1_whitelist_path}"
    )

    # Team directory cache used for historical ID backfill.
    team_directory = load_team_directory_cache(out_dir)
    team_dir_test_days_raw = cfg.get("refresh", {}).get("team_dir_test_days", None)
    team_dir_test_days = None
    if team_dir_test_days_raw is not None:
        try:
            team_dir_test_days = int(team_dir_test_days_raw)
            if team_dir_test_days <= 0:
                team_dir_test_days = None
        except Exception:
            logger.warning(f"TEAM_DIR invalid refresh.team_dir_test_days={team_dir_test_days_raw!r}; ignoring")
            team_dir_test_days = None

    team_dir_start = None
    team_dir_end = today
    if team_dir_test_days is not None:
        team_dir_start = today - timedelta(days=team_dir_test_days)
        logger.info(
            f"TEAM_DIR test slice enabled days={team_dir_test_days} "
            f"range={team_dir_start}..{team_dir_end}"
        )
    else:
        team_dir_start = season_start_day

    force_team_dir_build = to_bool_flag(refresh_cfg.get("force_team_directory_build", False))
    need_team_dir_build = (not diagnostic_mode) and (
        (team_dir_test_days is not None)
        or force_team_dir_build
        or ((d1_source != "whitelist") and (len(team_directory) < 300))
    )
    if not need_team_dir_build:
        if diagnostic_mode:
            logger.info(
                f"TEAM_DIR diagnostic_mode skip build; using cached directory teams={len(team_directory)}"
            )
        elif d1_source == "whitelist" and len(team_directory) < 300:
            logger.info(
                "TEAM_DIR whitelist mode skip full-season build "
                f"teams={len(team_directory)} force_team_directory_build={force_team_dir_build}"
            )
        else:
            logger.info(f"TEAM_DIR using cached directory teams={len(team_directory)}")
    else:
        if team_dir_test_days is None:
            logger.info(
                f"TEAM_DIR cache small (teams={len(team_directory)}). "
                f"Building full-season directory from {team_dir_start} to {team_dir_end}"
            )
        team_directory, td_stats = build_team_directory_full_season(
            client, season, logger, team_dir_start, team_dir_end, return_stats=True
        )
        write_team_directory_cache(out_dir, team_directory, logger)
        team_name_map_seed = build_team_directory_names_map(team_directory)
        write_team_directory_names_cache(out_dir, team_name_map_seed, logger)
        logger.info(
            "TEAM_DIR build summary "
            f"range={team_dir_start}..{team_dir_end} "
            f"total_teams={td_stats.get('total_teams')} "
            f"total_game_ids_fetched={td_stats.get('total_game_ids_fetched')} "
            f"fetch_failures={td_stats.get('total_failures')} "
            f"invalid_team_ids={td_stats.get('total_invalid_team_ids')}"
        )
        if len(team_directory) < 300:
            logger.warning(f"TEAM_DIR still small after build teams={len(team_directory)}")

    d1_team_meta: dict = {}

    cached_team_dir = load_team_directory_cache(out_dir)
    if cached_team_dir:
        d1_team_meta.update(cached_team_dir)
        logger.info(f"TEAM_DIR loaded cached teams={len(cached_team_dir)}")

    d1_conf_by_id: dict[str, str] = {}
    conf_map: dict[str, str] = {}
    whitelist_resolved_ids: set[str] = set()
    master_stats: dict[str, int] = {
        "master_rows": 0,
        "matched_rows": 0,
        "ambiguous_rows": 0,
        "unresolved_rows": 0,
    }
    if whitelist_required and not allow_dynamic_d1_rebuild:
        d1_team_ids = load_d1_team_ids_cache(out_dir)
        d1_team_ids = {tid for tid in (canonical_team_id(x) for x in d1_team_ids) if tid}
        logger.info(
            "D1_PRELOAD source=whitelist "
            f"teams={len(master_whitelist_rows)} path={d1_whitelist_path} cache_loaded=true"
        )
    else:
        if not dynamic_rebuild_allowed:
            raise RuntimeError(
                "Dynamic D1 rebuild is disabled by config. "
                f"d1_source={d1_source!r} allow_dynamic_d1_rebuild={allow_dynamic_d1_rebuild}. "
                "Set refresh.d1_source='dynamic' or refresh.allow_dynamic_d1_rebuild=true."
            )
        if whitelist_required and allow_dynamic_d1_rebuild:
            logger.info(
                "D1_REBUILD override enabled while d1_source=whitelist "
                "(allow_dynamic_d1_rebuild=true)"
            )
        if diagnostic_mode:
            d1_rebuild_start = diagnostic_day
            d1_rebuild_end = diagnostic_day
        elif d1_rebuild_mode == "season_to_date":
            d1_rebuild_start = season_start_day
            d1_rebuild_end = today
        else:
            d1_rebuild_start = start_day
            d1_rebuild_end = today
        d1_rebuild_days = max((d1_rebuild_end - d1_rebuild_start).days + 1, 0)
        logger.info(
            f"D1_REBUILD plan mode={d1_rebuild_mode} range={d1_rebuild_start}..{d1_rebuild_end} "
            f"days={d1_rebuild_days} days_limit={d1_rebuild_days_limit}"
        )

        d1_team_ids = rebuild_d1_ids_from_scoreboard(
            client=client,
            logger=logger,
            start_date=d1_rebuild_start,
            end_date=d1_rebuild_end,
            mode=d1_rebuild_mode,
            days_limit=d1_rebuild_days_limit,
            max_boxscores_total=diagnostic_max_boxscores,
        )
        write_d1_cache(out_dir, d1_team_ids, d1_team_meta, logger)

    d1_ids_sample = sorted(d1_team_ids)[:10]
    if whitelist_required and not allow_dynamic_d1_rebuild:
        logger.info("D1_PRELOAD source=whitelist using_cached_ids_for_fetch=true")
    else:
        logger.info(f"D1_BUILD d1_ids_count={len(d1_team_ids)} sample={d1_ids_sample}")

    if diagnostic_mode:
        logger.info(
            "DIAGNOSTIC_SUMMARY "
            f"day={diagnostic_day} d1_ids_count={len(d1_team_ids)} sample={d1_ids_sample} "
            "boxscores_cap=5 wrote_d1_cache=true"
        )
        logger.info("DIAGNOSTIC_MODE complete; exiting before history/rating exports.")
        return

    sleep_s = float(cfg["refresh"]["backfill"].get("sleep_seconds_per_game", 0.15))
    players_rows_new: list[dict] = []
    if needs_season_backfill:
        logger.info(
            "SEASON_BACKFILL required "
            f"hist_exists={hist_exists} hist_min_day={hist_min_day} "
            f"season_start_day={season_start_day} range={season_start_day}..{today}"
        )
        games_backfill = fetch_games_range(
            client=client,
            season=season,
            logger=logger,
            start_day=season_start_day,
            end_day=today,
            sleep_s=sleep_s,
            d1_team_ids=d1_team_ids,
            d1_team_meta=d1_team_meta,
            label="SEASON_BACKFILL",
            debug_out_dir=out_dir,
        )
        games_hist_after_backfill = upsert_games_history(games_backfill, out_dir, logger)
        games_hist_after_backfill = validate_games_df(games_hist_after_backfill, logger)
        _, backfill_hist_max_day = games_date_bounds(games_hist_after_backfill)
        if backfill_hist_max_day is not None:
            start_day = (pd.Timestamp(backfill_hist_max_day) - pd.Timedelta(days=2)).date()
            start_day = max(start_day, season_start_day)
        else:
            start_day = season_start_day
        logger.info(
            f"SEASON_BACKFILL complete rows={len(games_hist_after_backfill)} "
            f"next_incremental_start_day={start_day}"
        )
    else:
        logger.info(
            "SEASON_BACKFILL not needed "
            f"hist_min_day={hist_min_day} season_start_day={season_start_day}"
        )

    games_new = fetch_games_range(
        client=client,
        season=season,
        logger=logger,
        start_day=start_day,
        end_day=today,
        sleep_s=sleep_s,
        d1_team_ids=d1_team_ids,
        d1_team_meta=d1_team_meta,
        label="INCREMENTAL",
        player_rows_sink=players_rows_new,
        debug_out_dir=out_dir,
    )
    logger.info(f"PLAYER_EXTRACT rows={len(players_rows_new)}")
    games_new.to_csv(os.path.join(out_dir, "games_long.csv"), index=False)

    games_all = upsert_games_history(games_new, out_dir, logger)
    games_all = validate_games_df(games_all, logger)
    players_new_df = pd.DataFrame(players_rows_new) if players_rows_new else pd.DataFrame()
    logger.info(
        f"PLAYER_NEW_DF rows={len(players_new_df)} "
        f"cols={list(players_new_df.columns)[:20]}"
    )
    players_hist_path = os.path.join(out_dir, "players_history.csv")
    players_history_df = upsert_players_history(players_new_df, out_dir, logger)

    _, games_max_date = games_date_bounds(games_all)
    _, players_max_date = games_date_bounds(players_history_df)
    player_backfill_state = load_player_backfill_state(out_dir)
    backfill_reasons: list[str] = []
    if force_player_backfill:
        backfill_reasons.append("forced")
    if (not os.path.exists(players_hist_path)) or (players_history_df is None) or players_history_df.empty:
        backfill_reasons.append("missing_or_empty_players_history")
    if player_backfill_state is None:
        backfill_reasons.append("missing_state")
    else:
        try:
            state_season = int(player_backfill_state.get("season"))
        except Exception:
            state_season = None
        if state_season != season:
            backfill_reasons.append("state_season_mismatch")
    if players_max_date is None:
        backfill_reasons.append("players_max_date_missing")
    if (
        games_max_date is not None
        and players_max_date is not None
        and players_max_date < games_max_date
    ):
        backfill_reasons.append("players_behind_games_history")

    if backfill_reasons:
        if games_max_date is None:
            logger.info(
                "PLAYER_BACKFILL_SKIP reason=no_games_max_date "
                f"reasons={','.join(backfill_reasons)}"
            )
        elif not src_base:
            logger.info(
                "PLAYER_BACKFILL_SKIP reason=ncaa_api_base_url_unset "
                f"through_date={games_max_date} reasons={','.join(backfill_reasons)}"
            )
        else:
            logger.info(
                "PLAYER_BACKFILL_START "
                f"season={season} start={season_start_day} end={games_max_date} "
                f"reason={','.join(backfill_reasons)}"
            )
            players_rows_backfill: list[dict] = []
            _ = fetch_games_range(
                client=client,
                season=season,
                logger=logger,
                start_day=season_start_day,
                end_day=games_max_date,
                sleep_s=sleep_s,
                d1_team_ids=d1_team_ids,
                d1_team_meta=d1_team_meta,
                label="PLAYER_BACKFILL",
                player_rows_sink=players_rows_backfill,
                debug_out_dir=out_dir,
            )
            players_backfill_df = (
                pd.DataFrame(players_rows_backfill)
                if players_rows_backfill
                else pd.DataFrame()
            )
            players_history_df = upsert_players_history(players_backfill_df, out_dir, logger)
            state_path = write_player_backfill_state(
                out_dir=out_dir,
                season=season,
                through_date=games_max_date,
                run_id=run_id,
            )
            logger.info(
                "PLAYER_BACKFILL_DONE "
                f"rows_new={len(players_backfill_df)} "
                f"players_history_rows={len(players_history_df)} "
                f"through_date={games_max_date} "
                f"path={os.path.abspath(state_path)}"
            )

    # Build/persist reusable team directory name map for backfill.
    if not src_base and len(team_directory) < 300:
        merge_team_directory_from_games_df(team_directory, games_all)
        logger.info(
            f"TEAM_DIR seeded from games history teams={len(team_directory)} source=games_history"
        )
    write_team_directory_cache(out_dir, team_directory, logger)
    team_name_map = _build_team_identity_map(
        out_dir=out_dir,
        team_meta=team_directory,
        games_df=games_all,
    )
    write_team_directory_names_cache(out_dir, team_name_map, logger)
    cached_team_name_map = load_team_directory_names_cache(out_dir, logger)
    if cached_team_name_map:
        team_name_map = cached_team_name_map
    team_name_id_map: dict[str, str] = {}
    for raw_name, raw_tid in team_name_map.items():
        tid = _safe_team_id(raw_tid)
        if not tid:
            continue
        normalized_key = _normalize_team_match_key(raw_name)
        if normalized_key and normalized_key not in team_name_id_map:
            team_name_id_map[normalized_key] = tid
    logger.info(f"TEAM_DIR counts teams={len(team_directory)} names={len(team_name_map)}")
    valid_team_ids = set(team_directory.keys())

    games_all = map_team_names_to_ids(games_all, team_name_id_map)

    # Backfill IDs for historical rows and persist updated history.
    games_all = backfill_ids_from_directory(
        games_all, team_name_map, out_dir, logger, valid_team_ids=valid_team_ids
    )
    missing_team_id = int((games_all["team_id"] == "").sum()) if "team_id" in games_all.columns else len(games_all)
    missing_opp_id = int((games_all["opponent_team_id"] == "").sum()) if "opponent_team_id" in games_all.columns else len(games_all)

    # If needed, enrich from historical boxscores and retry name-based backfill.
    if missing_team_id > 0 or missing_opp_id > 0:
        games_all = enrich_ids_from_history_boxscores(client, games_all, team_directory, logger)
        write_team_directory_cache(out_dir, team_directory, logger)
        team_name_map = _build_team_identity_map(
            out_dir=out_dir,
            team_meta=team_directory,
            games_df=games_all,
        )
        write_team_directory_names_cache(out_dir, team_name_map, logger)
        cached_team_name_map = load_team_directory_names_cache(out_dir, logger)
        if cached_team_name_map:
            team_name_map = cached_team_name_map
        team_name_id_map = {}
        for raw_name, raw_tid in team_name_map.items():
            tid = _safe_team_id(raw_tid)
            if not tid:
                continue
            normalized_key = _normalize_team_match_key(raw_name)
            if normalized_key and normalized_key not in team_name_id_map:
                team_name_id_map[normalized_key] = tid
        logger.info(f"TEAM_DIR counts teams={len(team_directory)} names={len(team_name_map)}")
        valid_team_ids = set(team_directory.keys())
        games_all = map_team_names_to_ids(games_all, team_name_id_map)
        games_all = backfill_ids_from_directory(
            games_all, team_name_map, out_dir, logger, valid_team_ids=valid_team_ids
        )

    games_all = filter_strict_d1_games(games_all, logger, label="GAMES_HISTORY_WRITE")
    games_all.to_csv(hist_path, index=False)
    logger.info("ID_BACKFILL wrote updated games_history.csv")

    # Persist D1 IDs after full-history ID backfill.
    merge_team_directory_from_games_df(d1_team_meta, games_all)
    if whitelist_required:
        d1_team_ids, d1_conf_by_id, master_stats = build_d1_ids_from_master_whitelist(
            master_rows=master_whitelist_rows,
            team_name_map=team_name_map,
            games_df=games_all,
            logger=logger,
        )
        if len(d1_team_ids) < 300:
            logger.warning(
                "D1_BUILD degraded mode "
                f"path={d1_whitelist_path} resolved_d1_ids={len(d1_team_ids)} expected>=300 "
                f"master_rows={master_stats.get('master_rows')} "
                f"ambiguous_rows={master_stats.get('ambiguous_rows')} "
                f"unresolved_rows={master_stats.get('unresolved_rows')} "
                f"fallback_used={master_stats.get('fallback_used')} "
                f"match_rate={master_stats.get('match_rate')}"
            )
        whitelist_resolved_ids = set(d1_team_ids)
        d1_team_ids = whitelist_resolved_ids
        logger.info(
            "D1_BUILD source=whitelist "
            f"master_rows={master_stats.get('master_rows')} "
            f"matched_rows={master_stats.get('matched_rows')} "
            f"ambiguous_rows={master_stats.get('ambiguous_rows')} "
            f"unresolved_rows={master_stats.get('unresolved_rows')} "
            f"fallback_used={master_stats.get('fallback_used')} "
            f"match_rate={master_stats.get('match_rate')} "
            f"d1_ids_count={len(d1_team_ids)}"
        )
        id_to_conf = {
            _safe_team_id(tid): str(conf).strip()
            for tid, conf in d1_conf_by_id.items()
            if _safe_team_id(tid) and str(conf).strip()
        }
        conf_map = dict(id_to_conf)
        missing_conf_ids = sorted([tid for tid in d1_team_ids if tid not in id_to_conf])
        if missing_conf_ids:
            raise RuntimeError(
                "D1 whitelist data bug: missing conference mapping for resolved D1 IDs. "
                f"missing_count={len(missing_conf_ids)} sample={missing_conf_ids[:15]}"
            )
        if "team_conf" not in games_all.columns:
            games_all["team_conf"] = ""
        if "opp_conf" not in games_all.columns:
            games_all["opp_conf"] = ""
        if "team_id" in games_all.columns:
            team_ids = games_all["team_id"].apply(_safe_team_id)
            team_mask = team_ids.isin(d1_team_ids)
            games_all.loc[team_mask, "team_conf"] = team_ids[team_mask].map(id_to_conf)
        if "opponent_team_id" in games_all.columns:
            opp_ids = games_all["opponent_team_id"].apply(_safe_team_id)
            opp_mask = opp_ids.isin(d1_team_ids)
            games_all.loc[opp_mask, "opp_conf"] = opp_ids[opp_mask].map(id_to_conf)
    else:
        d1_team_ids = {tid for tid in (canonical_team_id(x) for x in d1_team_ids) if tid}
    write_d1_cache(out_dir, d1_team_ids, d1_team_meta, logger)
    d1_ids_sample = sorted(d1_team_ids)[:10]
    d1_final_source = "whitelist" if whitelist_required else "dynamic"
    logger.info(
        f"D1_FINAL source={d1_final_source} ids_count={len(d1_team_ids)} sample={d1_ids_sample}"
    )

    hist_min_after, hist_max_after = games_date_bounds(games_all)
    unique_games = int(games_all["game_id"].nunique()) if "game_id" in games_all.columns else 0
    unique_team_names = set()
    for c in ["team", "opponent"]:
        if c not in games_all.columns:
            continue
        unique_team_names |= {str(x).strip() for x in games_all[c].dropna().astype(str) if str(x).strip()}
    unique_teams = len(unique_team_names)
    d1_ids_count = len(d1_team_ids)
    logger.info(
        "BACKFILL_SUMMARY "
        f"games_history_min_date={hist_min_after} games_history_max_date={hist_max_after} "
        f"unique_games={unique_games} unique_teams={unique_teams} d1_ids_count={d1_ids_count}"
    )
    if d1_ids_count < 300 or d1_ids_count > 400:
        raise RuntimeError(
            f"D1_GUARD invalid d1_ids_count={d1_ids_count}; expected between 300 and 400."
        )

    missing_team_id = int((games_all["team_id"] == "").sum()) if "team_id" in games_all.columns else len(games_all)
    missing_opp_id = int((games_all["opponent_team_id"] == "").sum()) if "opponent_team_id" in games_all.columns else len(games_all)
    unresolved_names = pd.concat(
        [
            games_all.loc[games_all["team_id"] == "", "team"] if "team" in games_all.columns else pd.Series(dtype=str),
            games_all.loc[games_all["opponent_team_id"] == "", "opponent"] if "opponent" in games_all.columns else pd.Series(dtype=str),
        ],
        ignore_index=True,
    )
    unresolved_top20 = (
        unresolved_names.dropna().astype(str).value_counts().head(20).to_dict()
        if not unresolved_names.empty
        else {}
    )
    logger.info(f"ID_BACKFILL unresolved_top20={unresolved_top20}")

    missing_rate = max(missing_team_id, missing_opp_id) / max(len(games_all), 1)
    if missing_team_id > 0 and missing_rate >= 0.05:
        sample = games_all[
            (games_all["team_id"] == "") | (games_all["opponent_team_id"] == "")
        ][["team", "opponent", "team_id", "opponent_team_id"]].head(20).to_dict(orient="records")
        logger.warning(
            "Missing team IDs before D1 filter. "
            f"missing_team_id={missing_team_id} missing_opponent_team_id={missing_opp_id} "
            f"missing_rate={missing_rate:.4f} sample={sample}"
        )
    if missing_team_id > 0 or missing_opp_id > 0:
        logger.warning(
            f"Missing IDs remain but below threshold. missing_team_id={missing_team_id} "
            f"missing_opponent_team_id={missing_opp_id} missing_rate={missing_rate:.4f}"
        )

    team_id_series = games_all["team_id"].apply(_safe_team_id)
    opp_id_series = games_all["opponent_team_id"].apply(_safe_team_id)
    games_all["is_d1_team"] = team_id_series.isin(d1_team_ids)
    games_all["is_d1_opponent"] = opp_id_series.isin(d1_team_ids)

    flag_counts_team = games_all["is_d1_team"].value_counts(dropna=False).to_dict()
    flag_counts_opp = games_all["is_d1_opponent"].value_counts(dropna=False).to_dict()
    rows_before_filter = len(games_all)
    logger.info(f"D1_FILTER_DEBUG is_d1_team value_counts={flag_counts_team}")
    logger.info(f"D1_FILTER_DEBUG is_d1_opponent value_counts={flag_counts_opp}")

    games_for_ratings = games_all[
        team_id_series.isin(d1_team_ids) & opp_id_series.isin(d1_team_ids)
    ].copy()
    unique_teams_after = 0 if games_for_ratings.empty else int(games_for_ratings["team"].nunique())
    logger.info(
        f"D1_FILTER rows_before={rows_before_filter} rows_after={len(games_for_ratings)} "
        f"unique_teams_after={unique_teams_after}"
    )

    if games_for_ratings.empty:
        sample_cols = ["team_id", "opponent_team_id", "team", "opponent", "is_d1_team", "is_d1_opponent"]
        sample = games_all[sample_cols].head(20).rename(columns={
            "team_id": "teamId",
            "opponent_team_id": "opponentTeamId",
            "team": "team_name",
            "opponent": "opp_name",
        })
        raise RuntimeError(
            "D1 filter removed all rows. "
            f"len(D1_TEAM_IDS)={len(d1_team_ids)} "
            f"first10={sorted(d1_team_ids)[:10]} "
            f"is_d1_team_counts={flag_counts_team} "
            f"is_d1_opponent_counts={flag_counts_opp} "
            f"sample_rows={sample.to_dict(orient='records')}"
        )

    games_for_ratings.to_csv(os.path.join(out_dir, "games_used_for_ratings.csv"), index=False)
    logger.info("Wrote games_used_for_ratings.csv")
    logger.info(
        f"RUN_STEP run_id={run_id} stage=GAMES_USED_WRITTEN rows={len(games_for_ratings)} "
        f"path={os.path.abspath(os.path.join(out_dir, 'games_used_for_ratings.csv'))}"
    )

    ratings_df = compute_ratings(games_for_ratings, cfg, logger)
    if len(ratings_df) != len(d1_team_ids):
        logger.warning(
            "D1_GUARD ratings row mismatch. "
            f"ratings_rows={len(ratings_df)} d1_ids_count={len(d1_team_ids)} "
            f"ratings_missing={max(len(d1_team_ids) - len(ratings_df), 0)}"
        )
    if verify_d1_mode:
        logger.info(
            "VERIFY_D1 "
            f"d1_ids_count={len(d1_team_ids)} "
            f"ratings_rows={len(ratings_df)} "
            f"unique_teams_after={unique_teams_after} "
            f"missing_count={master_stats.get('unresolved_rows', 0)} "
            f"ambiguous_count={master_stats.get('ambiguous_rows', 0)}"
        )
        logger.info("VERIFY_D1 complete; exiting before full exports.")
        return
    logger.info(
        f"CHECK ratings rows={len(ratings_df)} cols={list(ratings_df.columns) if hasattr(ratings_df,'columns') else 'N/A'}"
    )
    _, team_adjustments_df = build_player_adjustments(project_root, out_dir, logger)
    ratings_df = apply_player_adjustments_to_ratings(
        ratings_df,
        team_adjustments_df,
        logger,
    )
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    try:
        hca_points_per_100 = float(model_cfg.get("hca_points_per_100", 3.0))
    except Exception:
        hca_points_per_100 = 3.0
    predictions_path = os.path.join(out_dir, "game_predictions.csv")
    predictions_log_path = os.path.join(out_dir, "predictions_log.csv")
    try:
        pred_df = build_game_predictions(
            ratings_df,
            season=season,
            logger=logger,
            base_url=src_base,
            hca_points_per_100=hca_points_per_100,
        )
    except Exception as ex:
        logger.warning(
            "PREDICTIONS build_failed "
            f"error={type(ex).__name__}: {ex}"
        )
        pred_df = _empty_predictions_df()
    pred_df.to_csv(predictions_path, index=False)
    logger.info(
        f"PREDICTIONS wrote rows={len(pred_df)} path={os.path.abspath(predictions_path)}"
    )
    _ = _ensure_predictions_log_schema(predictions_log_path, logger)
    pred_log_df = pred_df.copy()
    pred_log_df["actual_result"] = None
    pred_log_df["actual_margin"] = None
    pred_log_df = pred_log_df.reindex(columns=PREDICTION_LOG_COLUMNS)
    pred_log_df.to_csv(
        predictions_log_path,
        mode="a",
        header=not os.path.exists(predictions_log_path),
        index=False,
    )
    logger.info(
        f"PREDICTIONS log_appended rows={len(pred_log_df)} path={os.path.abspath(predictions_log_path)}"
    )
    sos = compute_sos(games_for_ratings, ratings_df)
    export_format = cfg.get("refresh", {}).get("export_format", "xlsx")
    cleanup_outputs = to_bool_flag(cfg.get("refresh", {}).get("cleanup_outputs", False))
    write_outputs(
        ratings_df,
        sos,
        out_dir,
        logger,
        export_format=export_format,
        cleanup_csv_outputs=cleanup_outputs,
        conf_map=conf_map,
    )
    logger.info(
        f"PLAYER_PIPELINE before_compute players_history_df_rows={len(players_history_df)}"
    )
    player_rankings_df = compute_player_rankings(
        players_history=players_history_df,
        games_for_ratings=games_for_ratings,
        ratings_df=ratings_df,
        conf_map=conf_map,
        cfg=cfg,
        logger=logger,
    )
    write_player_rankings_outputs(
        player_rankings_df,
        out_dir,
        logger,
        export_format=export_format,
    )
    logger.info(f"RUN_STEP run_id={run_id} stage=EXPORT_DONE format={export_format}")
    historical_backfill_end = today - timedelta(days=1)
    backfill_rows_generated = 0
    if season_start_day <= historical_backfill_end:
        try:
            backfill_rows_generated = backfill_historical_predictions(
                start_date=season_start_day,
                end_date=historical_backfill_end,
                logger=logger,
            )
        except Exception as ex:
            logger.warning(
                "PREDICTIONS backfill_run_failed "
                f"start_date={season_start_day} end_date={historical_backfill_end} "
                f"error={type(ex).__name__}: {ex}"
            )
    if backfill_rows_generated == 0:
        update_prediction_results(logger)
        log_prediction_metrics(logger)

    manifest_paths = [
        os.path.join(out_dir, "game_predictions.csv"),
        os.path.join(out_dir, "player_adjustments.csv"),
        os.path.join(out_dir, "teams_power_full.xlsx"),
        os.path.join(out_dir, "teams_power_top25.xlsx"),
        os.path.join(out_dir, "teams_power_top100.xlsx"),
        os.path.join(out_dir, "games_history.csv"),
        os.path.join(out_dir, "games_used_for_ratings.csv"),
        os.path.join(out_dir, "d1_team_ids.json"),
    ]
    manifest_path = write_run_manifest(
        run_id=run_id,
        config_hash=config_hash,
        games_used_df=games_for_ratings,
        ratings_df=ratings_df,
        artifact_paths=manifest_paths,
        out_dir=out_dir,
    )
    logger.info(f"RUN_MANIFEST run_id={run_id} path={os.path.abspath(manifest_path)}")

    latest_run_id_path = os.path.join(out_dir, "latest_run_id.txt")
    with open(latest_run_id_path, "w", encoding="utf-8") as f:
        f.write(f"{run_id}\n")
    logger.info(
        f"RUN_LATEST_ID run_id={run_id} path={os.path.abspath(latest_run_id_path)}"
    )
    logger.info("Refresh done.")


def rebuild_players_only(cfg_path: str) -> int:
    cfg = load_config(cfg_path)
    project_root = _resolve_project_root_from_cfg(cfg_path)
    out_dir = os.path.join(project_root, "outputs")
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "run.log"))

    logger.info(
        f"PLAYER_REBUILD_START cfg={os.path.abspath(cfg_path)} "
        f"out_dir={os.path.abspath(out_dir)}"
    )

    players_history_path = os.path.join(out_dir, "players_history.csv")
    games_used_path = os.path.join(out_dir, "games_used_for_ratings.csv")
    ratings_xlsx_path = os.path.join(out_dir, "teams_power_full.xlsx")

    if not os.path.exists(players_history_path):
        logger.error(
            "PLAYER_REBUILD_FAIL reason=missing_or_empty_players_history "
            f"path={os.path.abspath(players_history_path)}"
        )
        return 2

    try:
        players_history_df = pd.read_csv(players_history_path)
    except Exception as ex:
        logger.error(
            "PLAYER_REBUILD_FAIL reason=missing_or_empty_players_history "
            f"path={os.path.abspath(players_history_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        return 2

    if players_history_df.empty:
        logger.error(
            "PLAYER_REBUILD_FAIL reason=missing_or_empty_players_history "
            f"path={os.path.abspath(players_history_path)}"
        )
        return 2

    try:
        games_for_ratings = pd.read_csv(games_used_path)
    except Exception as ex:
        logger.error(
            "PLAYER_REBUILD_FAIL reason=missing_or_invalid_games_used "
            f"path={os.path.abspath(games_used_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        return 2

    try:
        ratings_df = pd.read_excel(ratings_xlsx_path, sheet_name="data", engine="openpyxl")
    except Exception as ex:
        logger.error(
            "PLAYER_REBUILD_FAIL reason=missing_or_invalid_ratings_export "
            f"path={os.path.abspath(ratings_xlsx_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        return 2

    season = int(cfg.get("season", date.today().year))
    refresh_cfg = cfg.get("refresh", {}) if isinstance(cfg.get("refresh", {}), dict) else {}
    d1_whitelist_relpath = str(
        refresh_cfg.get("d1_whitelist_path", DEFAULT_D1_WHITELIST_RELPATH)
    ).strip() or DEFAULT_D1_WHITELIST_RELPATH
    if os.path.isabs(d1_whitelist_relpath):
        d1_whitelist_path = d1_whitelist_relpath
    else:
        d1_whitelist_path = os.path.join(project_root, d1_whitelist_relpath)

    try:
        master_whitelist_rows = load_d1_master_whitelist(
            d1_whitelist_path, logger, expected_season=season
        )
        if len(master_whitelist_rows) < 300:
            raise RuntimeError(
                f"whitelist_too_small rows={len(master_whitelist_rows)} path={d1_whitelist_path}"
            )
        team_name_map = load_team_directory_names_cache(out_dir, logger)
        if not team_name_map:
            team_dir = load_team_directory_cache(out_dir)
            if team_dir:
                team_name_map = _build_team_identity_map(
                    out_dir=out_dir,
                    team_meta=team_dir,
                    games_df=games_for_ratings,
                )
        if not team_name_map:
            raise RuntimeError("missing_team_directory_names_cache")
        _, d1_conf_by_id, master_stats = build_d1_ids_from_master_whitelist(
            master_rows=master_whitelist_rows,
            team_name_map=team_name_map,
            games_df=games_for_ratings,
            logger=logger,
        )
        conf_map = {
            _safe_team_id(tid): str(conf).strip()
            for tid, conf in d1_conf_by_id.items()
            if _safe_team_id(tid) and str(conf).strip()
        }
        logger.info(
            "PLAYER_REBUILD_CONF "
            f"master_rows={master_stats.get('master_rows')} "
            f"matched_rows={master_stats.get('matched_rows')} "
            f"conf_map_count={len(conf_map)}"
        )
    except Exception as ex:
        logger.error(
            "PLAYER_REBUILD_FAIL reason=conf_map_build_failed "
            f"error={type(ex).__name__}: {ex}"
        )
        return 2

    logger.info(
        "PLAYER_REBUILD_INPUT "
        f"players_history_rows={len(players_history_df)} "
        f"games_used_rows={len(games_for_ratings)} "
        f"ratings_rows={len(ratings_df)}"
    )
    player_rankings_df = compute_player_rankings(
        players_history=players_history_df,
        games_for_ratings=games_for_ratings,
        ratings_df=ratings_df,
        conf_map=conf_map,
        cfg=cfg,
        logger=logger,
    )
    if player_rankings_df is None or player_rankings_df.empty:
        logger.info("PLAYER_REBUILD_DONE status=empty reason=no_rows_after_filters")
        return 3

    export_format = str(refresh_cfg.get("export_format", "xlsx")).strip().lower()
    write_player_rankings_outputs(
        player_rankings_df,
        out_dir,
        logger,
        export_format=export_format,
    )

    full_xlsx_path = os.path.join(out_dir, "player_rankings_full.xlsx")
    top100_xlsx_path = os.path.join(out_dir, "player_rankings_top100.xlsx")

    def _mtime_utc_or_missing(path: str) -> str:
        if not os.path.exists(path):
            return "missing"
        ts = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        return ts.isoformat(timespec="seconds").replace("+00:00", "Z")

    logger.info(
        "PLAYER_REBUILD_DONE status=OK "
        f"rows={len(player_rankings_df)} "
        f"full_mtime={_mtime_utc_or_missing(full_xlsx_path)} "
        f"top100_mtime={_mtime_utc_or_missing(top100_xlsx_path)}"
    )
    return 0


def _xlsx_sheet_summary(path: str) -> dict:
    summary = {
        "exists": os.path.exists(path),
        "path": os.path.abspath(path),
        "sheet": "",
        "rows_total": 0,
        "rows_data": 0,
        "cols": 0,
        "headers": [],
        "formulas": 0,
        "has_vba": False,
        "size_kb": 0.0,
        "error": "",
    }
    if not summary["exists"]:
        return summary

    summary["size_kb"] = round(os.path.getsize(path) / 1024.0, 1)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_id_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            summary["has_vba"] = any(
                ("vba" in n.lower()) and n.lower().endswith(".bin") for n in names
            )
            wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
            sheet_el = wb_root.find("x:sheets/x:sheet", ns)
            if sheet_el is None:
                summary["error"] = "No sheets found in workbook.xml"
                return summary
            summary["sheet"] = str(sheet_el.attrib.get("name", ""))
            rid = str(sheet_el.attrib.get(rel_id_attr, "")).strip()
            if not rid:
                summary["error"] = "Missing sheet relationship ID"
                return summary

            rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            target = ""
            for rel in rels_root:
                if str(rel.attrib.get("Id", "")) == rid:
                    target = str(rel.attrib.get("Target", "")).strip()
                    break
            if not target:
                summary["error"] = f"Missing relationship target for {rid}"
                return summary
            if target.startswith("/"):
                target = target[1:]
            elif not target.startswith("xl/"):
                target = f"xl/{target}"

            shared_strings = []
            if "xl/sharedStrings.xml" in names:
                sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in sst_root.findall("x:si", ns):
                    txt = "".join((t.text or "") for t in si.findall(".//x:t", ns))
                    shared_strings.append(txt)

            ws_root = ET.fromstring(zf.read(target))
            rows = ws_root.findall(".//x:sheetData/x:row", ns)
            summary["rows_total"] = len(rows)
            summary["rows_data"] = max(0, len(rows) - 1)
            summary["cols"] = max((len(r.findall("x:c", ns)) for r in rows), default=0)
            summary["formulas"] = len(ws_root.findall(".//x:f", ns))

            if rows:
                headers = []
                for cell in rows[0].findall("x:c", ns):
                    ctype = str(cell.attrib.get("t", ""))
                    value = ""
                    if ctype == "inlineStr":
                        is_el = cell.find("x:is", ns)
                        if is_el is not None:
                            value = "".join((t.text or "") for t in is_el.findall(".//x:t", ns))
                    else:
                        v_el = cell.find("x:v", ns)
                        if v_el is not None and v_el.text is not None:
                            raw = str(v_el.text)
                            if ctype == "s":
                                try:
                                    idx = int(raw)
                                except Exception:
                                    idx = -1
                                if 0 <= idx < len(shared_strings):
                                    value = shared_strings[idx]
                                else:
                                    value = raw
                            else:
                                value = raw
                    headers.append(value)
                summary["headers"] = headers
    except Exception as ex:
        summary["error"] = f"{type(ex).__name__}: {ex}"
    return summary


def verify_outputs(cfg_path: str) -> int:
    cfg = load_config(cfg_path)
    project_root = _resolve_project_root_from_cfg(cfg_path)
    out_dir = os.path.join(project_root, "outputs")
    refresh_cfg = cfg.get("refresh", {})
    verify_cfg = cfg.get("verify", {}) if isinstance(cfg.get("verify", {}), dict) else {}
    failures = 0

    print(f"VERIFY start cfg={os.path.abspath(cfg_path)} outputs={os.path.abspath(out_dir)}")

    manifest_path = os.path.join(out_dir, "run_manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as ex:
            failures += 1
            print(
                f"VERIFY manifest status=FAIL path={os.path.abspath(manifest_path)} "
                f"error={type(ex).__name__}: {ex}"
            )
    if manifest:
        print(
            f"VERIFY manifest status=OK path={os.path.abspath(manifest_path)} "
            f"run_id={manifest.get('run_id')} timestamp_utc={manifest.get('timestamp_utc')}"
        )
    else:
        print(f"VERIFY manifest status=NONE path={os.path.abspath(manifest_path)}")

    d1_whitelist_relpath = str(
        refresh_cfg.get("d1_whitelist_path", DEFAULT_D1_WHITELIST_RELPATH)
    ).strip() or DEFAULT_D1_WHITELIST_RELPATH
    if os.path.isabs(d1_whitelist_relpath):
        d1_whitelist_path = d1_whitelist_relpath
    else:
        d1_whitelist_path = os.path.join(project_root, d1_whitelist_relpath)

    whitelist_rows = []
    whitelist_confs = set()
    whitelist_season = None
    try:
        with open(d1_whitelist_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            whitelist_season = payload.get("season")
            if isinstance(payload.get("teams"), list):
                source_rows = payload.get("teams", [])
            elif isinstance(payload.get("rows"), list):
                source_rows = payload.get("rows", [])
            else:
                source_rows = []
        elif isinstance(payload, list):
            source_rows = payload
        else:
            source_rows = []
        for row in source_rows:
            team_name = ""
            conf = ""
            if isinstance(row, dict):
                team_name = str(row.get("team") or row.get("name") or "").strip()
                conf = str(row.get("conference") or row.get("conf") or "").strip()
            elif isinstance(row, str) and "," in row:
                team_name, conf = [str(x).strip() for x in row.split(",", 1)]
            if team_name and conf:
                whitelist_rows.append({"team": team_name, "conference": conf})
                whitelist_confs.add(conf)
    except Exception as ex:
        failures += 1
        print(
            f"VERIFY whitelist status=FAIL path={os.path.abspath(d1_whitelist_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
    else:
        print(
            f"VERIFY whitelist status=OK path={os.path.abspath(d1_whitelist_path)} "
            f"season={whitelist_season} teams={len(whitelist_rows)} conferences={len(whitelist_confs)}"
        )

    ids_path = os.path.join(out_dir, "d1_team_ids.json")
    d1_ids = []
    try:
        with open(ids_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            raise ValueError("d1_team_ids.json is not a list")
        d1_ids = [_safe_team_id(x) for x in arr if _safe_team_id(x)]
        numeric_all = all(bool(re.fullmatch(r"\d+", x)) for x in d1_ids)
    except Exception as ex:
        failures += 1
        print(
            f"VERIFY d1_cache status=FAIL path={os.path.abspath(ids_path)} "
            f"error={type(ex).__name__}: {ex}"
        )
        d1_id_set = set()
    else:
        d1_id_set = set(d1_ids)
        print(
            f"VERIFY d1_cache status=OK path={os.path.abspath(ids_path)} "
            f"ids_count={len(d1_ids)} numeric_ids={numeric_all}"
        )

    games_used_path = os.path.join(out_dir, "games_used_for_ratings.csv")
    try:
        games_used = pd.read_csv(games_used_path, dtype=str)
        rows = len(games_used)
        team_ids = games_used.get("team_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        opp_ids = games_used.get("opponent_team_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        missing_team = int((team_ids == "").sum())
        missing_opp = int((opp_ids == "").sum())
        unique_team_ids = int(team_ids[team_ids != ""].nunique())
        unique_opp_ids = int(opp_ids[opp_ids != ""].nunique())
        outside = int((~team_ids[team_ids != ""].isin(d1_id_set)).sum())
        outside += int((~opp_ids[opp_ids != ""].isin(d1_id_set)).sum())
        print(
            f"VERIFY games_used status=OK path={os.path.abspath(games_used_path)} "
            f"rows={rows} unique_team_ids={unique_team_ids} unique_opp_ids={unique_opp_ids} "
            f"missing_team_id={missing_team} missing_opp_id={missing_opp} ids_outside_d1={outside}"
        )
    except Exception as ex:
        failures += 1
        print(
            f"VERIFY games_used status=FAIL path={os.path.abspath(games_used_path)} "
            f"error={type(ex).__name__}: {ex}"
        )

    rating_required_headers = {
        "Conference",
        "Conf_Rank",
        "Conf_Avg_AdjEM",
        "Conf_Tier",
        "impact_total",
        "AdjEM_adj",
    }
    rating_expected_min_cols = 15

    full_xlsx = os.path.join(out_dir, "teams_power_full.xlsx")
    full_summary = _xlsx_sheet_summary(full_xlsx)
    if full_summary["exists"] and not full_summary["error"]:
        print(
            f"VERIFY ratings status=OK source={full_summary['path']} "
            f"rows={full_summary['rows_data']} cols={full_summary['cols']} sheet={full_summary['sheet']}"
        )
        if full_summary["cols"] < rating_expected_min_cols:
            failures += 1
            print(
                f"VERIFY ratings schema=FAIL expected_min_cols={rating_expected_min_cols} "
                f"actual_cols={full_summary['cols']}"
            )
        if not rating_required_headers.issubset(set(full_summary.get("headers", []))):
            failures += 1
            print(
                "VERIFY ratings schema=FAIL missing_columns="
                f"{sorted(rating_required_headers - set(full_summary.get('headers', [])))}"
            )
    else:
        failures += 1
        print(
            f"VERIFY ratings status=FAIL source={full_summary['path']} "
            f"error={full_summary['error'] or 'missing file'}"
        )

    for fn in ["teams_power_full.xlsx", "teams_power_top25.xlsx", "teams_power_top100.xlsx"]:
        xlsx_path = os.path.join(out_dir, fn)
        s = _xlsx_sheet_summary(xlsx_path)
        mtime_iso = "n/a"
        if s["exists"]:
            mtime_iso = datetime.fromtimestamp(os.path.getmtime(xlsx_path)).isoformat(sep=" ", timespec="seconds")
        if s["exists"] and not s["error"]:
            print(
                f"VERIFY export status=OK file={s['path']} size_kb={s['size_kb']} mtime={mtime_iso} "
                f"sheet={s['sheet']} rows_total={s['rows_total']} rows_data={s['rows_data']} "
                f"cols={s['cols']} formulas={s['formulas']} has_vba={s['has_vba']}"
            )
            if s["cols"] < rating_expected_min_cols:
                failures += 1
                print(
                    f"VERIFY export schema=FAIL file={s['path']} "
                    f"expected_min_cols={rating_expected_min_cols} actual_cols={s['cols']}"
                )
            if not rating_required_headers.issubset(set(s.get("headers", []))):
                failures += 1
                print(
                    f"VERIFY export schema=FAIL file={s['path']} missing_columns="
                    f"{sorted(rating_required_headers - set(s.get('headers', [])))}"
                )
            else:
                try:
                    verify_df = pd.read_excel(xlsx_path, sheet_name="data", engine="openpyxl")
                    conf_series = verify_df["Conference"].fillna("").astype(str).str.strip()
                    conf_rank_series = verify_df["Conf_Rank"]
                    conf_tier_series = verify_df["Conf_Tier"].fillna("").astype(str).str.strip()
                    conf_populated = conf_series != ""
                    missing_conf_rank = int(conf_rank_series[conf_populated].isna().sum())
                    missing_conf_tier = int((conf_tier_series[conf_populated] == "").sum())
                    if missing_conf_rank > 0:
                        failures += 1
                        print(
                            f"VERIFY export schema=FAIL file={s['path']} "
                            f"missing_conf_rank_rows={missing_conf_rank}"
                        )
                    if missing_conf_tier > 0:
                        failures += 1
                        print(
                            f"VERIFY export schema=FAIL file={s['path']} "
                            f"missing_conf_tier_rows={missing_conf_tier}"
                        )
                except Exception as ex:
                    failures += 1
                    print(
                        f"VERIFY export schema=FAIL file={s['path']} "
                        f"conf_rank_check_error={type(ex).__name__}: {ex}"
                    )
        else:
            failures += 1
            print(
                f"VERIFY export status=FAIL file={s['path']} mtime={mtime_iso} "
                f"error={s['error'] or 'missing file'}"
            )

    for fn in ["d1_team_ids.json", "games_history.csv", "games_used_for_ratings.csv"]:
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            mtime_iso = datetime.fromtimestamp(os.path.getmtime(p)).isoformat(sep=" ", timespec="seconds")
            print(f"VERIFY mtime file={os.path.abspath(p)} mtime={mtime_iso}")
        else:
            failures += 1
            print(f"VERIFY mtime file={os.path.abspath(p)} missing=true")

    conf_path = os.path.join(out_dir, "conference_power.csv")
    if os.path.exists(conf_path):
        try:
            conf_df = pd.read_csv(conf_path)
            print(
                f"VERIFY conference_power status=OK rows={len(conf_df)} "
                f"cols={list(conf_df.columns)}"
            )
        except Exception as ex:
            print(
                f"VERIFY conference_power status=ERROR error={type(ex).__name__}: {ex}"
            )

    adjustments_path = os.path.join(out_dir, "player_adjustments.csv")
    if os.path.exists(adjustments_path):
        try:
            adjustments_df = pd.read_csv(adjustments_path)
            required_adjustment_headers = {
                "team",
                "player",
                "status",
                "impact",
                "status_weight",
                "impact_contribution",
                "impact_total",
            }
            missing_adjustment_headers = sorted(
                required_adjustment_headers - set(adjustments_df.columns)
            )
            if missing_adjustment_headers:
                failures += 1
                print(
                    "VERIFY player_adjustments schema=FAIL "
                    f"path={os.path.abspath(adjustments_path)} "
                    f"missing_columns={missing_adjustment_headers}"
                )
            else:
                print(
                    f"VERIFY player_adjustments status=OK path={os.path.abspath(adjustments_path)} "
                    f"rows={len(adjustments_df)} cols={list(adjustments_df.columns)}"
                )
        except Exception as ex:
            failures += 1
            print(
                f"VERIFY player_adjustments status=FAIL path={os.path.abspath(adjustments_path)} "
                f"error={type(ex).__name__}: {ex}"
            )
    else:
        failures += 1
        print(
            f"VERIFY player_adjustments status=FAIL path={os.path.abspath(adjustments_path)} "
            "error=missing file"
        )

    players_hist_path = os.path.join(out_dir, "players_history.csv")
    if os.path.exists(players_hist_path):
        try:
            players_hist_df = pd.read_csv(players_hist_path)
            print(
                f"VERIFY players_history status=OK path={os.path.abspath(players_hist_path)} "
                f"rows={len(players_hist_df)}"
            )
        except Exception as ex:
            print(
                f"VERIFY players_history status=ERROR path={os.path.abspath(players_hist_path)} "
                f"error={type(ex).__name__}: {ex}"
            )

    player_full_xlsx = os.path.join(out_dir, "player_rankings_full.xlsx")
    if os.path.exists(player_full_xlsx):
        s = _xlsx_sheet_summary(player_full_xlsx)
        if s["exists"] and not s["error"]:
            print(
                f"VERIFY player_rankings status=OK file={s['path']} "
                f"rows_data={s['rows_data']} cols={s['cols']}"
            )
        else:
            print(
                f"VERIFY player_rankings status=ERROR file={os.path.abspath(player_full_xlsx)} "
                f"error={s.get('error') or 'unknown'}"
            )

    max_age_raw = verify_cfg.get("max_output_age_hours", None)
    max_age_hours = None
    if max_age_raw is not None:
        try:
            max_age_hours = float(max_age_raw)
            if max_age_hours < 0:
                max_age_hours = None
        except Exception:
            max_age_hours = None

    if max_age_hours is not None:
        if not manifest:
            failures += 1
            print(
                f"VERIFY freshness status=FAIL reason=missing_manifest "
                f"max_output_age_hours={max_age_hours}"
            )
        else:
            ts_raw = str(manifest.get("timestamp_utc", "")).strip()
            if not ts_raw:
                failures += 1
                print(
                    f"VERIFY freshness status=FAIL reason=bad_timestamp "
                    f"max_output_age_hours={max_age_hours}"
                )
            else:
                ts_norm = ts_raw.replace("Z", "+00:00")
                try:
                    ts = datetime.fromisoformat(ts_norm)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    age_hours = (now_utc - ts).total_seconds() / 3600.0
                    if age_hours > max_age_hours:
                        failures += 1
                        print(
                            f"VERIFY freshness status=FAIL age_hours={age_hours:.2f} "
                            f"max_output_age_hours={max_age_hours}"
                        )
                    else:
                        print(
                            f"VERIFY freshness status=OK age_hours={age_hours:.2f} "
                            f"max_output_age_hours={max_age_hours}"
                        )
                except Exception as ex:
                    failures += 1
                    print(
                        f"VERIFY freshness status=FAIL error={type(ex).__name__}: {ex} "
                        f"timestamp_utc={ts_raw!r}"
                    )

    print(f"VERIFY done failures={failures}")
    return 1 if failures else 0


def run_daemon(cfg_path: str):
    cfg = load_config(cfg_path)
    project_root = _resolve_project_root_from_cfg(cfg_path)
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, "run.log"))
    reg = int(cfg["refresh"]["poll_seconds_regular"])
    gw = int(cfg["refresh"]["poll_seconds_game_window"])
    start = cfg["refresh"]["game_window_local"]["start"]
    end = cfg["refresh"]["game_window_local"]["end"]

    def in_window(now: datetime) -> bool:
        s_h, s_m = map(int, start.split(":"))
        e_h, e_m = map(int, end.split(":"))
        s = now.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
        e = now.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
        if e <= s:
            return now >= s or now <= e
        return s <= now <= e

    while True:
        now = datetime.now(tz=NY)
        try:
            run_once(cfg_path)
        except Exception as ex:
            logger.exception(f"DAEMON run_once failed: {type(ex).__name__}: {ex}")
            print(f"{datetime.now(tz=NY).isoformat()} | ERROR | {ex}", file=sys.stderr)
        time.sleep(max(60, gw if in_window(now) else reg))

if __name__ == "__main__":
    cfg_path = _default_config_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=["daemon", "verify"])
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rebuild-players", action="store_true")
    parser.add_argument("--backfill-players", action="store_true")
    args = parser.parse_args()

    if args.rebuild_players:
        sys.exit(rebuild_players_only(cfg_path))
    if args.verify or args.command == "verify":
        sys.exit(verify_outputs(cfg_path))
    if args.command == "daemon":
        run_daemon(cfg_path)
    else:
        run_once(cfg_path, force_player_backfill=args.backfill_players)
