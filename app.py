from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

TOP_RANKING_FILES = (
    "teams_power_top25.csv",
    "teams_power_top100.csv",
    "teams_power_top25.xlsx",
    "teams_power_top100.xlsx",
)
FULL_RANKING_FILES = (
    "teams_power_full.csv",
    "teams_power_full.xlsx",
)
GAME_PREDICTION_PATTERNS = (
    "*prediction*.csv",
    "*predictions*.csv",
    "*forecast*.csv",
    "*projected*.csv",
    "*win_prob*.csv",
    "*odds*.csv",
)
GAME_OUTPUT_HINTS = (
    "games_long.csv",
    "games_history.csv",
    "games_used_for_ratings.csv",
)


@dataclass(frozen=True)
class TableResult:
    title: str
    path: Path | None
    dataframe: pd.DataFrame | None
    message: str = ""


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def output_directories() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []

    subdirs = sorted(
        (path for path in OUTPUTS_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [OUTPUTS_DIR, *subdirs]


def named_candidates(filenames: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    matches: list[Path] = []

    for directory in output_directories():
        for filename in filenames:
            candidate = directory / filename
            resolved = candidate.resolve()
            if candidate.is_file() and resolved not in seen:
                matches.append(candidate)
                seen.add(resolved)

    return matches


def glob_candidates(patterns: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    matches: list[Path] = []

    for directory in output_directories():
        for pattern in patterns:
            for candidate in sorted(directory.glob(pattern)):
                resolved = candidate.resolve()
                if candidate.is_file() and resolved not in seen:
                    matches.append(candidate)
                    seen.add(resolved)

    return matches


@st.cache_data(show_spinner=False)
def load_table(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def resolve_table(title: str, candidates: Iterable[Path]) -> TableResult:
    errors: list[str] = []
    empty_files: list[str] = []

    for path in candidates:
        try:
            dataframe = load_table(str(path))
        except Exception as exc:
            errors.append(
                f"Could not read `{relative_path(path)}` ({type(exc).__name__}: {exc})."
            )
            continue

        if dataframe.empty:
            empty_files.append(relative_path(path))
            continue

        return TableResult(title=title, path=path, dataframe=dataframe)

    if empty_files:
        errors.append(f"Found only empty files: {', '.join(empty_files)}.")
    if not errors:
        errors.append("No matching output files were found.")

    return TableResult(title=title, path=None, dataframe=None, message=" ".join(errors))


def load_top_rankings() -> TableResult:
    return resolve_table("Top Rankings", named_candidates(TOP_RANKING_FILES))


def load_full_rankings() -> TableResult:
    return resolve_table("Full Rankings", named_candidates(FULL_RANKING_FILES))


def load_game_predictions() -> TableResult:
    result = resolve_table("Game Predictions", glob_candidates(GAME_PREDICTION_PATTERNS))
    if result.dataframe is not None:
        return result

    available_game_files = [relative_path(path) for path in named_candidates(GAME_OUTPUT_HINTS)]
    message = result.message
    if available_game_files:
        message = (
            f"{message} No dedicated game prediction export was found. "
            f"Available game-related outputs: {', '.join(available_game_files)}."
        )

    return TableResult(title=result.title, path=None, dataframe=None, message=message)


def render_table(result: TableResult) -> None:
    st.subheader(result.title)

    if result.dataframe is None:
        st.warning(result.message)
        return

    st.caption(f"Source: `{relative_path(result.path)}`")
    st.caption(
        f"Rows: {len(result.dataframe):,} | Columns: {len(result.dataframe.columns):,}"
    )
    st.dataframe(result.dataframe, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="NCAAB Rankings Dashboard", layout="wide")
    st.title("NCAAB Rankings Dashboard")
    st.caption(f"Scanning output artifacts under `{relative_path(OUTPUTS_DIR)}`")

    if not OUTPUTS_DIR.exists():
        st.error(f"Outputs folder not found at `{relative_path(OUTPUTS_DIR)}`.")
        return

    views = {
        "Top Rankings": load_top_rankings,
        "Full Rankings": load_full_rankings,
        "Game Predictions": load_game_predictions,
    }

    selected_view = st.sidebar.radio("View", tuple(views))
    render_table(views[selected_view]())


if __name__ == "__main__":
    main()
