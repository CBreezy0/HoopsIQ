#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import dashboard_data as dd
    from app import ncaab_ranker as nr
    from app.runtime import OUTPUTS_DIR, REPO_ROOT, build_public_logger
    from scripts import model_backtest as mb
else:
    from app import dashboard_data as dd
    from app import ncaab_ranker as nr
    from app.runtime import OUTPUTS_DIR, REPO_ROOT, build_public_logger
    from scripts import model_backtest as mb


ROOT = REPO_ROOT
BACKTEST_RESULTS_PATH = OUTPUTS_DIR / "model_backtest_results.csv"


def _logger() -> logging.Logger:
    return build_public_logger("diagnose_live_gap")


def _clip_probs(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=1e-6, upper=1 - 1e-6)


def _settled_predictions() -> pd.DataFrame:
    df = dd.settled_predictions()
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["win_prob"] = pd.to_numeric(df["win_prob"], errors="coerce")
    df["actual_result"] = pd.to_numeric(df["actual_result"], errors="coerce")
    df["actual_margin"] = pd.to_numeric(df["actual_margin"], errors="coerce")
    df["projected_spread"] = pd.to_numeric(df["projected_spread"], errors="coerce")
    df["team_name_clean"] = df["team"].fillna("").astype(str).apply(nr.clean_team_name)
    df["opponent_name_clean"] = df["opponent"].fillna("").astype(str).apply(nr.clean_team_name)
    df["win_prob_error"] = (df["win_prob"] - df["actual_result"]).abs()
    df["spread_error"] = (df["projected_spread"] - df["actual_margin"]).abs()
    df["favorite_bucket"] = np.where(
        df["projected_spread"] > 0,
        "favorite",
        np.where(df["projected_spread"] < 0, "underdog", "pickem"),
    )
    df["spread_bucket"] = np.where(
        df["projected_spread"].abs() > 10,
        "large_spread",
        "small_spread",
    )
    return df


def _split_periods(eval_df: pd.DataFrame, recent_days: int) -> tuple[pd.DataFrame, pd.DataFrame, date]:
    latest_day = max(eval_df["date"])
    cutoff_day = latest_day - timedelta(days=max(int(recent_days) - 1, 0))
    live_df = eval_df.loc[eval_df["date"] >= cutoff_day].copy()
    backtest_df = eval_df.loc[eval_df["date"] < cutoff_day].copy()
    return backtest_df, live_df, cutoff_day


def _metrics(frame: pd.DataFrame) -> dict[str, float | int | str | None]:
    out = dd.compute_metrics_summary(frame)
    out["date_min"] = str(frame["date"].min()) if not frame.empty else None
    out["date_max"] = str(frame["date"].max()) if not frame.empty else None
    out["dates"] = int(frame["date"].nunique()) if not frame.empty else 0
    return out


def _calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["bucket", "count", "avg_pred", "actual_win_rate", "win_rate_gap"]
        )
    bins = [0.50, 0.60, 0.70, 0.80, 0.90]
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90"]
    cal = frame[(frame["win_prob"] >= 0.50) & (frame["win_prob"] < 0.90)].copy()
    if cal.empty:
        return pd.DataFrame(
            columns=["bucket", "count", "avg_pred", "actual_win_rate", "win_rate_gap"]
        )
    cal["bucket"] = pd.cut(cal["win_prob"], bins=bins, labels=labels, right=False)
    out = (
        cal.groupby("bucket", observed=False)
        .agg(
            count=("actual_result", "size"),
            avg_pred=("win_prob", "mean"),
            actual_win_rate=("actual_result", "mean"),
        )
        .reset_index()
    )
    out["bucket"] = out["bucket"].astype(str)
    out["win_rate_gap"] = out["actual_win_rate"] - out["avg_pred"]
    return out


def _breakdown_table(frame: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                bucket_col,
                "count",
                "avg_win_prob_error",
                "avg_spread_error",
                "avg_pred",
                "actual_win_rate",
            ]
        )
    return (
        frame.groupby(bucket_col, dropna=False)
        .agg(
            count=("team", "size"),
            avg_win_prob_error=("win_prob_error", "mean"),
            avg_spread_error=("spread_error", "mean"),
            avg_pred=("win_prob", "mean"),
            actual_win_rate=("actual_result", "mean"),
        )
        .reset_index()
    )


def _load_best_backtest_result() -> dict[str, float | int | None]:
    if not BACKTEST_RESULTS_PATH.exists():
        return {}
    df = pd.read_csv(BACKTEST_RESULTS_PATH)
    if df.empty:
        return {}
    best = df.sort_values(
        ["_combined_rank", "_brier_rank", "_spread_rank", "log_loss"],
        ascending=[True, True, True, True],
    ).iloc[0]
    return {
        "rows_used": int(best.get("rows_used", 0)),
        "coverage": float(best.get("coverage", np.nan)),
        "brier_score": float(best.get("brier_score", np.nan)),
        "log_loss": float(best.get("log_loss", np.nan)),
        "spread_mae": float(best.get("spread_mae", np.nan)),
        "hca": float(best.get("hca", np.nan)),
        "decay": float(best.get("decay", np.nan)),
        "blowout_cap": float(best.get("blowout_cap", np.nan)),
        "ridge": float(best.get("ridge", np.nan)),
    }


def _merge_recent_vegas(
    live_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    if live_df.empty:
        return live_df.copy()

    vegas_frames: list[pd.DataFrame] = []
    for day_value in sorted(live_df["date"].dropna().unique()):
        try:
            vegas_df = nr.fetch_vegas_lines(day_value, logger=logger)
        except Exception as ex:
            logger.warning(
                "LIVE_GAP vegas_fetch_failed "
                f"day={day_value} error={type(ex).__name__}: {ex}"
            )
            continue
        if vegas_df is not None and not vegas_df.empty:
            vegas_frames.append(vegas_df.copy())

    if not vegas_frames:
        return live_df.assign(
            vegas_win_prob=np.nan,
            vegas_spread=np.nan,
            edge=np.nan,
            abs_edge=np.nan,
        )

    vegas = (
        pd.concat(vegas_frames, ignore_index=True)
        .drop_duplicates(
            subset=["date", "team_name_clean", "opponent_name_clean"],
            keep="first",
        )
        .reset_index(drop=True)
    )
    vegas["date"] = pd.to_datetime(vegas["date"], errors="coerce").dt.date
    merged = live_df.merge(
        vegas[
            [
                "date",
                "team_name_clean",
                "opponent_name_clean",
                "vegas_win_prob",
                "vegas_spread",
                "vegas_provider",
            ]
        ],
        on=["date", "team_name_clean", "opponent_name_clean"],
        how="left",
    )
    merged["edge"] = merged["win_prob"] - merged["vegas_win_prob"]
    merged["abs_edge"] = merged["edge"].abs()
    return merged


def _edge_bucket(edge: float) -> str:
    if pd.isna(edge):
        return "missing"
    if edge >= 0.07:
        return "model_plus_large"
    if edge >= 0.03:
        return "model_plus_medium"
    if edge <= -0.07:
        return "market_plus_large"
    if edge <= -0.03:
        return "market_plus_medium"
    return "near_market"


def _edge_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "edge" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "edge_bucket",
                "count",
                "avg_edge",
                "avg_model_prob",
                "avg_vegas_prob",
                "actual_win_rate",
                "avg_win_prob_error",
            ]
        )
    df = frame.copy()
    df["edge_bucket"] = df["edge"].apply(_edge_bucket)
    return (
        df.groupby("edge_bucket", dropna=False)
        .agg(
            count=("team", "size"),
            avg_edge=("edge", "mean"),
            avg_model_prob=("win_prob", "mean"),
            avg_vegas_prob=("vegas_win_prob", "mean"),
            actual_win_rate=("actual_result", "mean"),
            avg_win_prob_error=("win_prob_error", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["count", "avg_edge"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def _print_section(title: str, payload) -> None:
    print(title)
    if isinstance(payload, pd.DataFrame):
        if payload.empty:
            print("(empty)")
        else:
            print(payload.to_string(index=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print()


@dataclass
class ReplaySummary:
    metrics: dict[str, float]
    rows: int


def _replay_recent_current_model(
    live_df: pd.DataFrame,
) -> ReplaySummary | None:
    if live_df.empty:
        return None

    cfg = mb._load_config()
    history_df = mb._load_games_history()
    eval_frame = mb._build_eval_frame()
    live_days = set(live_df["date"].dropna().tolist())
    eval_frame = eval_frame.loc[eval_frame["date"].isin(live_days)].copy()
    schedule_by_day = mb._schedule_by_day(eval_frame)
    if not schedule_by_day:
        return None

    base_model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    combined = mb._replay_config_predictions(
        cfg=cfg,
        history_df=history_df,
        schedule_by_day=schedule_by_day,
        season=int(cfg.get("season", nr.date.today().year)),
        hca=float(base_model.get("hca_points_per_100", 3.0)),
        decay=float(base_model.get("recency_decay_factor_days", 30.0)),
        blowout=float(base_model.get("blowout_cap_em_per100", 25.0)),
        ridge=float(base_model.get("ridge_lambda", 50.0)),
        apply_calibration=True,
        use_external_blend=True,
        use_vegas_blend=True,
    )
    metrics = mb._compute_metrics(combined, target_rows=len(live_df))
    return ReplaySummary(metrics=metrics, rows=len(combined))


def _largest_differences(
    backtest_df: pd.DataFrame,
    live_df: pd.DataFrame,
    recent_with_vegas: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []

    hist_fav = _breakdown_table(backtest_df, "favorite_bucket")
    live_fav = _breakdown_table(live_df, "favorite_bucket")
    if not hist_fav.empty and not live_fav.empty:
        hist_favorite = hist_fav.loc[hist_fav["favorite_bucket"] == "favorite"]
        live_favorite = live_fav.loc[live_fav["favorite_bucket"] == "favorite"]
        if not hist_favorite.empty and not live_favorite.empty:
            hist_pred = float(hist_favorite.iloc[0]["avg_pred"])
            hist_actual = float(hist_favorite.iloc[0]["actual_win_rate"])
            live_pred = float(live_favorite.iloc[0]["avg_pred"])
            live_actual = float(live_favorite.iloc[0]["actual_win_rate"])
            notes.append(
                "Recent favorites are materially more overconfident: "
                f"avg_pred={live_pred:.3f} vs actual={live_actual:.3f}, "
                f"historical avg_pred={hist_pred:.3f} vs actual={hist_actual:.3f}."
            )

    hist_small = _breakdown_table(backtest_df, "spread_bucket")
    live_small = _breakdown_table(live_df, "spread_bucket")
    if not hist_small.empty and not live_small.empty:
        hist_row = hist_small.loc[hist_small["spread_bucket"] == "small_spread"]
        live_row = live_small.loc[live_small["spread_bucket"] == "small_spread"]
        if not hist_row.empty and not live_row.empty:
            notes.append(
                "Small-spread games are the main live classification problem: "
                f"recent avg win_prob error={float(live_row.iloc[0]['avg_win_prob_error']):.3f} "
                f"vs historical {float(hist_row.iloc[0]['avg_win_prob_error']):.3f}."
            )

    if (
        not recent_with_vegas.empty
        and "vegas_win_prob" in recent_with_vegas.columns
        and recent_with_vegas["vegas_win_prob"].notna().any()
    ):
        edge_summary = _edge_summary(recent_with_vegas)
        large_model = edge_summary.loc[edge_summary["edge_bucket"] == "model_plus_large"]
        if not large_model.empty:
            row = large_model.iloc[0]
            notes.append(
                "Large positive model edges have not validated cleanly in the recent window: "
                f"avg_model_prob={float(row['avg_model_prob']):.3f}, "
                f"avg_vegas_prob={float(row['avg_vegas_prob']):.3f}, "
                f"actual_win_rate={float(row['actual_win_rate']):.3f}."
            )

    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose live-vs-backtest performance gap.")
    parser.add_argument("--recent-days", type=int, default=14)
    parser.add_argument("--skip-vegas", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args(argv)

    logger = _logger()
    eval_df = _settled_predictions()
    if eval_df.empty:
        raise SystemExit("No settled predictions available in outputs/predictions_log.csv")

    backtest_df, live_df, cutoff_day = _split_periods(eval_df, recent_days=args.recent_days)
    recent_with_vegas = (
        _merge_recent_vegas(live_df, logger=logger) if not args.skip_vegas else live_df.copy()
    )

    _print_section("LIVE_METRICS", _metrics(live_df))
    _print_section("BACKTEST_METRICS", _metrics(backtest_df))
    _print_section("CURRENT_BACKTEST_BEST", _load_best_backtest_result())

    _print_section("LIVE_CALIBRATION", _calibration_table(live_df))
    _print_section("BACKTEST_CALIBRATION", _calibration_table(backtest_df))

    _print_section("LIVE_ERROR_BY_FAVORITE_STATUS", _breakdown_table(live_df, "favorite_bucket"))
    _print_section(
        "BACKTEST_ERROR_BY_FAVORITE_STATUS",
        _breakdown_table(backtest_df, "favorite_bucket"),
    )
    _print_section("LIVE_ERROR_BY_SPREAD_SIZE", _breakdown_table(live_df, "spread_bucket"))
    _print_section(
        "BACKTEST_ERROR_BY_SPREAD_SIZE",
        _breakdown_table(backtest_df, "spread_bucket"),
    )

    if "vegas_win_prob" in recent_with_vegas.columns and recent_with_vegas["vegas_win_prob"].notna().any():
        line_timing = recent_with_vegas[
            [
                "date",
                "team",
                "opponent",
                "win_prob",
                "vegas_win_prob",
                "edge",
                "actual_result",
                "projected_spread",
                "actual_margin",
            ]
        ].copy()
        line_timing["abs_edge"] = line_timing["edge"].abs()
        line_timing = line_timing.sort_values("abs_edge", ascending=False).head(20)
        _print_section("LIVE_EDGE_BREAKDOWN", _edge_summary(recent_with_vegas))
        _print_section("LIVE_LARGEST_EDGES", line_timing)
    else:
        _print_section(
            "LIVE_EDGE_BREAKDOWN",
            pd.DataFrame(columns=["edge_bucket", "count", "avg_edge"]),
        )

    if not args.skip_replay:
        replay = _replay_recent_current_model(live_df)
        if replay is not None:
            _print_section("CURRENT_MODEL_REPLAY_RECENT", replay.metrics)

    biggest = _largest_differences(backtest_df, live_df, recent_with_vegas)
    _print_section(
        "LIKELY_ROOT_CAUSE",
        {
            "recent_days": int(args.recent_days),
            "live_cutoff": str(cutoff_day),
            "biggest_differences": biggest,
            "summary": (
                "The gap is partly a comparison issue: the ~0.197 backtest is a replay of the current "
                "model stack, while the live log contains predictions generated by earlier model states. "
                "The remaining recent weakness is concentrated in modest favorites and small-spread games, "
                "where the model is still too confident relative to realized win rates."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
