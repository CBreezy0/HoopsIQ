#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import ncaab_ranker as nr
    from app.runtime import DEFAULT_CONFIG_PATH, OUTPUTS_DIR, REPO_ROOT, build_public_logger
else:
    from app import ncaab_ranker as nr
    from app.runtime import DEFAULT_CONFIG_PATH, OUTPUTS_DIR, REPO_ROOT, build_public_logger


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")


ROOT = REPO_ROOT
CONFIG_PATH = DEFAULT_CONFIG_PATH
PREDICTIONS_LOG_PATH = OUTPUTS_DIR / "predictions_log.csv"
GAMES_HISTORY_PATH = OUTPUTS_DIR / "games_history.csv"
RESULTS_PATH = OUTPUTS_DIR / "model_backtest_results.csv"

DEFAULT_HCA_VALUES = (2.5, 3.0, 3.5, 4.0)
DEFAULT_DECAY_VALUES = (20.0, 30.0, 40.0)
DEFAULT_BLOWOUT_VALUES = (20.0, 25.0, 30.0)
DEFAULT_RIDGE_VALUES = (20.0, 50.0, 100.0)


def _quiet_logger() -> logging.Logger:
    logger = build_public_logger("model_backtest")
    logger.setLevel(logging.CRITICAL)
    return logger


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _stable_eval_predictions() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS_LOG_PATH)
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.date
    pred["actual_result"] = pd.to_numeric(pred["actual_result"], errors="coerce")
    pred["actual_margin"] = pd.to_numeric(pred["actual_margin"], errors="coerce")
    pred["win_prob"] = pd.to_numeric(pred["win_prob"], errors="coerce")
    pred["projected_spread"] = pd.to_numeric(pred["projected_spread"], errors="coerce")
    pred = pred.dropna(subset=["date", "team", "opponent"])
    pred = pred.drop_duplicates(subset=["date", "team", "opponent"], keep="last")
    pred = pred[pred["actual_result"].notna() & pred["actual_margin"].notna()].copy()
    pred["date_str"] = pred["date"].astype(str)
    pred["team_name_clean"] = pred["team"].apply(nr.clean_team_name)
    pred["opponent_name_clean"] = pred["opponent"].apply(nr.clean_team_name)
    pred["win_prob_error"] = (pred["win_prob"] - pred["actual_result"]).abs()
    pred["spread_error"] = (pred["projected_spread"] - pred["actual_margin"]).abs()
    return pred


def _load_games_history() -> pd.DataFrame:
    history = pd.read_csv(GAMES_HISTORY_PATH)
    history = nr.validate_games_df(history, _quiet_logger())
    if "is_d1_team" in history.columns and "is_d1_opponent" in history.columns:
        history = history[
            history["is_d1_team"].fillna(False)
            & history["is_d1_opponent"].fillna(False)
        ].copy()
    history["_game_day"] = pd.to_datetime(history["game_date"], errors="coerce").dt.date
    history = history.dropna(subset=["_game_day", "team", "opponent"]).copy()
    if "team_id" not in history.columns:
        history["team_id"] = ""
    if "opponent_team_id" not in history.columns:
        history["opponent_team_id"] = ""
    history["team_id"] = history["team_id"].fillna("").apply(nr._safe_team_id)
    history["opponent_team_id"] = history["opponent_team_id"].fillna("").apply(nr._safe_team_id)
    history = history.drop_duplicates(
        subset=["_game_day", "team", "opponent"],
        keep="last",
    )
    return history


def _build_eval_frame() -> pd.DataFrame:
    pred = _stable_eval_predictions()
    history = _load_games_history()
    meta = history[
        [
            "_game_day",
            "team",
            "opponent",
            "location",
            "team_id",
            "opponent_team_id",
            "is_d1_team",
            "is_d1_opponent",
        ]
    ].rename(columns={"_game_day": "date"})
    eval_df = pred.merge(meta, how="left", on=["date", "team", "opponent"])
    eval_df["neutral_site"] = eval_df["location"].fillna("").eq("N")
    eval_df["team_id"] = eval_df["team_id"].fillna("").apply(nr._safe_team_id)
    eval_df["opponent_team_id"] = eval_df["opponent_team_id"].fillna("").apply(nr._safe_team_id)
    eval_df["is_d1_team"] = eval_df["is_d1_team"].fillna(False).astype(bool)
    eval_df["is_d1_opponent"] = eval_df["is_d1_opponent"].fillna(False).astype(bool)
    bad_opponent_mask = (
        eval_df["actual_margin"].abs() > 40
    ) & (~eval_df["is_d1_opponent"])
    eval_df = eval_df.loc[~bad_opponent_mask].copy()
    return eval_df


def _print_error_analysis(eval_df: pd.DataFrame) -> None:
    spread_cols = [
        "date",
        "team",
        "opponent",
        "projected_spread",
        "actual_margin",
        "spread_error",
        "win_prob",
        "actual_result",
    ]
    win_cols = [
        "date",
        "team",
        "opponent",
        "win_prob",
        "actual_result",
        "win_prob_error",
        "projected_spread",
        "actual_margin",
    ]
    print("TOP_20_SPREAD_ERRORS")
    print(eval_df.sort_values("spread_error", ascending=False)[spread_cols].head(20).to_string(index=False))
    print()
    print("TOP_20_WIN_PROB_ERRORS")
    print(eval_df.sort_values("win_prob_error", ascending=False)[win_cols].head(20).to_string(index=False))
    print()


def _print_calibration(eval_df: pd.DataFrame) -> None:
    bins = [0.50, 0.60, 0.70, 0.80, 0.90]
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90"]
    cal = eval_df[(eval_df["win_prob"] >= 0.50) & (eval_df["win_prob"] < 0.90)].copy()
    cal["bucket"] = pd.cut(cal["win_prob"], bins=bins, labels=labels, right=False)
    summary = cal.groupby("bucket", observed=False).agg(
        count=("actual_result", "size"),
        actual_win_rate=("actual_result", "mean"),
    ).reset_index()
    print("CALIBRATION")
    print(summary.to_string(index=False))
    print()


def _print_weaknesses(eval_df: pd.DataFrame) -> None:
    bucket_df = eval_df.copy()
    bucket_df["favorite_bucket"] = np.where(
        bucket_df["projected_spread"] > 0,
        "favorite",
        np.where(bucket_df["projected_spread"] < 0, "underdog", "pickem"),
    )
    bucket_df["spread_bucket"] = np.where(
        bucket_df["projected_spread"].abs() > 10,
        "large_spread",
        "small_spread",
    )

    location_summary = bucket_df.groupby("location", dropna=False).agg(
        count=("team", "size"),
        avg_win_prob_error=("win_prob_error", "mean"),
        avg_spread_error=("spread_error", "mean"),
    ).reset_index()
    favorite_summary = bucket_df.groupby("favorite_bucket").agg(
        count=("team", "size"),
        avg_win_prob_error=("win_prob_error", "mean"),
        avg_spread_error=("spread_error", "mean"),
    ).reset_index()
    spread_summary = bucket_df.groupby("spread_bucket").agg(
        count=("team", "size"),
        avg_win_prob_error=("win_prob_error", "mean"),
        avg_spread_error=("spread_error", "mean"),
    ).reset_index()

    print("AVERAGE_ERROR_BY_LOCATION")
    print(location_summary.to_string(index=False))
    print()
    print("AVERAGE_ERROR_BY_FAVORITE_STATUS")
    print(favorite_summary.to_string(index=False))
    print()
    print("AVERAGE_ERROR_BY_SPREAD_SIZE")
    print(spread_summary.to_string(index=False))
    print()


def _schedule_by_day(eval_df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    schedule_cols = [
        "date",
        "team_id",
        "opponent_team_id",
        "team",
        "opponent",
        "team_name_clean",
        "opponent_name_clean",
        "neutral_site",
        "actual_result",
        "actual_margin",
    ]
    out: dict[date, pd.DataFrame] = {}
    for day_value, day_df in eval_df.groupby("date", sort=True):
        schedule = day_df[schedule_cols].copy()
        schedule["date"] = schedule["date"].astype(str)
        schedule["neutral_site"] = schedule["neutral_site"].fillna(False).astype(bool)
        schedule["team_id"] = schedule["team_id"].fillna("").apply(nr._safe_team_id)
        schedule["opponent_team_id"] = schedule["opponent_team_id"].fillna("").apply(nr._safe_team_id)
        out[day_value] = schedule.reset_index(drop=True)
    return out


def _compute_metrics(pred_df: pd.DataFrame, target_rows: int) -> dict[str, float]:
    if pred_df.empty:
        return {
            "rows_used": 0,
            "coverage": 0.0,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "spread_mae": np.nan,
        }
    p = pred_df["win_prob"].clip(1e-6, 1 - 1e-6)
    y = pred_df["actual_result"]
    return {
        "rows_used": int(len(pred_df)),
        "coverage": float(len(pred_df) / max(target_rows, 1)),
        "brier_score": float(((p - y) ** 2).mean()),
        "log_loss": float((-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean()),
        "spread_mae": float((pred_df["projected_spread"] - pred_df["actual_margin"]).abs().mean()),
    }


def _replay_config_predictions(
    cfg: dict,
    history_df: pd.DataFrame,
    schedule_by_day: dict,
    season: int,
    hca: float,
    decay: float,
    blowout: float,
    ridge: float,
    apply_calibration: bool = True,
    use_external_blend: bool = True,
    use_vegas_blend: bool = True,
) -> pd.DataFrame:
    local_cfg = json.loads(json.dumps(cfg))
    local_cfg.setdefault("model", {})
    local_cfg["model"]["hca_points_per_100"] = float(hca)
    local_cfg["model"]["recency_decay_factor_days"] = float(decay)
    local_cfg["model"]["recency_half_life_days"] = float(decay)
    local_cfg["model"]["blowout_cap_em_per100"] = float(blowout)
    local_cfg["model"]["ridge_lambda"] = float(ridge)

    quiet = _quiet_logger()
    predicted_frames: list[pd.DataFrame] = []
    shared_probability_adjustment_model = None
    if use_vegas_blend and schedule_by_day:
        replay_cutoff = max(schedule_by_day) + timedelta(days=1)
        shared_probability_adjustment_model = nr.fit_prediction_probability_adjustment_model(
            logger=quiet,
            asof_date=replay_cutoff,
        )

    for day_value, day_schedule in schedule_by_day.items():
        games_subset = history_df.loc[history_df["_game_day"] < day_value].copy()
        if games_subset.empty:
            continue
        ratings_df = nr.compute_ratings(
            games_subset,
            local_cfg,
            quiet,
            asof=pd.Timestamp(day_value, tz=nr.NY),
        )
        if ratings_df is None or ratings_df.empty:
            continue
        pred_df = nr.build_game_predictions(
            ratings_df,
            season=season,
            logger=quiet,
            base_url=None,
            day=day_value,
            use_external_blend=use_external_blend,
            use_vegas_blend=use_vegas_blend,
            include_completed=True,
            schedule_df=day_schedule,
            hca_points_per_100=float(hca),
            apply_calibration=apply_calibration,
            probability_adjustment_model=shared_probability_adjustment_model,
        )
        if pred_df is None or pred_df.empty:
            continue
        merged = pred_df.merge(
            day_schedule[["date", "team", "opponent", "actual_result", "actual_margin"]],
            on=["date", "team", "opponent"],
            how="inner",
        )
        if merged.empty:
            continue
        predicted_frames.append(merged)

    return (
        pd.concat(predicted_frames, ignore_index=True)
        if predicted_frames
        else pd.DataFrame(columns=nr.PREDICTION_LOG_COLUMNS)
    )


def _evaluate_config(
    cfg: dict,
    history_df: pd.DataFrame,
    schedule_by_day: dict,
    season: int,
    hca: float,
    decay: float,
    blowout: float,
    ridge: float,
    apply_calibration: bool = True,
    use_external_blend: bool = True,
    use_vegas_blend: bool = True,
) -> dict[str, float]:
    target_rows = sum(len(day_df) for day_df in schedule_by_day.values())
    combined = _replay_config_predictions(
        cfg=cfg,
        history_df=history_df,
        schedule_by_day=schedule_by_day,
        season=season,
        hca=hca,
        decay=decay,
        blowout=blowout,
        ridge=ridge,
        apply_calibration=apply_calibration,
        use_external_blend=use_external_blend,
        use_vegas_blend=use_vegas_blend,
    )
    metrics = _compute_metrics(combined, target_rows=target_rows)
    metrics.update(
        {
            "hca": float(hca),
            "decay": float(decay),
            "blowout_cap": float(blowout),
            "ridge": float(ridge),
        }
    )
    return metrics


def _pick_best(results_df: pd.DataFrame) -> pd.Series:
    ranked = results_df.copy()
    ranked["_brier_rank"] = ranked["brier_score"].rank(method="dense", ascending=True)
    ranked["_spread_rank"] = ranked["spread_mae"].rank(method="dense", ascending=True)
    ranked["_combined_rank"] = ranked["_brier_rank"] + ranked["_spread_rank"]
    ranked = ranked.sort_values(
        ["_combined_rank", "_brier_rank", "_spread_rank", "log_loss", "coverage", "rows_used"],
        ascending=[True, True, True, True, False, False],
    ).reset_index(drop=True)
    return ranked.iloc[0]


def run_sweep(mode: str) -> pd.DataFrame:
    cfg = _load_config()
    season = int(cfg.get("season", nr.date.today().year))
    eval_df = _build_eval_frame()
    history_df = _load_games_history()
    schedule_by_day = _schedule_by_day(eval_df)
    if not schedule_by_day:
        raise RuntimeError("No settled predictions available for backtest")

    base_model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    current = {
        "hca": float(base_model.get("hca_points_per_100", 3.0)),
        "decay": float(base_model.get("recency_decay_factor_days", 30.0)),
        "blowout_cap": float(base_model.get("blowout_cap_em_per100", 25.0)),
        "ridge": float(base_model.get("ridge_lambda", 50.0)),
    }

    results: list[dict[str, float]] = []

    if mode == "grid":
        for hca in DEFAULT_HCA_VALUES:
            for decay in DEFAULT_DECAY_VALUES:
                for blowout in DEFAULT_BLOWOUT_VALUES:
                    for ridge in DEFAULT_RIDGE_VALUES:
                        metrics = _evaluate_config(
                            cfg=cfg,
                            history_df=history_df,
                            schedule_by_day=schedule_by_day,
                            season=season,
                            hca=hca,
                            decay=decay,
                            blowout=blowout,
                            ridge=ridge,
                        )
                        results.append(metrics)
                        print(
                            "BACKTEST "
                            f"hca={hca:.1f} decay={decay:.1f} blowout_cap={blowout:.1f} ridge={ridge:.1f} "
                            f"rows={metrics['rows_used']} brier={metrics['brier_score']:.4f} "
                            f"spread_mae={metrics['spread_mae']:.4f}"
                        )
    else:
        baseline = _evaluate_config(
            cfg=cfg,
            history_df=history_df,
            schedule_by_day=schedule_by_day,
            season=season,
            hca=current["hca"],
            decay=current["decay"],
            blowout=current["blowout_cap"],
            ridge=current["ridge"],
        )
        results.append(baseline)
        print(
            "BACKTEST baseline "
            f"hca={baseline['hca']:.1f} decay={baseline['decay']:.1f} "
            f"blowout_cap={baseline['blowout_cap']:.1f} ridge={baseline['ridge']:.1f} "
            f"rows={baseline['rows_used']} brier={baseline['brier_score']:.4f} "
            f"spread_mae={baseline['spread_mae']:.4f}"
        )

        sweep_order = [
            ("hca", DEFAULT_HCA_VALUES),
            ("decay", DEFAULT_DECAY_VALUES),
            ("blowout_cap", DEFAULT_BLOWOUT_VALUES),
            ("ridge", DEFAULT_RIDGE_VALUES),
        ]
        for field, values in sweep_order:
            field_results: list[dict[str, float]] = []
            for value in values:
                params = dict(current)
                params[field] = float(value)
                metrics = _evaluate_config(
                    cfg=cfg,
                    history_df=history_df,
                    schedule_by_day=schedule_by_day,
                    season=season,
                    hca=params["hca"],
                    decay=params["decay"],
                    blowout=params["blowout_cap"],
                    ridge=params["ridge"],
                )
                field_results.append(metrics)
                results.append(metrics)
                print(
                    "BACKTEST "
                    f"field={field} hca={metrics['hca']:.1f} decay={metrics['decay']:.1f} "
                    f"blowout_cap={metrics['blowout_cap']:.1f} ridge={metrics['ridge']:.1f} "
                    f"rows={metrics['rows_used']} brier={metrics['brier_score']:.4f} "
                    f"spread_mae={metrics['spread_mae']:.4f}"
                )
            best_field = _pick_best(pd.DataFrame(field_results))
            current = {
                "hca": float(best_field["hca"]),
                "decay": float(best_field["decay"]),
                "blowout_cap": float(best_field["blowout_cap"]),
                "ridge": float(best_field["ridge"]),
            }

        final_metrics = _evaluate_config(
            cfg=cfg,
            history_df=history_df,
            schedule_by_day=schedule_by_day,
            season=season,
            hca=current["hca"],
            decay=current["decay"],
            blowout=current["blowout_cap"],
            ridge=current["ridge"],
        )
        results.append(final_metrics)
        print(
            "BACKTEST final "
            f"hca={final_metrics['hca']:.1f} decay={final_metrics['decay']:.1f} "
            f"blowout_cap={final_metrics['blowout_cap']:.1f} ridge={final_metrics['ridge']:.1f} "
            f"rows={final_metrics['rows_used']} brier={final_metrics['brier_score']:.4f} "
            f"spread_mae={final_metrics['spread_mae']:.4f}"
        )

    results_df = pd.DataFrame(results).drop_duplicates(
        subset=["hca", "decay", "blowout_cap", "ridge"],
        keep="last",
    )
    ranked = results_df.copy()
    ranked["_brier_rank"] = ranked["brier_score"].rank(method="dense", ascending=True)
    ranked["_spread_rank"] = ranked["spread_mae"].rank(method="dense", ascending=True)
    ranked["_combined_rank"] = ranked["_brier_rank"] + ranked["_spread_rank"]
    results_df = ranked.sort_values(
        ["_combined_rank", "_brier_rank", "_spread_rank", "log_loss", "coverage", "rows_used"],
        ascending=[True, True, True, True, False, False],
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    return results_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sequential", "grid"],
        default="sequential",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write the best discovered parameters back to config/config.json",
    )
    args = parser.parse_args(argv)

    eval_df = _build_eval_frame()
    _print_error_analysis(eval_df)
    _print_calibration(eval_df)
    _print_weaknesses(eval_df)

    results_df = run_sweep(mode=args.mode)
    best = _pick_best(results_df)

    print("BACKTEST_RESULTS_TOP")
    print(results_df.head(20).to_string(index=False))
    print()
    print("BEST_MODEL")
    print(f"- hca = {best['hca']:.1f}")
    print(f"- decay = {best['decay']:.1f}")
    print(f"- blowout_cap = {best['blowout_cap']:.1f}")
    print(f"- ridge = {best['ridge']:.1f}")
    print(f"- rows_used = {int(best['rows_used'])}")
    print(f"- coverage = {best['coverage']:.4f}")
    print(f"- brier = {best['brier_score']:.4f}")
    print(f"- log_loss = {best['log_loss']:.4f}")
    print(f"- spread_mae = {best['spread_mae']:.4f}")

    if args.apply_best:
        cfg = _load_config()
        cfg.setdefault("model", {})
        cfg["model"]["hca_points_per_100"] = float(best["hca"])
        cfg["model"]["recency_decay_factor_days"] = float(best["decay"])
        cfg["model"]["recency_half_life_days"] = float(best["decay"])
        cfg["model"]["blowout_cap_em_per100"] = float(best["blowout_cap"])
        cfg["model"]["ridge_lambda"] = float(best["ridge"])
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=4)
            fh.write("\n")
        print()
        print(f"UPDATED_CONFIG {CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
