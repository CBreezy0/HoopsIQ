from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import ncaab_ranker as nr
    from app.runtime import OUTPUTS_DIR, load_public_config
else:
    from . import ncaab_ranker as nr
    from .runtime import OUTPUTS_DIR, load_public_config


GAME_PREDICTIONS_PATH = OUTPUTS_DIR / "game_predictions.csv"
PREDICTIONS_LOG_PATH = OUTPUTS_DIR / "predictions_log.csv"
LIVE_PREDICTIONS_PATH = OUTPUTS_DIR / "live_predictions_dashboard.csv"
TOP_EDGES_PATH = OUTPUTS_DIR / "top_edges.csv"
BETS_PATH = OUTPUTS_DIR / "bets.csv"
BETS_LOG_PATH = OUTPUTS_DIR / "bets_log.csv"
MODEL_METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
DAILY_METRICS_PATH = OUTPUTS_DIR / "daily_metrics.csv"
CALIBRATION_PATH = OUTPUTS_DIR / "calibration_summary.csv"
BETTING_METRICS_PATH = OUTPUTS_DIR / "betting_metrics.json"
BETTING_BUCKET_METRICS_PATH = OUTPUTS_DIR / "betting_bucket_metrics.csv"
LIVE_BETTING_PERFORMANCE_PATH = OUTPUTS_DIR / "live_betting_performance.csv"
BRACKET_PREDICTIONS_PATH = OUTPUTS_DIR / "bracket_predictions.csv"
MOST_LIKELY_BRACKET_PATH = OUTPUTS_DIR / "most_likely_bracket.csv"
POOL_BRACKET_PATH = OUTPUTS_DIR / "pool_bracket.csv"
BRACKET_MATCHUP_PROBS_PATH = OUTPUTS_DIR / "bracket_matchup_probs.csv"
_PUBLIC_CONFIG = load_public_config()
_BETTING_CFG = (
    _PUBLIC_CONFIG.get("betting", {})
    if isinstance(_PUBLIC_CONFIG.get("betting", {}), dict)
    else {}
)
MARKET_VIG_APPROX = float(_BETTING_CFG.get("market_vig_approx", 0.02))
LIVE_BANKROLL_START = float(_BETTING_CFG.get("live_bankroll_start", 1.0))


def _dashboard_logger(logger: logging.Logger | None = None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger("dashboard_data")


def _empty_live_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "team",
            "opponent",
            "win_prob",
            "projected_spread",
            "projected_score_team",
            "projected_score_opp",
            "vegas_win_prob",
            "fair_vegas_win_prob",
            "vegas_spread",
            "vegas_provider",
            "edge",
            "abs_edge",
            "spread_edge",
            "team_name_clean",
            "opponent_name_clean",
        ]
    )


def _empty_bets_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "team",
            "opponent",
            "edge",
            "win_prob",
            "vegas_prob",
            "bet_size",
            "confidence_bucket",
            "is_live",
        ]
    )


def _empty_bets_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "team",
            "opponent",
            "source_team",
            "source_opponent",
            "bet_on_team_side",
            "edge",
            "win_prob",
            "vegas_prob",
            "market_prob_raw",
            "bet_size",
            "confidence_bucket",
            "edge_bucket",
            "spread_bucket",
            "projected_spread",
            "vegas_provider",
            "is_live",
            "logged_at_utc",
            "actual_result",
            "actual_margin",
            "bet_won",
            "profit",
            "closing_vegas_prob",
            "closing_fair_prob",
            "closing_vegas_provider",
            "clv",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _clip_probs(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=1e-6, upper=1 - 1e-6)


def _raw_market_prob_to_fair(series: pd.Series | np.ndarray) -> pd.Series:
    probs = _clip_probs(pd.Series(series))
    fair = (probs - MARKET_VIG_APPROX) / (1.0 - 2.0 * MARKET_VIG_APPROX)
    return _clip_probs(fair)


def _fair_market_prob_to_priced(series: pd.Series | np.ndarray) -> pd.Series:
    probs = _clip_probs(pd.Series(series))
    priced = probs * (1.0 - 2.0 * MARKET_VIG_APPROX) + MARKET_VIG_APPROX
    return _clip_probs(priced)


def load_predictions_log() -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        return pd.DataFrame(columns=nr.PREDICTION_LOG_COLUMNS)

    df = pd.read_csv(PREDICTIONS_LOG_PATH)
    for col in nr.PREDICTION_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["win_prob"] = pd.to_numeric(df["win_prob"], errors="coerce")
    df["actual_result"] = pd.to_numeric(df["actual_result"], errors="coerce")
    df["actual_margin"] = pd.to_numeric(df["actual_margin"], errors="coerce")
    df["projected_spread"] = pd.to_numeric(df["projected_spread"], errors="coerce")
    for col in [
        "vegas_win_prob",
        "vegas_spread",
        "closing_vegas_win_prob",
        "closing_vegas_spread",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "team", "opponent"]).copy()
    df = df.drop_duplicates(subset=["date", "team", "opponent"], keep="last").reset_index(drop=True)
    return df


def load_bets_log() -> pd.DataFrame:
    if not BETS_LOG_PATH.exists():
        return _empty_bets_log_df()

    df = pd.read_csv(BETS_LOG_PATH)
    empty = _empty_bets_log_df()
    for col in empty.columns:
        if col not in df.columns:
            df[col] = empty[col].dtype.type() if col in {} else np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    for col in [
        "bet_on_team_side",
        "is_live",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in [
        "edge",
        "win_prob",
        "vegas_prob",
        "market_prob_raw",
        "bet_size",
        "projected_spread",
        "actual_result",
        "actual_margin",
        "bet_won",
        "profit",
        "closing_vegas_prob",
        "closing_fair_prob",
        "clv",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "team",
        "opponent",
        "source_team",
        "source_opponent",
        "confidence_bucket",
        "edge_bucket",
        "spread_bucket",
        "vegas_provider",
        "closing_vegas_provider",
        "logged_at_utc",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    df = df.reindex(columns=empty.columns)
    df = df.drop_duplicates(subset=["date", "team", "opponent"], keep="first").reset_index(drop=True)
    return df


def settled_predictions(pred_log_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_predictions_log() if pred_log_df is None else pred_log_df.copy()
    if df.empty:
        return df
    return df[
        df["win_prob"].notna()
        & df["actual_result"].notna()
        & df["projected_spread"].notna()
    ].copy()


def compute_metrics_summary(pred_log_df: pd.DataFrame | None = None) -> dict[str, float | int | None]:
    eval_df = settled_predictions(pred_log_df)
    metrics: dict[str, float | int | None] = {
        "rows_used": int(len(eval_df)),
        "brier_score": None,
        "log_loss": None,
        "spread_mae": None,
    }
    if eval_df.empty:
        return metrics

    probs = _clip_probs(eval_df["win_prob"])
    actuals = pd.to_numeric(eval_df["actual_result"], errors="coerce")
    metrics["brier_score"] = float(((probs - actuals) ** 2).mean())
    metrics["log_loss"] = float(
        (-(actuals * np.log(probs) + (1.0 - actuals) * np.log(1.0 - probs))).mean()
    )

    spread_df = eval_df[eval_df["actual_margin"].notna()].copy()
    if not spread_df.empty:
        metrics["spread_mae"] = float(
            (spread_df["projected_spread"] - spread_df["actual_margin"]).abs().mean()
        )

    return metrics


def compute_daily_metrics(pred_log_df: pd.DataFrame | None = None) -> pd.DataFrame:
    eval_df = settled_predictions(pred_log_df)
    if eval_df.empty:
        return pd.DataFrame(
            columns=["date", "rows_used", "brier_score", "log_loss", "spread_mae"]
        )

    rows: list[dict[str, object]] = []
    for day_value, day_df in eval_df.groupby("date", sort=True):
        probs = _clip_probs(day_df["win_prob"])
        actuals = pd.to_numeric(day_df["actual_result"], errors="coerce")
        spread_df = day_df[day_df["actual_margin"].notna()].copy()
        rows.append(
            {
                "date": str(day_value),
                "rows_used": int(len(day_df)),
                "brier_score": float(((probs - actuals) ** 2).mean()),
                "log_loss": float(
                    (-(actuals * np.log(probs) + (1.0 - actuals) * np.log(1.0 - probs))).mean()
                ),
                "spread_mae": (
                    float((spread_df["projected_spread"] - spread_df["actual_margin"]).abs().mean())
                    if not spread_df.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_calibration_summary(pred_log_df: pd.DataFrame | None = None) -> pd.DataFrame:
    eval_df = settled_predictions(pred_log_df)
    if eval_df.empty:
        return pd.DataFrame(
            columns=["bucket", "bucket_mid", "count", "avg_pred", "actual_win_rate"]
        )

    cal_df = eval_df.copy()
    cal_df["win_prob"] = _clip_probs(cal_df["win_prob"])
    bin_edges = np.linspace(0.0, 1.0, 11)
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]
    cal_df["bucket"] = pd.cut(
        cal_df["win_prob"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    summary = (
        cal_df.groupby("bucket", observed=False)
        .agg(
            count=("actual_result", "size"),
            avg_pred=("win_prob", "mean"),
            actual_win_rate=("actual_result", "mean"),
        )
        .reset_index()
    )
    summary["bucket"] = summary["bucket"].astype(str)
    summary["bucket_mid"] = [
        float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)
        for idx in range(len(labels))
    ]
    return summary


def build_live_predictions_dataframe(
    predictions_df: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
    fetch_market: bool = True,
) -> pd.DataFrame:
    logger = _dashboard_logger(logger)
    if predictions_df is None:
        predictions_df = _read_csv(GAME_PREDICTIONS_PATH)
    if predictions_df is None or predictions_df.empty:
        return _empty_live_predictions_df()

    live_df = predictions_df.copy()
    live_df["date"] = pd.to_datetime(live_df["date"], errors="coerce").dt.date.astype("string")
    live_df["win_prob"] = pd.to_numeric(live_df["win_prob"], errors="coerce")
    live_df["projected_spread"] = pd.to_numeric(live_df["projected_spread"], errors="coerce")
    live_df["projected_score_team"] = pd.to_numeric(
        live_df["projected_score_team"], errors="coerce"
    )
    live_df["projected_score_opp"] = pd.to_numeric(
        live_df["projected_score_opp"], errors="coerce"
    )
    live_df["team_name_clean"] = live_df["team"].fillna("").astype(str).apply(nr.clean_team_name)
    live_df["opponent_name_clean"] = live_df["opponent"].fillna("").astype(str).apply(nr.clean_team_name)
    if "vegas_win_prob" in live_df.columns:
        live_df["vegas_win_prob"] = pd.to_numeric(live_df["vegas_win_prob"], errors="coerce")
    else:
        live_df["vegas_win_prob"] = np.nan
    if "vegas_spread" in live_df.columns:
        live_df["vegas_spread"] = pd.to_numeric(live_df["vegas_spread"], errors="coerce")
    else:
        live_df["vegas_spread"] = np.nan
    if "vegas_provider" in live_df.columns:
        live_df["vegas_provider"] = live_df["vegas_provider"].fillna("").astype(str)
    else:
        live_df["vegas_provider"] = ""

    if fetch_market:
        vegas_frames: list[pd.DataFrame] = []
        unique_days = sorted(
            {
                parsed_day.date()
                for parsed_day in pd.to_datetime(live_df["date"], errors="coerce")
                if pd.notna(parsed_day)
            }
        )
        for day_value in unique_days:
            try:
                vegas_df = nr.fetch_vegas_lines(day_value, logger=logger)
            except Exception as ex:
                logger.warning(
                    "DASHBOARD vegas_failed "
                    f"day={day_value} error={type(ex).__name__}: {ex}"
                )
                continue
            if vegas_df is not None and not vegas_df.empty:
                vegas_frames.append(vegas_df)

        if vegas_frames:
            market_df = (
                pd.concat(vegas_frames, ignore_index=True)
                .drop_duplicates(
                    subset=["date", "team_name_clean", "opponent_name_clean"],
                    keep="first",
                )
                .reset_index(drop=True)
            )
            live_df = live_df.merge(
                market_df[
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
                suffixes=("", "_market"),
            )
            for col in ["vegas_win_prob", "vegas_spread", "vegas_provider"]:
                market_col = f"{col}_market"
                if market_col in live_df.columns:
                    live_df[col] = live_df[market_col].where(
                        live_df[market_col].notna() if col != "vegas_provider" else live_df[market_col] != "",
                        live_df[col],
                    )
                    live_df = live_df.drop(columns=[market_col])

    live_df["fair_vegas_win_prob"] = _raw_market_prob_to_fair(live_df["vegas_win_prob"])
    live_df["edge"] = live_df["win_prob"] - live_df["fair_vegas_win_prob"]
    live_df["abs_edge"] = live_df["edge"].abs()
    live_df["spread_edge"] = live_df["projected_spread"] - pd.to_numeric(
        live_df["vegas_spread"], errors="coerce"
    )
    live_df = live_df.reindex(columns=_empty_live_predictions_df().columns)
    return live_df.sort_values(["date", "edge"], ascending=[True, False], na_position="last")


def _confidence_bucket(abs_edge: float) -> str:
    if not np.isfinite(abs_edge):
        return "none"
    if abs_edge >= 0.12:
        return "high"
    if abs_edge >= 0.10:
        return "strong"
    if abs_edge >= nr.BET_EDGE_THRESHOLD:
        return "standard"
    return "none"


def _edge_bucket(abs_edge: float) -> str:
    if not np.isfinite(abs_edge):
        return "unknown"
    max_edge = float(nr.BET_EDGE_MAX_ABS)
    if abs_edge < nr.BET_EDGE_THRESHOLD:
        return f"<{nr.BET_EDGE_THRESHOLD:.2f}"
    if abs_edge < 0.10:
        return f"{nr.BET_EDGE_THRESHOLD:.2f}-0.10"
    if abs_edge <= max_edge:
        return f"0.10-{max_edge:.2f}"
    return f">{max_edge:.2f}"


def _spread_bucket(abs_spread: float) -> str:
    if not np.isfinite(abs_spread):
        return "unknown"
    if abs_spread < 4.0:
        return "<4"
    if abs_spread < 8.0:
        return "4-8"
    if abs_spread < 12.0:
        return "8-12"
    return "12+"


def _prepare_bet_candidates(
    frame: pd.DataFrame,
    edge_threshold: float = nr.BET_EDGE_THRESHOLD,
    kelly_scale: float = nr.BET_KELLY_SCALE,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["win_prob"] = pd.to_numeric(work["win_prob"], errors="coerce")
    work["vegas_win_prob"] = pd.to_numeric(work["vegas_win_prob"], errors="coerce")
    work["fair_vegas_win_prob"] = _raw_market_prob_to_fair(work["vegas_win_prob"])
    work["projected_spread"] = pd.to_numeric(work.get("projected_spread"), errors="coerce")
    work = work.dropna(
        subset=["date", "team", "opponent", "win_prob", "vegas_win_prob", "fair_vegas_win_prob"]
    ).copy()
    if work.empty:
        return _empty_bets_df()

    work["edge_raw"] = work["win_prob"] - work["fair_vegas_win_prob"]
    work["abs_edge"] = work["edge_raw"].abs()
    work = work.loc[
        (work["abs_edge"] >= float(edge_threshold))
        & (work["abs_edge"] <= float(nr.BET_EDGE_MAX_ABS))
    ].copy()
    work = work.loc[work["projected_spread"].abs() >= 4.0].copy()
    if work.empty:
        return pd.DataFrame()

    team_side_mask = work["edge_raw"] >= 0
    work["source_team"] = work["team"]
    work["source_opponent"] = work["opponent"]
    work["bet_on_team_side"] = team_side_mask.astype(bool)
    work["bet_team"] = np.where(team_side_mask, work["team"], work["opponent"])
    work["bet_opponent"] = np.where(team_side_mask, work["opponent"], work["team"])
    work["bet_win_prob"] = np.where(team_side_mask, work["win_prob"], 1.0 - work["win_prob"])
    work["bet_fair_prob"] = np.where(
        team_side_mask,
        work["fair_vegas_win_prob"],
        1.0 - work["fair_vegas_win_prob"],
    )
    opponent_priced_prob = _fair_market_prob_to_priced(1.0 - work["fair_vegas_win_prob"])
    work["bet_market_prob_raw"] = np.where(
        team_side_mask,
        work["vegas_win_prob"],
        opponent_priced_prob,
    )
    work["bet_win_prob"] = _clip_probs(work["bet_win_prob"])
    work["bet_fair_prob"] = _clip_probs(work["bet_fair_prob"])
    work["bet_market_prob_raw"] = _clip_probs(work["bet_market_prob_raw"])
    work["decimal_odds"] = 1.0 / work["bet_market_prob_raw"]
    b = work["decimal_odds"] - 1.0
    work["kelly_fraction"] = (
        (work["bet_win_prob"] * b - (1.0 - work["bet_win_prob"])) / b
    ).clip(lower=0.0)
    work["bet_size"] = (float(kelly_scale) * work["kelly_fraction"]).clip(lower=0.0)
    work["confidence_bucket"] = work["abs_edge"].apply(_confidence_bucket)
    work["edge_bucket"] = work["abs_edge"].apply(_edge_bucket)
    work["spread_bucket"] = work["projected_spread"].abs().apply(_spread_bucket)
    work = work.loc[work["bet_size"] > 0].copy()
    if work.empty:
        return pd.DataFrame()

    work["edge"] = work["abs_edge"]
    work["date"] = work["date"].astype(str)
    return work.sort_values(
        ["edge", "bet_size", "date"], ascending=[False, False, True]
    ).reset_index(drop=True)


def build_bets_dataframe(
    live_df: pd.DataFrame | None = None,
    edge_threshold: float = nr.BET_EDGE_THRESHOLD,
    kelly_scale: float = nr.BET_KELLY_SCALE,
) -> pd.DataFrame:
    if live_df is None:
        live_df = build_live_predictions_dataframe(fetch_market=True)
    prepared = _prepare_bet_candidates(
        live_df,
        edge_threshold=edge_threshold,
        kelly_scale=kelly_scale,
    )
    if prepared.empty:
        return _empty_bets_df()
    return prepared[
        [
            "date",
            "bet_team",
            "bet_opponent",
            "edge",
            "bet_win_prob",
            "bet_fair_prob",
            "bet_size",
            "confidence_bucket",
        ]
    ].rename(
        columns={
            "bet_team": "team",
            "bet_opponent": "opponent",
            "bet_win_prob": "win_prob",
            "bet_fair_prob": "vegas_prob",
        }
    ).assign(
        edge=lambda df: df["edge"].round(4),
        win_prob=lambda df: df["win_prob"].round(4),
        vegas_prob=lambda df: df["vegas_prob"].round(4),
        bet_size=lambda df: df["bet_size"].round(4),
        is_live=True,
    ).reset_index(drop=True)


def compute_betting_performance_summary(
    pred_log_df: pd.DataFrame | None = None,
    bets_log_df: pd.DataFrame | None = None,
    edge_threshold: float = nr.BET_EDGE_THRESHOLD,
    kelly_scale: float = nr.BET_KELLY_SCALE,
) -> dict[str, float | int | None]:
    df = load_predictions_log() if pred_log_df is None else pred_log_df.copy()
    historical_metrics: dict[str, float | int | None] = {
        "bet_count": 0,
        "settled_bets": 0,
        "win_rate": None,
        "roi": None,
        "avg_clv": None,
        "total_stake": 0.0,
        "total_profit": 0.0,
    }
    if df.empty:
        historical_metrics = historical_metrics
    else:
        bet_candidates = _prepare_bet_candidates(
            df,
            edge_threshold=edge_threshold,
            kelly_scale=kelly_scale,
        )
        historical_metrics["bet_count"] = int(len(bet_candidates))
        if not bet_candidates.empty:
            raw_df = df.copy()
            raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.date.astype(str)

            merged = bet_candidates.merge(
                raw_df,
                left_on=["date", "source_team", "source_opponent"],
                right_on=["date", "team", "opponent"],
                how="left",
                suffixes=("", "_raw"),
            )
            if not merged.empty:
                merged["actual_result"] = pd.to_numeric(merged["actual_result"], errors="coerce")
                settled = merged.loc[merged["actual_result"].notna()].copy()
                historical_metrics["settled_bets"] = int(len(settled))
                if not settled.empty:
                    settled["bet_won"] = np.where(
                        settled["bet_on_team_side"],
                        settled["actual_result"],
                        1.0 - settled["actual_result"],
                    )
                    settled["decimal_odds"] = 1.0 / _clip_probs(settled["bet_market_prob_raw"])
                    settled["profit"] = np.where(
                        settled["bet_won"] >= 0.5,
                        settled["bet_size"] * (settled["decimal_odds"] - 1.0),
                        -settled["bet_size"],
                    )
                    total_stake = float(settled["bet_size"].sum())
                    total_profit = float(settled["profit"].sum())
                    historical_metrics["total_stake"] = total_stake
                    historical_metrics["total_profit"] = total_profit
                    historical_metrics["win_rate"] = float(settled["bet_won"].mean())
                    historical_metrics["roi"] = (
                        float(total_profit / total_stake) if total_stake > 0 else None
                    )

                    closing_team_prob_raw = pd.to_numeric(
                        settled.get("closing_vegas_win_prob"), errors="coerce"
                    )
                    closing_team_prob_fair = _raw_market_prob_to_fair(closing_team_prob_raw)
                    settled["closing_bet_prob"] = np.where(
                        settled["bet_on_team_side"],
                        closing_team_prob_fair,
                        1.0 - closing_team_prob_fair,
                    )
                    clv_df = settled.loc[settled["closing_bet_prob"].notna()].copy()
                    if not clv_df.empty:
                        clv_df["clv"] = clv_df["closing_bet_prob"] - clv_df["bet_fair_prob"]
                        historical_metrics["avg_clv"] = float(clv_df["clv"].mean())

    live_metrics = compute_live_betting_summary(bets_log_df=bets_log_df)
    return {
        **historical_metrics,
        "historical_bet_count": historical_metrics["bet_count"],
        "historical_settled_bets": historical_metrics["settled_bets"],
        "historical_win_rate": historical_metrics["win_rate"],
        "historical_roi": historical_metrics["roi"],
        "historical_avg_clv": historical_metrics["avg_clv"],
        "historical_total_stake": historical_metrics["total_stake"],
        "historical_total_profit": historical_metrics["total_profit"],
        **live_metrics,
    }


def compute_betting_bucket_metrics(
    pred_log_df: pd.DataFrame | None = None,
    edge_threshold: float = nr.BET_EDGE_THRESHOLD,
    kelly_scale: float = nr.BET_KELLY_SCALE,
) -> pd.DataFrame:
    df = load_predictions_log() if pred_log_df is None else pred_log_df.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "bucket",
                "bet_count",
                "settled_bets",
                "win_rate",
                "roi",
                "total_stake",
                "total_profit",
            ]
        )

    bet_candidates = _prepare_bet_candidates(
        df,
        edge_threshold=edge_threshold,
        kelly_scale=kelly_scale,
    )
    if bet_candidates.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "bucket",
                "bet_count",
                "settled_bets",
                "win_rate",
                "roi",
                "total_stake",
                "total_profit",
            ]
        )

    raw_df = df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.date.astype(str)
    merged = bet_candidates.merge(
        raw_df,
        left_on=["date", "source_team", "source_opponent"],
        right_on=["date", "team", "opponent"],
        how="left",
        suffixes=("", "_raw"),
    )
    merged["actual_result"] = pd.to_numeric(merged["actual_result"], errors="coerce")
    settled = merged.loc[merged["actual_result"].notna()].copy()
    if settled.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "bucket",
                "bet_count",
                "settled_bets",
                "win_rate",
                "roi",
                "total_stake",
                "total_profit",
            ]
        )

    settled["bet_won"] = np.where(
        settled["bet_on_team_side"],
        settled["actual_result"],
        1.0 - settled["actual_result"],
    )
    settled["decimal_odds"] = 1.0 / _clip_probs(settled["bet_market_prob_raw"])
    settled["profit"] = np.where(
        settled["bet_won"] >= 0.5,
        settled["bet_size"] * (settled["decimal_odds"] - 1.0),
        -settled["bet_size"],
    )

    rows: list[dict[str, float | int | str | None]] = []
    for dimension, bucket_col in [("edge", "edge_bucket"), ("spread", "spread_bucket")]:
        grouped = settled.groupby(bucket_col, dropna=False)
        for bucket, bucket_df in grouped:
            total_stake = float(bucket_df["bet_size"].sum())
            total_profit = float(bucket_df["profit"].sum())
            rows.append(
                {
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "bet_count": int(len(bucket_df)),
                    "settled_bets": int(len(bucket_df)),
                    "win_rate": float(bucket_df["bet_won"].mean()),
                    "roi": float(total_profit / total_stake) if total_stake > 0 else None,
                    "total_stake": total_stake,
                    "total_profit": total_profit,
                }
            )

    return pd.DataFrame(rows).sort_values(["dimension", "bucket"]).reset_index(drop=True)


def append_live_bets_log(
    live_df: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
    edge_threshold: float = nr.BET_EDGE_THRESHOLD,
    kelly_scale: float = nr.BET_KELLY_SCALE,
) -> int:
    logger = _dashboard_logger(logger)
    if live_df is None:
        live_df = build_live_predictions_dataframe(fetch_market=True)

    prepared = _prepare_bet_candidates(
        live_df,
        edge_threshold=edge_threshold,
        kelly_scale=kelly_scale,
    )
    existing_df = load_bets_log()
    existing_keys = {
        (str(row["date"]).strip(), str(row["team"]).strip(), str(row["opponent"]).strip())
        for _, row in existing_df[["date", "team", "opponent"]].iterrows()
        if str(row["date"]).strip() and str(row["team"]).strip() and str(row["opponent"]).strip()
    }

    if prepared.empty:
        if not BETS_LOG_PATH.exists():
            existing_df.to_csv(BETS_LOG_PATH, index=False)
        return 0

    logged_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    log_df = pd.DataFrame(
        {
            "date": prepared["date"].astype(str),
            "team": prepared["bet_team"].astype(str),
            "opponent": prepared["bet_opponent"].astype(str),
            "source_team": prepared["source_team"].astype(str),
            "source_opponent": prepared["source_opponent"].astype(str),
            "bet_on_team_side": prepared["bet_on_team_side"].astype(bool),
            "edge": pd.to_numeric(prepared["edge"], errors="coerce"),
            "win_prob": pd.to_numeric(prepared["bet_win_prob"], errors="coerce"),
            "vegas_prob": pd.to_numeric(prepared["bet_fair_prob"], errors="coerce"),
            "market_prob_raw": pd.to_numeric(prepared["bet_market_prob_raw"], errors="coerce"),
            "bet_size": pd.to_numeric(prepared["bet_size"], errors="coerce"),
            "confidence_bucket": prepared["confidence_bucket"].astype(str),
            "edge_bucket": prepared["edge_bucket"].astype(str),
            "spread_bucket": prepared["spread_bucket"].astype(str),
            "projected_spread": pd.to_numeric(prepared["projected_spread"], errors="coerce"),
            "vegas_provider": prepared["vegas_provider"].fillna("").astype(str),
            "is_live": True,
            "logged_at_utc": logged_at,
            "actual_result": np.nan,
            "actual_margin": np.nan,
            "bet_won": np.nan,
            "profit": np.nan,
            "closing_vegas_prob": np.nan,
            "closing_fair_prob": np.nan,
            "closing_vegas_provider": "",
            "clv": np.nan,
        }
    ).reindex(columns=_empty_bets_log_df().columns)

    if log_df.empty:
        if not BETS_LOG_PATH.exists():
            existing_df.to_csv(BETS_LOG_PATH, index=False)
        return 0

    dedupe_mask = log_df.apply(
        lambda row: (
            str(row.get("date", "")).strip(),
            str(row.get("team", "")).strip(),
            str(row.get("opponent", "")).strip(),
        ) not in existing_keys,
        axis=1,
    )
    new_df = log_df.loc[dedupe_mask].copy()
    if new_df.empty:
        if not BETS_LOG_PATH.exists():
            existing_df.to_csv(BETS_LOG_PATH, index=False)
        return 0

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.reindex(columns=_empty_bets_log_df().columns)
    combined.to_csv(BETS_LOG_PATH, index=False)
    logger.info(
        "BETS live_logged "
        f"rows={len(new_df)} path={BETS_LOG_PATH}"
    )
    return int(len(new_df))


def _enrich_bets_with_results(
    bets_df: pd.DataFrame,
    pred_log_df: pd.DataFrame,
) -> pd.DataFrame:
    if bets_df is None or bets_df.empty:
        return _empty_bets_log_df()
    if pred_log_df is None or pred_log_df.empty:
        return bets_df.reindex(columns=_empty_bets_log_df().columns)

    work = bets_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date.astype(str)
    pred = pred_log_df.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.date.astype(str)

    merged = work.merge(
        pred[
            [
                "date",
                "team",
                "opponent",
                "actual_result",
                "actual_margin",
                "closing_vegas_win_prob",
                "closing_vegas_provider",
            ]
        ],
        left_on=["date", "source_team", "source_opponent"],
        right_on=["date", "team", "opponent"],
        how="left",
        suffixes=("", "_pred"),
    )

    merged["actual_result"] = pd.to_numeric(merged["actual_result_pred"], errors="coerce")
    merged["actual_margin"] = pd.to_numeric(merged["actual_margin_pred"], errors="coerce")
    merged["bet_won"] = np.where(
        merged["actual_result"].notna(),
        np.where(
            merged["bet_on_team_side"].fillna(False),
            merged["actual_result"],
            1.0 - merged["actual_result"],
        ),
        np.nan,
    )
    decimal_odds = 1.0 / _clip_probs(merged["market_prob_raw"])
    merged["profit"] = np.where(
        merged["bet_won"].notna(),
        np.where(
            merged["bet_won"] >= 0.5,
            merged["bet_size"] * (decimal_odds - 1.0),
            -merged["bet_size"],
        ),
        np.nan,
    )

    closing_team_prob_raw = pd.to_numeric(merged["closing_vegas_win_prob"], errors="coerce")
    closing_team_prob_fair = _raw_market_prob_to_fair(closing_team_prob_raw)
    merged["closing_vegas_prob"] = np.where(
        merged["bet_on_team_side"].fillna(False),
        closing_team_prob_raw,
        1.0 - closing_team_prob_raw,
    )
    merged["closing_fair_prob"] = np.where(
        merged["bet_on_team_side"].fillna(False),
        closing_team_prob_fair,
        1.0 - closing_team_prob_fair,
    )
    merged["closing_vegas_provider"] = merged["closing_vegas_provider"].fillna("").astype(str)
    merged["clv"] = np.where(
        merged["closing_fair_prob"].notna(),
        merged["closing_fair_prob"] - pd.to_numeric(merged["vegas_prob"], errors="coerce"),
        np.nan,
    )

    drop_cols = [
        "team_pred",
        "opponent_pred",
        "actual_result_pred",
        "actual_margin_pred",
        "closing_vegas_win_prob",
    ]
    merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])
    return merged.reindex(columns=_empty_bets_log_df().columns)


def update_live_bets_log(
    pred_log_df: pd.DataFrame | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = _dashboard_logger(logger)
    bets_df = load_bets_log()
    if bets_df.empty:
        bets_df.to_csv(BETS_LOG_PATH, index=False)
        return bets_df

    pred_df = load_predictions_log() if pred_log_df is None else pred_log_df.copy()
    enriched = _enrich_bets_with_results(bets_df, pred_df)
    enriched.to_csv(BETS_LOG_PATH, index=False)
    settled_rows = int(pd.to_numeric(enriched["bet_won"], errors="coerce").notna().sum())
    logger.info(
        "BETS live_results_updated "
        f"settled_rows={settled_rows} path={BETS_LOG_PATH}"
    )
    return enriched


def compute_live_betting_summary(
    bets_log_df: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    df = load_bets_log() if bets_log_df is None else bets_log_df.copy()
    metrics: dict[str, float | int | None] = {
        "live_bet_count": 0,
        "live_settled_bets": 0,
        "live_win_rate": None,
        "live_roi": None,
        "live_avg_clv": None,
        "live_total_stake": 0.0,
        "live_total_profit": 0.0,
    }
    if df.empty:
        return metrics

    df = df.loc[df["is_live"].fillna(False)].copy()
    metrics["live_bet_count"] = int(len(df))
    if df.empty:
        return metrics

    settled = df.loc[pd.to_numeric(df["bet_won"], errors="coerce").notna()].copy()
    metrics["live_settled_bets"] = int(len(settled))
    if settled.empty:
        return metrics

    total_stake = float(pd.to_numeric(settled["bet_size"], errors="coerce").sum())
    total_profit = float(pd.to_numeric(settled["profit"], errors="coerce").sum())
    metrics["live_total_stake"] = total_stake
    metrics["live_total_profit"] = total_profit
    metrics["live_win_rate"] = float(pd.to_numeric(settled["bet_won"], errors="coerce").mean())
    metrics["live_roi"] = float(total_profit / total_stake) if total_stake > 0 else None
    clv_df = settled.loc[pd.to_numeric(settled["clv"], errors="coerce").notna()].copy()
    if not clv_df.empty:
        metrics["live_avg_clv"] = float(pd.to_numeric(clv_df["clv"], errors="coerce").mean())
    return metrics


def compute_live_betting_performance_log(
    bets_log_df: pd.DataFrame | None = None,
    starting_bankroll: float = LIVE_BANKROLL_START,
) -> pd.DataFrame:
    df = load_bets_log() if bets_log_df is None else bets_log_df.copy()
    columns = [
        "date",
        "bets_settled",
        "day_stake",
        "day_profit",
        "cumulative_profit",
        "bankroll",
        "rolling_roi_50",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    settled = df.loc[
        df["is_live"].fillna(False)
        & pd.to_numeric(df["bet_won"], errors="coerce").notna()
    ].copy()
    if settled.empty:
        return pd.DataFrame(columns=columns)

    settled["date"] = pd.to_datetime(settled["date"], errors="coerce").dt.date.astype(str)
    settled["bet_size"] = pd.to_numeric(settled["bet_size"], errors="coerce")
    settled["profit"] = pd.to_numeric(settled["profit"], errors="coerce")
    settled = settled.sort_values(["date", "logged_at_utc", "team", "opponent"]).reset_index(drop=True)
    settled["rolling_profit_50"] = settled["profit"].rolling(50, min_periods=1).sum()
    settled["rolling_stake_50"] = settled["bet_size"].rolling(50, min_periods=1).sum()
    settled["rolling_roi_50"] = settled["rolling_profit_50"] / settled["rolling_stake_50"].replace(0.0, np.nan)

    daily = (
        settled.groupby("date", sort=True)
        .agg(
            bets_settled=("team", "size"),
            day_stake=("bet_size", "sum"),
            day_profit=("profit", "sum"),
            rolling_roi_50=("rolling_roi_50", "last"),
        )
        .reset_index()
    )
    daily["cumulative_profit"] = pd.to_numeric(daily["day_profit"], errors="coerce").cumsum()
    daily["bankroll"] = float(starting_bankroll) + daily["cumulative_profit"]
    return daily[columns].reset_index(drop=True)


def write_dashboard_artifacts(
    logger: logging.Logger | None = None,
    fetch_market: bool = True,
) -> dict[str, object]:
    logger = _dashboard_logger(logger)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = _read_csv(GAME_PREDICTIONS_PATH)
    live_df = build_live_predictions_dataframe(
        predictions_df=predictions_df,
        logger=logger,
        fetch_market=fetch_market,
    )
    live_df.to_csv(LIVE_PREDICTIONS_PATH, index=False)

    top_edges_df = live_df[live_df["edge"].notna()].copy()
    top_edges_df = top_edges_df.sort_values(
        ["edge", "abs_edge", "date"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    top_edges_df.to_csv(TOP_EDGES_PATH, index=False)

    bets_df = build_bets_dataframe(live_df)
    bets_df.to_csv(BETS_PATH, index=False)

    pred_log_df = load_predictions_log()
    live_bets_logged = append_live_bets_log(live_df=live_df, logger=logger)
    bets_log_df = update_live_bets_log(pred_log_df=pred_log_df, logger=logger)
    metrics = compute_metrics_summary(pred_log_df)
    metrics_payload = {
        "updated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        **metrics,
    }
    MODEL_METRICS_PATH.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    daily_metrics_df = compute_daily_metrics(pred_log_df)
    daily_metrics_df.to_csv(DAILY_METRICS_PATH, index=False)

    calibration_df = compute_calibration_summary(pred_log_df)
    calibration_df.to_csv(CALIBRATION_PATH, index=False)

    betting_metrics = compute_betting_performance_summary(
        pred_log_df,
        bets_log_df=bets_log_df,
    )
    betting_bucket_df = compute_betting_bucket_metrics(pred_log_df)
    betting_bucket_df.to_csv(BETTING_BUCKET_METRICS_PATH, index=False)
    live_betting_perf_df = compute_live_betting_performance_log(bets_log_df)
    live_betting_perf_df.to_csv(LIVE_BETTING_PERFORMANCE_PATH, index=False)
    betting_payload = {
        "updated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        **betting_metrics,
    }
    BETTING_METRICS_PATH.write_text(
        json.dumps(betting_payload, indent=2),
        encoding="utf-8",
    )

    summary = {
        "live_rows": int(len(live_df)),
        "top_edge_rows": int(len(top_edges_df)),
        "bet_rows": int(len(bets_df)),
        "live_bets_logged": int(live_bets_logged),
        "live_bet_rows": int(len(bets_log_df)),
        "bet_bucket_rows": int(len(betting_bucket_df)),
        "rows_used": int(metrics.get("rows_used") or 0),
        "betting_metrics": betting_metrics,
        "paths": {
            "live_predictions": str(LIVE_PREDICTIONS_PATH),
            "top_edges": str(TOP_EDGES_PATH),
            "bets": str(BETS_PATH),
            "bets_log": str(BETS_LOG_PATH),
            "betting_buckets": str(BETTING_BUCKET_METRICS_PATH),
            "live_betting_performance": str(LIVE_BETTING_PERFORMANCE_PATH),
            "metrics": str(MODEL_METRICS_PATH),
            "daily_metrics": str(DAILY_METRICS_PATH),
            "calibration": str(CALIBRATION_PATH),
            "betting_metrics": str(BETTING_METRICS_PATH),
        },
    }
    logger.info(
        "DASHBOARD artifacts_written "
        f"live_rows={summary['live_rows']} top_edge_rows={summary['top_edge_rows']} "
        f"bet_rows={summary['bet_rows']} rows_used={summary['rows_used']}"
    )
    return summary
