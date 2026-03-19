from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app.dashboard_data import (
        BRACKET_MATCHUP_PROBS_PATH,
        BRACKET_PREDICTIONS_PATH,
        CALIBRATION_PATH,
        DAILY_METRICS_PATH,
        GAME_PREDICTIONS_PATH,
        LIVE_PREDICTIONS_PATH,
        MODEL_METRICS_PATH,
        MOST_LIKELY_BRACKET_PATH,
        POOL_BRACKET_PATH,
    )
    from app.runtime import OUTPUTS_DIR, bootstrap_live_data_generation, build_public_logger
else:
    from .dashboard_data import (
        BRACKET_MATCHUP_PROBS_PATH,
        BRACKET_PREDICTIONS_PATH,
        CALIBRATION_PATH,
        DAILY_METRICS_PATH,
        GAME_PREDICTIONS_PATH,
        LIVE_PREDICTIONS_PATH,
        MODEL_METRICS_PATH,
        MOST_LIKELY_BRACKET_PATH,
        POOL_BRACKET_PATH,
    )
    from .runtime import OUTPUTS_DIR, bootstrap_live_data_generation, build_public_logger

try:
    import altair as alt
except Exception:
    alt = None


@st.cache_resource(show_spinner=False)
def bootstrap_live_data() -> dict[str, object]:
    logger = build_public_logger("dashboard.bootstrap")
    return bootstrap_live_data_generation(logger=logger)


@st.cache_data(show_spinner=False)
def load_live_predictions() -> pd.DataFrame:
    if LIVE_PREDICTIONS_PATH.exists():
        try:
            df = pd.read_csv(LIVE_PREDICTIONS_PATH)
            if not df.empty:
                return df
        except Exception:
            pass
    if GAME_PREDICTIONS_PATH.exists():
        try:
            df = pd.read_csv(GAME_PREDICTIONS_PATH)
            if not df.empty:
                for col, default in {
                    "vegas_win_prob": pd.NA,
                    "fair_vegas_win_prob": pd.NA,
                    "vegas_spread": pd.NA,
                    "vegas_provider": "",
                    "edge": pd.NA,
                    "abs_edge": pd.NA,
                    "spread_edge": pd.NA,
                    "team_name_clean": "",
                    "opponent_name_clean": "",
                }.items():
                    if col not in df.columns:
                        df[col] = default
                return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_metrics_summary() -> dict[str, object]:
    if MODEL_METRICS_PATH.exists():
        try:
            return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "updated_at_utc": "unknown",
        "rows_used": 0,
        "brier_score": None,
        "log_loss": None,
        "spread_mae": None,
    }


@st.cache_data(show_spinner=False)
def load_daily_metrics_df() -> pd.DataFrame:
    if DAILY_METRICS_PATH.exists():
        try:
            return pd.read_csv(DAILY_METRICS_PATH)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_calibration_df() -> pd.DataFrame:
    if CALIBRATION_PATH.exists():
        try:
            return pd.read_csv(CALIBRATION_PATH)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_bracket_predictions_df() -> pd.DataFrame:
    if not BRACKET_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(BRACKET_PREDICTIONS_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_most_likely_bracket_df() -> pd.DataFrame:
    if not MOST_LIKELY_BRACKET_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(MOST_LIKELY_BRACKET_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pool_bracket_df() -> pd.DataFrame:
    if not POOL_BRACKET_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(POOL_BRACKET_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_bracket_matchup_probs_df() -> pd.DataFrame:
    if not BRACKET_MATCHUP_PROBS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(BRACKET_MATCHUP_PROBS_PATH)
    except Exception:
        return pd.DataFrame()


def format_metric(value: object, scale: float = 1.0, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value) * scale:.3f}{suffix}"
    except Exception:
        return "n/a"


def render_calibration_chart(chart_df):
    import streamlit as st
    import pandas as pd

    # Prevent crash when no data exists
    if chart_df is None or chart_df.empty:
        st.info("Calibration data not available yet")
        return

    if "avg_pred" not in chart_df.columns:
        st.warning("Calibration data missing expected columns")
        return

    chart_df["avg_pred"] = pd.to_numeric(chart_df["avg_pred"], errors="coerce")
    chart_df["actual_win_rate"] = pd.to_numeric(
        chart_df["actual_win_rate"], errors="coerce"
    )
    chart_df = chart_df.dropna(subset=["avg_pred", "actual_win_rate"])
    if chart_df.empty:
        st.info("No settled predictions available for calibration yet.")
        return

    if alt is None:
        st.line_chart(
            chart_df.set_index("bucket")[["avg_pred", "actual_win_rate"]],
            use_container_width=True,
        )
        return

    perfect = pd.DataFrame({"avg_pred": [0.0, 1.0], "actual_win_rate": [0.0, 1.0]})
    perfect_line = (
        alt.Chart(perfect)
        .mark_line(color="#9aa0a6", strokeDash=[4, 4])
        .encode(x="avg_pred:Q", y="actual_win_rate:Q")
    )
    actual_line = (
        alt.Chart(chart_df)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("avg_pred:Q", title="Average Predicted Win Probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("actual_win_rate:Q", title="Actual Win Rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["bucket", "count", "avg_pred", "actual_win_rate"],
        )
    )
    st.altair_chart(perfect_line + actual_line, use_container_width=True)


def main() -> None:
    logger = build_public_logger("dashboard.startup")
    st.set_page_config(page_title="NCAAB Live Dashboard", layout="wide")
    st.title("NCAAB Live Prediction Dashboard")

    bootstrap_status = bootstrap_live_data()
    metrics = load_metrics_summary()
    live_df = load_live_predictions()
    daily_metrics_df = load_daily_metrics_df()
    calibration_df = load_calibration_df()
    bracket_predictions_df = load_bracket_predictions_df()
    most_likely_bracket_df = load_most_likely_bracket_df()
    pool_bracket_df = load_pool_bracket_df()
    bracket_matchup_probs_df = load_bracket_matchup_probs_df()

    updated_at = metrics.get("updated_at_utc", "unknown")
    st.caption(f"Data root: `{OUTPUTS_DIR}` | Updated: `{updated_at}`")
    if bootstrap_status.get("mode") == "light" and bootstrap_status.get("missing_or_empty"):
        st.info("Live predictions not generated yet. Cached artifacts are being used.")
    elif bootstrap_status.get("ran") and not bootstrap_status.get("ready"):
        st.warning("Live data bootstrap ran, but some required output files are still missing.")
    logger.info(
        "DASHBOARD startup_ready=%s mode=%s live_rows=%s rows_used=%s",
        str(True).lower(),
        bootstrap_status.get("mode", "unknown"),
        len(live_df),
        metrics.get("rows_used"),
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Settled Predictions", str(int(metrics.get("rows_used") or 0)))
    metric_cols[1].metric("Brier Score", format_metric(metrics.get("brier_score")))
    metric_cols[2].metric("Log Loss", format_metric(metrics.get("log_loss")))
    metric_cols[3].metric("Spread MAE", format_metric(metrics.get("spread_mae")))

    tabs = st.tabs(["Today's Games", "Top Edges", "Performance", "March Madness"])

    with tabs[0]:
        if live_df.empty:
            st.info("Live predictions not generated yet.")
        else:
            live_df = live_df.copy()
            live_df["date"] = pd.to_datetime(live_df["date"], errors="coerce").dt.date
            latest_day = live_df["date"].dropna().max()
            today_df = live_df[live_df["date"] == latest_day].copy()
            today_df["win_prob_pct"] = (pd.to_numeric(today_df["win_prob"], errors="coerce") * 100.0).round(1)
            today_df["vegas_win_prob_pct"] = (
                pd.to_numeric(today_df["vegas_win_prob"], errors="coerce") * 100.0
            ).round(1)
            today_df["edge_pct"] = (pd.to_numeric(today_df["edge"], errors="coerce") * 100.0).round(1)
            st.subheader(f"Slate: {latest_day}")
            st.dataframe(
                today_df[
                    [
                        "team",
                        "opponent",
                        "win_prob_pct",
                        "projected_spread",
                        "projected_score_team",
                        "projected_score_opp",
                        "vegas_win_prob_pct",
                        "vegas_spread",
                        "edge_pct",
                    ]
                ].rename(
                    columns={
                        "team": "Team",
                        "opponent": "Opponent",
                        "win_prob_pct": "Model Win %",
                        "projected_spread": "Model Spread",
                        "projected_score_team": "Proj Team Score",
                        "projected_score_opp": "Proj Opp Score",
                        "vegas_win_prob_pct": "Vegas Win %",
                        "vegas_spread": "Vegas Spread",
                        "edge_pct": "Edge %",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[1]:
        if live_df.empty or live_df["edge"].isna().all():
            st.warning("Vegas lines are not available in the live prediction artifact.")
        else:
            edge_df = live_df.copy()
            edge_df["date"] = pd.to_datetime(edge_df["date"], errors="coerce").dt.date
            edge_df["edge_pct"] = (pd.to_numeric(edge_df["edge"], errors="coerce") * 100.0).round(2)
            edge_df["vegas_win_prob_pct"] = (
                pd.to_numeric(edge_df["vegas_win_prob"], errors="coerce") * 100.0
            ).round(1)
            edge_df["model_win_prob_pct"] = (
                pd.to_numeric(edge_df["win_prob"], errors="coerce") * 100.0
            ).round(1)

            positive_edges = edge_df[edge_df["edge"] > 0].copy().sort_values(
                ["edge", "date"],
                ascending=[False, True],
            )
            disagreements = edge_df[edge_df["edge"].notna()].copy().sort_values(
                ["abs_edge", "date"],
                ascending=[False, True],
            )

            left_col, right_col = st.columns(2)
            with left_col:
                st.subheader("Best Betting Opportunities")
                st.dataframe(
                    positive_edges[
                        [
                            "date",
                            "team",
                            "opponent",
                            "model_win_prob_pct",
                            "vegas_win_prob_pct",
                            "projected_spread",
                            "vegas_spread",
                            "edge_pct",
                        ]
                    ].head(15).rename(
                        columns={
                            "date": "Date",
                            "team": "Team",
                            "opponent": "Opponent",
                            "model_win_prob_pct": "Model Win %",
                            "vegas_win_prob_pct": "Vegas Win %",
                            "projected_spread": "Model Spread",
                            "vegas_spread": "Vegas Spread",
                            "edge_pct": "Edge %",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with right_col:
                st.subheader("Largest Model vs Vegas Gaps")
                st.dataframe(
                    disagreements[
                        [
                            "date",
                            "team",
                            "opponent",
                            "model_win_prob_pct",
                            "vegas_win_prob_pct",
                            "edge_pct",
                            "vegas_provider",
                        ]
                    ].head(15).rename(
                        columns={
                            "date": "Date",
                            "team": "Team",
                            "opponent": "Opponent",
                            "model_win_prob_pct": "Model Win %",
                            "vegas_win_prob_pct": "Vegas Win %",
                            "edge_pct": "Edge %",
                            "vegas_provider": "Book",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[2]:
        st.subheader("Historical Performance")
        if daily_metrics_df.empty:
            st.info("No settled predictions available yet.")
        else:
            trend_df = daily_metrics_df.copy()
            trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
            trend_df = trend_df.sort_values("date")

            trend_cols = st.columns(2)
            with trend_cols[0]:
                st.caption("Daily Brier Score")
                st.line_chart(
                    trend_df.set_index("date")[["brier_score"]],
                    use_container_width=True,
                )
            with trend_cols[1]:
                st.caption("Daily Spread MAE")
                st.line_chart(
                    trend_df.set_index("date")[["spread_mae"]],
                    use_container_width=True,
                )

            st.dataframe(
                trend_df.rename(
                    columns={
                        "date": "Date",
                        "rows_used": "Rows",
                        "brier_score": "Brier",
                        "log_loss": "Log Loss",
                        "spread_mae": "Spread MAE",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Calibration Chart")
        render_calibration_chart(calibration_df)
        if not calibration_df.empty:
            st.dataframe(
                calibration_df.rename(
                    columns={
                        "bucket": "Bucket",
                        "count": "Count",
                        "avg_pred": "Avg Predicted Win %",
                        "actual_win_rate": "Actual Win %",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        st.subheader("Tournament Forecast")
        if bracket_predictions_df.empty:
            st.warning(
                "Bracket artifacts not found. Run `python main.py --bracket --pdf path/to/bracket.pdf` first."
            )
        else:
            odds_df = bracket_predictions_df.copy()
            for col in [
                "r32_prob",
                "sweet16_prob",
                "elite8_prob",
                "final4_prob",
                "title_prob",
            ]:
                odds_df[col] = (pd.to_numeric(odds_df[col], errors="coerce") * 100.0).round(1)

            left_col, right_col = st.columns(2)
            with left_col:
                st.caption("Title Odds")
                st.dataframe(
                    odds_df[
                        [
                            "team",
                            "seed",
                            "region",
                            "title_prob",
                            "final4_prob",
                            "elite8_prob",
                            "sweet16_prob",
                        ]
                    ]
                    .sort_values(["title_prob", "final4_prob"], ascending=[False, False])
                    .rename(
                        columns={
                            "team": "Team",
                            "seed": "Seed",
                            "region": "Region",
                            "title_prob": "Title %",
                            "final4_prob": "Final Four %",
                            "elite8_prob": "Elite 8 %",
                            "sweet16_prob": "Sweet 16 %",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with right_col:
                st.caption("Final Four Odds")
                st.dataframe(
                    odds_df[
                        [
                            "team",
                            "seed",
                            "region",
                            "final4_prob",
                            "elite8_prob",
                            "sweet16_prob",
                            "title_prob",
                        ]
                    ]
                    .sort_values(["final4_prob", "elite8_prob"], ascending=[False, False])
                    .rename(
                        columns={
                            "team": "Team",
                            "seed": "Seed",
                            "region": "Region",
                            "final4_prob": "Final Four %",
                            "elite8_prob": "Elite 8 %",
                            "sweet16_prob": "Sweet 16 %",
                            "title_prob": "Title %",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            bracket_cols = st.columns(2)
            with bracket_cols[0]:
                st.subheader("Most Likely Bracket")
                if most_likely_bracket_df.empty:
                    st.info("`outputs/most_likely_bracket.csv` is not available.")
                else:
                    display_df = most_likely_bracket_df.copy()
                    for col in ["win_prob", "pick_rate", "leverage_score"]:
                        if col in display_df.columns:
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
                    display_df["win_prob"] = (
                        pd.to_numeric(display_df["win_prob"], errors="coerce") * 100.0
                    ).round(1)
                    if "pick_rate" in display_df.columns:
                        display_df["pick_rate"] = (display_df["pick_rate"] * 100.0).round(1)
                    if "leverage_score" in display_df.columns:
                        display_df["leverage_score"] = (display_df["leverage_score"] * 100.0).round(1)
                    st.dataframe(
                        display_df.rename(
                            columns={
                                "round": "Round",
                                "region": "Region",
                                "team": "Team",
                                "seed": "Seed",
                                "opponent": "Opponent",
                                "seed_opponent": "Opp Seed",
                                "winner": "Pick",
                                "winner_seed": "Pick Seed",
                                "win_prob": "Pick Win %",
                                "pick_rate": "Public Pick %",
                                "leverage_score": "Leverage %",
                                "pool_size": "Pool Size",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            with bracket_cols[1]:
                st.subheader("Pool EV Bracket")
                if pool_bracket_df.empty:
                    st.info("`outputs/pool_bracket.csv` is not available.")
                else:
                    display_df = pool_bracket_df.copy()
                    for col in ["win_prob", "pick_rate", "leverage_score"]:
                        if col in display_df.columns:
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
                    display_df["win_prob"] = (
                        pd.to_numeric(display_df["win_prob"], errors="coerce") * 100.0
                    ).round(1)
                    if "pick_rate" in display_df.columns:
                        display_df["pick_rate"] = (display_df["pick_rate"] * 100.0).round(1)
                    if "leverage_score" in display_df.columns:
                        display_df["leverage_score"] = (display_df["leverage_score"] * 100.0).round(1)
                    st.dataframe(
                        display_df.rename(
                            columns={
                                "mode": "Mode",
                                "round": "Round",
                                "region": "Region",
                                "team": "Team",
                                "seed": "Seed",
                                "opponent": "Opponent",
                                "seed_opponent": "Opp Seed",
                                "winner": "Pick",
                                "winner_seed": "Pick Seed",
                                "win_prob": "Pick Win %",
                                "pick_rate": "Public Pick %",
                                "leverage_score": "Leverage %",
                                "pool_size": "Pool Size",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader("Possible Tournament Matchups")
            if bracket_matchup_probs_df.empty:
                st.info("`outputs/bracket_matchup_probs.csv` is not available.")
            else:
                matchup_df = bracket_matchup_probs_df.copy()
                matchup_df["win_prob"] = (
                    pd.to_numeric(matchup_df["win_prob"], errors="coerce") * 100.0
                ).round(1)
                st.dataframe(
                    matchup_df.rename(
                        columns={
                            "round": "Round",
                            "region": "Region",
                            "team": "Team",
                            "seed": "Seed",
                            "opponent": "Opponent",
                            "seed_opponent": "Opp Seed",
                            "win_prob": "Team Win %",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
