from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from app import ncaab_ranker as nr
    from app.dashboard_data import write_dashboard_artifacts
    from app.runtime import DEFAULT_CONFIG_PATH, REPO_ROOT, build_public_logger
else:
    from . import ncaab_ranker as nr
    from .dashboard_data import write_dashboard_artifacts
    from .runtime import DEFAULT_CONFIG_PATH, REPO_ROOT, build_public_logger


def run_ranker(logger) -> None:
    cmd = [sys.executable, "-m", "app.ncaab_ranker"]
    logger.info("RUN_DAILY step=ranker command=%s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ranker",
        action="store_true",
        help="Reuse existing outputs and only refresh dashboard artifacts.",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="Do not fetch fresh Vegas lines while building dashboard artifacts.",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Skip full rebuild/backfill and only refresh today's live predictions and bets.",
    )
    args = parser.parse_args(argv)

    logger = build_public_logger("run_daily")

    try:
        if args.live_only:
            live_summary = nr.refresh_live_predictions(
                cfg_path=str(DEFAULT_CONFIG_PATH),
                logger=logger,
            )
            artifact_summary = write_dashboard_artifacts(
                logger=logger,
                fetch_market=not args.skip_market,
            )
            logger.info("PREDICTIONS live_rows=%s", live_summary.get("rows"))
            logger.info("BETS generated=%s", artifact_summary.get("bet_rows"))
            logger.info(
                "RUN_DAILY live_only complete live_rows=%s schedule_rows=%s bet_rows=%s debug_path=%s",
                live_summary.get("rows"),
                live_summary.get("schedule_rows"),
                artifact_summary.get("bet_rows"),
                live_summary.get("debug_path"),
            )
            return 0

        if not args.skip_ranker:
            run_ranker(logger)

        # These are idempotent and keep the prediction log / metrics current.
        results_updated = nr.update_prediction_results(logger)
        metrics = nr.log_prediction_metrics(logger)
        artifact_summary = write_dashboard_artifacts(
            logger=logger,
            fetch_market=not args.skip_market,
        )

        logger.info(
            "RUN_DAILY complete results_updated=%s rows_used=%s live_rows=%s top_edge_rows=%s "
            "bet_rows=%s historical_roi=%s historical_win_rate=%s historical_avg_clv=%s "
            "live_bet_count=%s live_roi=%s live_win_rate=%s live_avg_clv=%s",
            results_updated,
            metrics.get("rows_used"),
            artifact_summary.get("live_rows"),
            artifact_summary.get("top_edge_rows"),
            artifact_summary.get("bet_rows"),
            artifact_summary.get("betting_metrics", {}).get("historical_roi"),
            artifact_summary.get("betting_metrics", {}).get("historical_win_rate"),
            artifact_summary.get("betting_metrics", {}).get("historical_avg_clv"),
            artifact_summary.get("betting_metrics", {}).get("live_bet_count"),
            artifact_summary.get("betting_metrics", {}).get("live_roi"),
            artifact_summary.get("betting_metrics", {}).get("live_win_rate"),
            artifact_summary.get("betting_metrics", {}).get("live_avg_clv"),
        )
        return 0
    except subprocess.CalledProcessError as ex:
        logger.error("RUN_DAILY failed step=ranker returncode=%s", ex.returncode)
        return int(ex.returncode or 1)
    except Exception as ex:
        logger.exception("RUN_DAILY failed error=%s: %s", type(ex).__name__, ex)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
