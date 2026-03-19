from __future__ import annotations

import argparse

from app import bracket_simulator
from app import run_daily


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HoopsIQ entry point")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Run the live-only daily pipeline.")
    mode.add_argument("--backfill", action="store_true", help="Run the full daily pipeline.")
    mode.add_argument("--bracket", action="store_true", help="Run March Madness bracket simulation.")
    parser.add_argument("--skip-market", action="store_true", help="Skip fresh market data fetches.")
    parser.add_argument("--skip-ranker", action="store_true", help="Reuse existing rating outputs.")
    parser.add_argument("--pdf", help="Path to a tournament bracket PDF.")
    parser.add_argument("--simulations", type=int, default=20000, help="Bracket simulation count.")
    parser.add_argument("--pool-size", type=int, default=50, help="Pool size for bracket EV mode.")
    args = parser.parse_args(argv)

    if args.live:
        live_args: list[str] = ["--live-only"]
        if args.skip_market:
            live_args.append("--skip-market")
        return run_daily.main(live_args)

    if args.backfill:
        daily_args: list[str] = []
        if args.skip_ranker:
            daily_args.append("--skip-ranker")
        if args.skip_market:
            daily_args.append("--skip-market")
        return run_daily.main(daily_args)

    if not args.pdf:
        parser.error("--pdf is required with --bracket")

    return bracket_simulator.main(
        [
            "--pdf",
            args.pdf,
            "--simulations",
            str(args.simulations),
            "--pool-size",
            str(args.pool_size),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
