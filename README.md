# HoopsIQ

College basketball ratings, live predictions, betting signals, and NCAA tournament simulations in one repo.

## Screenshot

Add a dashboard screenshot here after deployment.

## Features

- Daily NCAA ratings and game prediction pipeline
- Streamlit dashboard for live games, edges, performance, and bracket forecasts
- Historical backtesting and live-vs-backtest diagnostics
- NCAA tournament PDF parser, Monte Carlo simulator, and pool-optimized bracket builder
- Repo-root entry points for local runs and deployment

## Quick Start

```bash
pip install -r requirements.txt
python main.py --live
python -m streamlit run app/dashboard.py
```

## Example Outputs

- `outputs/game_predictions.csv`: today’s model probabilities and spreads
- `outputs/bets.csv`: filtered betting signals with conservative sizing
- `outputs/bracket_predictions.csv`: title, Final Four, Elite 8, and Sweet 16 odds
- `outputs/pool_bracket.csv`: pool-oriented bracket picks with ownership leverage

## Folder Layout

```text
HoopsIQ/
├── app/                  # core pipeline, dashboard, and bracket simulator
├── config/               # runtime configuration
├── data/                 # static inputs and generated player impact table
├── logs/                 # runtime logs (gitignored)
├── outputs/              # generated artifacts (gitignored)
├── scripts/              # diagnostics and backtests
├── main.py               # repo-root CLI entry point
├── render.yaml           # Render deployment definition
└── requirements.txt
```

## Deployment

Render configuration is included in `render.yaml`.

- Web service: `streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port $PORT`
- Cron job: `python main.py --backfill`

Optional environment variables:

- `NCAA_API_BASE_URL`: private NCAA proxy, if available
- `HOOPSIQ_DEBUG=1`: enable more verbose logging

## License

Add a license before public release.
