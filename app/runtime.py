from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess
import sys
import warnings


warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"

_CONFIG_CACHE: dict[Path, dict] = {}
RATINGS_OUTPUT_PATH = OUTPUTS_DIR / "teams_power_full.xlsx"
LIVE_REQUIRED_OUTPUTS = (
    OUTPUTS_DIR / "game_predictions.csv",
    OUTPUTS_DIR / "predictions_log.csv",
)


def resolve_project_root(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return REPO_ROOT
    path = Path(config_path).expanduser().resolve()
    if path.parent == CONFIG_DIR:
        return REPO_ROOT
    return path.parent


def load_public_config(config_path: str | Path | None = None, *, refresh: bool = False) -> dict:
    path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not refresh and path in _CONFIG_CACHE:
        return _CONFIG_CACHE[path]
    if not path.exists():
        _CONFIG_CACHE[path] = {}
        return _CONFIG_CACHE[path]
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    _CONFIG_CACHE[path] = payload if isinstance(payload, dict) else {}
    return _CONFIG_CACHE[path]


def ensure_runtime_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def build_public_logger(name: str, *, debug: bool = False) -> logging.Logger:
    cfg = load_public_config()
    logging_cfg = cfg.get("logging", {}) if isinstance(cfg.get("logging", {}), dict) else {}
    level_name = str(
        logging_cfg.get("debug_level" if debug else "level", "INFO")
    ).upper()
    format_key = "debug_format" if debug else "format"
    fmt = str(logging_cfg.get(format_key, "%(asctime)s %(levelname)s %(message)s"))
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _csv_has_data_rows(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
    except Exception:
        return False

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header_found = False
            for raw_line in fh:
                if raw_line.strip():
                    header_found = True
                    break
            if not header_found:
                return False
            for raw_line in fh:
                if raw_line.strip():
                    return True
    except Exception:
        return False
    return False


def _file_exists_with_content(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except Exception:
        return False


def ratings_output_status() -> dict[str, object]:
    ready = _file_exists_with_content(RATINGS_OUTPUT_PATH)
    return {
        "ready": ready,
        "missing_or_empty": [] if ready else [str(RATINGS_OUTPUT_PATH.relative_to(REPO_ROOT))],
    }


def live_outputs_status() -> dict[str, object]:
    missing_or_empty = [
        str(path.relative_to(REPO_ROOT))
        for path in LIVE_REQUIRED_OUTPUTS
        if not _csv_has_data_rows(path)
    ]
    return {
        "ready": not missing_or_empty,
        "missing_or_empty": missing_or_empty,
    }


def _run_main_command(args: list[str], logger: logging.Logger, log_message: str) -> bool:
    logger.info(log_message)
    try:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "main.py"), *args],
            cwd=REPO_ROOT,
            check=True,
        )
        return True
    except Exception as ex:
        logger.warning(
            "BOOTSTRAP command failed args=%s error=%s: %s",
            " ".join(args),
            type(ex).__name__,
            ex,
        )
        return False


def bootstrap_live_data_generation(logger: logging.Logger | None = None) -> dict[str, object]:
    ensure_runtime_dirs()
    logger = logger or build_public_logger("bootstrap")
    ratings_status = ratings_output_status()
    live_status = live_outputs_status()

    if ratings_status["ready"] and live_status["ready"]:
        return {
            "ran": False,
            "ready": True,
            "missing_or_empty": [],
            "ratings_ready": True,
        }

    ran = False
    if not ratings_status["ready"]:
        ran = True
        _run_main_command(
            ["--backfill"],
            logger,
            "BOOTSTRAP building ratings from scratch",
        )
        ratings_status = ratings_output_status()
        live_status = live_outputs_status()

    if ratings_status["ready"] and not live_status["ready"]:
        ran = True
        _run_main_command(
            ["--live"],
            logger,
            "BOOTSTRAP running live predictions",
        )
        live_status = live_outputs_status()

    return {
        "ran": ran,
        "ready": bool(ratings_status["ready"] and live_status["ready"]),
        "missing_or_empty": list(live_status["missing_or_empty"]),
        "ratings_ready": bool(ratings_status["ready"]),
        "ratings_missing_or_empty": list(ratings_status["missing_or_empty"]),
    }
