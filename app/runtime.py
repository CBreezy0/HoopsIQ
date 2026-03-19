from __future__ import annotations

import json
import logging
from pathlib import Path
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
