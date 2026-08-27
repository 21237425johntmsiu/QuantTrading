"""Shared loader: read all credentials/config from the single env.txt at the repo root.

env.txt lives OUTSIDE Python/ and Rust/ so sensitive data (DB passwords,
MySQL credentials, MT5 account) is maintained in exactly one place.
Scripts should call db_config() / mysql_config() / mt5_credentials()
instead of hardcoding credentials. Importing this module also puts the
repo root on sys.path so the private root modules (feature_method.py)
and root assets (*.npy) are importable / resolvable.
"""
import sys
from pathlib import Path

_SELF = Path(__file__).resolve().parent

ENV_FILE = None
for _candidate in [_SELF, *_SELF.parents]:
    _candidate = _candidate / "env.txt"
    if _candidate.is_file():
        ENV_FILE = _candidate
        break

if ENV_FILE is None:
    raise FileNotFoundError(
        "env.txt not found above Python/ — expected at the repo root "
        "(next to the Python/ and Rust/ folders)"
    )

REPO_ROOT = ENV_FILE.parent

# Make the private root modules (e.g. feature_method.py) importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV = {}
for _line in ENV_FILE.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if not _line or _line.startswith("#") or "=" not in _line:
        continue
    _key, _value = _line.split("=", 1)
    ENV[_key.strip()] = _value.strip()


def db_config(**overrides):
    """TimescaleDB/PostgreSQL connection params from env.txt (DB_* keys)."""
    cfg = {
        "host": ENV.get("DB_HOST", "localhost"),
        "port": int(ENV.get("DB_PORT", "5432")),
        "user": ENV.get("DB_USER", "postgres"),
        "password": ENV.get("DB_PASSWORD", ""),
        "database": ENV.get("DB_NAME", "btcusd"),
    }
    cfg.update(overrides)
    return cfg


def mysql_config(**overrides):
    """MySQL connection params from env.txt (MYSQL_* keys)."""
    cfg = {
        "host": ENV.get("MYSQL_HOST", "192.168.1.86"),
        "user": ENV.get("MYSQL_USER", "root"),
        "password": ENV.get("MYSQL_PASSWORD", ""),
        "database": ENV.get("MYSQL_DB", "BTCUSD"),
        "charset": ENV.get("MYSQL_CHARSET", "utf8mb4"),
    }
    cfg.update(overrides)
    return cfg


def mt5_credentials():
    """MetaTrader5 login/password/server from env.txt (MT5_* keys)."""
    return {
        "login": int(ENV.get("MT5_LOGIN", "0")),
        "password": ENV.get("MT5_PASSWORD", ""),
        "server": ENV.get("MT5_SERVER", ""),
    }


def env_get(key, default=None, cast=None):
    """Read a single env.txt key, optionally cast (e.g. cast=float)."""
    value = ENV.get(key, default)
    return cast(value) if cast is not None else value


def asset_path(name):
    """Absolute path of a private asset kept at the repo root (e.g. *.npy)."""
    return str(REPO_ROOT / name)
