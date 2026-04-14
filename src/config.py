from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = BASE_DIR / "params.yaml"


@dataclass(frozen=True)
class Settings:
    dataset: dict[str, Any]
    preprocessing: dict[str, Any]
    modeling: dict[str, Any]
    paths: dict[str, Any]


def load_settings() -> Settings:
    with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return Settings(
        dataset=payload["dataset"],
        preprocessing=payload["preprocessing"],
        modeling=payload["modeling"],
        paths=payload["paths"],
    )


def resolve_path(relative_path: str) -> Path:
    path = BASE_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
