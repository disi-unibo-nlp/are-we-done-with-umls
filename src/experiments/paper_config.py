from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


DEFAULT_CONFIG_PATH = Path("config") / "paper_experiments.toml"


@dataclass(frozen=True)
class PaperExperimentConfig:
    primary_datasets: tuple[str, ...]
    secondary_datasets: tuple[str, ...]
    splits: tuple[str, ...]
    linkers: tuple[str, ...]
    evidence_views: tuple[str, ...]
    retrieval_methods: tuple[str, ...]
    downstream_probe_conditions: tuple[str, ...]

    @property
    def diagnostic_datasets(self) -> tuple[str, ...]:
        return (*self.primary_datasets, *self.secondary_datasets)


def _tuple_value(raw: object, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(raw)


def load_paper_experiment_config(path: Path = DEFAULT_CONFIG_PATH) -> PaperExperimentConfig:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("paper experiment config must contain a [datasets] table")
    paper = data.get("paper")
    if not isinstance(paper, dict):
        paper = {}
    return PaperExperimentConfig(
        primary_datasets=_tuple_value(datasets.get("primary"), "datasets.primary"),
        secondary_datasets=_tuple_value(datasets.get("secondary"), "datasets.secondary"),
        splits=_tuple_value(datasets.get("splits"), "datasets.splits"),
        linkers=_tuple_value(data.get("linkers", paper.get("linkers", datasets.get("linkers"))), "linkers"),
        evidence_views=_tuple_value(
            data.get("evidence_views", paper.get("evidence_views", datasets.get("evidence_views"))),
            "evidence_views",
        ),
        retrieval_methods=_tuple_value(
            data.get("retrieval_methods", paper.get("retrieval_methods", datasets.get("retrieval_methods"))),
            "retrieval_methods",
        ),
        downstream_probe_conditions=_tuple_value(
            data.get(
                "downstream_probe_conditions",
                paper.get("downstream_probe_conditions", datasets.get("downstream_probe_conditions")),
            ),
            "downstream_probe_conditions",
        ),
    )
