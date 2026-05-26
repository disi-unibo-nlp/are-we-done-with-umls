from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QAItem:
    item_id: str
    dataset: str
    split: str
    linker: str
    question: str
    options: dict[str, str]
    gold: str | None
    answer_text: str | None
    question_cuis: list[str] = field(default_factory=list)
    option_cuis: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def gold_cuis(self) -> list[str]:
        if self.gold is None:
            return []
        return self.option_cuis.get(self.gold, [])

    def distractor_cuis(self) -> dict[str, list[str]]:
        return {
            key: value
            for key, value in self.option_cuis.items()
            if self.gold is None or key != self.gold
        }


@dataclass(frozen=True)
class CuiMetadata:
    cui: str
    name: str = ""
    tuis: list[str] = field(default_factory=list)
    semantic_types: list[str] = field(default_factory=list)
    definition_source: str = ""
    definition: str = ""

