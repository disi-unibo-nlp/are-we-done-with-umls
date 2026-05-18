from __future__ import annotations

import csv
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from .schemas import CuiMetadata


def _first_data_member(path: Path, suffix: str) -> str:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX/"):
                continue
            if name.endswith(suffix):
                return name
    raise ValueError(f"No {suffix} member found in {path}")


class CuiMetadataIndex:
    def __init__(self, rows: dict[str, CuiMetadata]) -> None:
        self._rows = rows
        self._exact_name_index: dict[str, list[str]] = defaultdict(list)
        self._normalized_name_index: dict[str, list[str]] = defaultdict(list)
        for cui, row in rows.items():
            if not row.name:
                continue
            exact = row.name.strip()
            normalized = self._normalize_name(row.name)
            if exact and cui not in self._exact_name_index[exact]:
                self._exact_name_index[exact].append(cui)
            if normalized and cui not in self._normalized_name_index[normalized]:
                self._normalized_name_index[normalized].append(cui)

    @classmethod
    def from_zip(cls, path: Path) -> "CuiMetadataIndex":
        member = _first_data_member(path, ".tsv")
        rows: dict[str, CuiMetadata] = {}
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as fh:
                text = (line.decode("utf-8") for line in fh)
                reader = csv.DictReader(text, delimiter="\t")
                for row in reader:
                    cui = row.get("CUI", "")
                    if not cui:
                        continue
                    rows[cui] = CuiMetadata(
                        cui=cui,
                        name=row.get("STR", "") or "",
                        tuis=[x for x in (row.get("TUI", "") or "").split("|") if x],
                        semantic_types=[x for x in (row.get("STY", "") or "").split("|") if x],
                        definition_source=row.get("DEF_SAB", "") or "",
                        definition=row.get("DEF", "") or "",
                    )
        return cls(rows)

    def get(self, cui: str) -> CuiMetadata:
        return self._rows.get(cui, CuiMetadata(cui=cui))

    def has_name(self, cui: str) -> bool:
        return bool(self.get(cui).name)

    def has_definition(self, cui: str) -> bool:
        return bool(self.get(cui).definition)

    def has_semantic_type(self, cui: str) -> bool:
        return bool(self.get(cui).semantic_types)

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(value.casefold().strip().split())

    def find_by_name(self, value: str | None) -> str | None:
        if not value:
            return None
        candidates = self.find_normalized_by_name(value)
        return candidates[0] if candidates else None

    def find_exact_by_name(self, value: str | None) -> list[str]:
        if not value:
            return []
        return list(self._exact_name_index.get(value.strip(), []))

    def find_normalized_by_name(self, value: str | None) -> list[str]:
        if not value:
            return []
        return list(self._normalized_name_index.get(self._normalize_name(value), []))


class RelationIndex:
    def __init__(
        self,
        adjacency: dict[str, set[str]],
        edge_labels: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> None:
        self._adjacency = adjacency
        self._edge_labels = edge_labels or {}

    @classmethod
    def from_zip(
        cls,
        path: Path,
        keep_cuis: set[str] | None = None,
        max_rows: int | None = None,
    ) -> "RelationIndex":
        member = _first_data_member(path, ".csv")
        adjacency: dict[str, set[str]] = defaultdict(set)
        edge_labels: dict[tuple[str, str], tuple[str, str]] = {}
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as fh:
                text = (line.decode("utf-8") for line in fh)
                reader = csv.DictReader(text)
                for idx, row in enumerate(reader):
                    if max_rows is not None and idx >= max_rows:
                        break
                    left = row.get("cui_1", "")
                    right = row.get("cui_2", "")
                    if not left or not right:
                        continue
                    if keep_cuis is not None and left not in keep_cuis and right not in keep_cuis:
                        continue
                    adjacency[left].add(right)
                    adjacency[right].add(left)
                    key = tuple(sorted((left, right)))
                    edge_labels.setdefault(key, (row.get("re", "") or "", row.get("rel_type", "") or ""))
        return cls(dict(adjacency), edge_labels=edge_labels)

    def degree(self, cui: str) -> int:
        return len(self._adjacency.get(cui, set()))

    def neighbors(self, cui: str) -> set[str]:
        return set(self._adjacency.get(cui, set()))

    def all_nodes(self) -> set[str]:
        nodes = set(self._adjacency)
        for neighbors in self._adjacency.values():
            nodes.update(neighbors)
        return nodes

    def relation_label(self, left: str, right: str) -> str:
        re_value, rel_type = self._edge_labels.get(tuple(sorted((left, right))), ("", ""))
        return rel_type or re_value or "related_to"

    def shortest_path_length(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        max_hops: int = 2,
    ) -> int | None:
        source_set = {cui for cui in sources if cui}
        target_set = {cui for cui in targets if cui}
        if not source_set or not target_set:
            return None
        if source_set & target_set:
            return 0

        queue: deque[tuple[str, int]] = deque((source, 0) for source in source_set)
        seen = set(source_set)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self._adjacency.get(node, set()):
                if neighbor in target_set:
                    return depth + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return None

    def shortest_path_nodes(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        max_hops: int = 2,
    ) -> list[str] | None:
        source_set = {cui for cui in sources if cui}
        target_set = {cui for cui in targets if cui}
        if not source_set or not target_set:
            return None
        shared = sorted(source_set & target_set)
        if shared:
            return [shared[0]]

        for source in sorted(source_set):
            queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
            seen = {source}
            while queue:
                node, path = queue.popleft()
                depth = len(path) - 1
                if depth >= max_hops:
                    continue
                for neighbor in sorted(self._adjacency.get(node, set())):
                    if neighbor in seen:
                        continue
                    next_path = [*path, neighbor]
                    if neighbor in target_set:
                        return next_path
                    seen.add(neighbor)
                    queue.append((neighbor, next_path))
        return None

    def path_nodes_between(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        max_hops: int = 2,
    ) -> set[str]:
        source_set = {cui for cui in sources if cui}
        out: set[str] = set()
        for target in sorted({cui for cui in targets if cui}):
            path = self.shortest_path_nodes(source_set, [target], max_hops=max_hops)
            if path:
                out.update(path)
        return out

    def path_edges_between(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        max_hops: int = 2,
    ) -> list[tuple[str, str, str]]:
        source_set = {cui for cui in sources if cui}
        edges: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for target in sorted({cui for cui in targets if cui}):
            path = self.shortest_path_nodes(source_set, [target], max_hops=max_hops)
            if not path or len(path) < 2:
                continue
            for left, right in zip(path, path[1:]):
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((left, right, self.relation_label(left, right)))
        return edges

    def pooled_neighborhood(self, seeds: Iterable[str], hops: int, cap: int | None = None) -> set[str]:
        nodes = {seed for seed in seeds if seed}
        frontier = set(nodes)
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in sorted(frontier):
                next_frontier.update(self._adjacency.get(node, set()))
                if cap is not None and len(nodes | next_frontier) >= cap:
                    break
            nodes.update(next_frontier)
            frontier = next_frontier
            if cap is not None and len(nodes) >= cap:
                return set(sorted(nodes)[:cap])
            if not frontier:
                break
        return nodes

    def edge_count(self, nodes: Iterable[str]) -> int:
        node_set = set(nodes)
        total = 0
        for node in node_set:
            total += len(self._adjacency.get(node, set()) & node_set)
        return total // 2

    def edges(self, nodes: Iterable[str], limit: int | None = None) -> list[tuple[str, str, str]]:
        node_set = set(nodes)
        out: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for left in sorted(node_set):
            for right in sorted(self._adjacency.get(left, set()) & node_set):
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                out.append((left, right, self.relation_label(left, right)))
                if limit is not None and len(out) >= limit:
                    return out
        return out

    def deterministic_sample(self, size: int, key: str = "") -> set[str]:
        nodes = sorted(self.all_nodes())
        if size <= 0 or not nodes:
            return set()
        if size >= len(nodes):
            return set(nodes)
        offset = sum(ord(char) for char in key) % len(nodes)
        rotated = nodes[offset:] + nodes[:offset]
        return set(rotated[:size])
