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
        self._sorted_adjacency = {
            node: tuple(sorted(neighbors))
            for node, neighbors in adjacency.items()
        }
        self._edge_labels = edge_labels or {}
        self._shortest_path_length_cache: dict[tuple[tuple[str, ...], tuple[str, ...], int], int | None] = {}
        self._shortest_path_nodes_cache: dict[tuple[tuple[str, ...], tuple[str, ...], int], tuple[str, ...] | None] = {}
        self._exact_path_nodes_cache: dict[tuple[tuple[str, ...], tuple[str, ...], int], tuple[str, ...] | None] = {}
        self._clean_exact_paths_cache: dict[
            tuple[tuple[str, ...], str, int, tuple[str, ...], int],
            tuple[tuple[str, ...], ...],
        ] = {}
        self._all_nodes_sorted_cache: tuple[str, ...] | None = None

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
        if self._all_nodes_sorted_cache is not None:
            return set(self._all_nodes_sorted_cache)
        nodes = set(self._adjacency)
        for neighbors in self._adjacency.values():
            nodes.update(neighbors)
        self._all_nodes_sorted_cache = tuple(sorted(nodes))
        return nodes

    def relation_label(self, left: str, right: str) -> str:
        re_value, rel_type = self._edge_labels.get(tuple(sorted((left, right))), ("", ""))
        return rel_type or re_value or "related_to"

    def _shortest_path_nodes_max2(self, source_set: set[str], target_set: set[str], max_hops: int) -> list[str] | None:
        shared = sorted(source_set & target_set)
        if shared:
            return [shared[0]]
        if max_hops <= 0:
            return None

        for source in sorted(source_set):
            source_neighbors = self._adjacency.get(source, set())
            direct = sorted(source_neighbors & target_set)
            if direct:
                return [source, direct[0]]
            if max_hops < 2:
                continue
            for target in sorted(target_set):
                target_neighbors = self._adjacency.get(target, set())
                common = source_neighbors & target_neighbors
                if common:
                    return [source, sorted(common)[0], target]
        return None

    def shortest_path_length(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        max_hops: int = 2,
    ) -> int | None:
        source_set = {cui for cui in sources if cui}
        target_set = {cui for cui in targets if cui}
        cache_key = (tuple(sorted(source_set)), tuple(sorted(target_set)), max_hops)
        if cache_key in self._shortest_path_length_cache:
            return self._shortest_path_length_cache[cache_key]
        if not source_set or not target_set:
            self._shortest_path_length_cache[cache_key] = None
            return None
        if max_hops <= 2:
            path = self._shortest_path_nodes_max2(source_set, target_set, max_hops)
            value = len(path) - 1 if path is not None else None
            self._shortest_path_length_cache[cache_key] = value
            return value

        queue: deque[tuple[str, int]] = deque((source, 0) for source in source_set)
        seen = set(source_set)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self._adjacency.get(node, set()):
                if neighbor in target_set:
                    self._shortest_path_length_cache[cache_key] = depth + 1
                    return depth + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        self._shortest_path_length_cache[cache_key] = None
        return None

    def shortest_path_nodes(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        max_hops: int = 2,
    ) -> list[str] | None:
        source_set = {cui for cui in sources if cui}
        target_set = {cui for cui in targets if cui}
        cache_key = (tuple(sorted(source_set)), tuple(sorted(target_set)), max_hops)
        if cache_key in self._shortest_path_nodes_cache:
            cached = self._shortest_path_nodes_cache[cache_key]
            return list(cached) if cached is not None else None
        if not source_set or not target_set:
            self._shortest_path_nodes_cache[cache_key] = None
            return None
        if max_hops <= 2:
            path = self._shortest_path_nodes_max2(source_set, target_set, max_hops)
            self._shortest_path_nodes_cache[cache_key] = tuple(path) if path is not None else None
            return path

        for source in sorted(source_set):
            queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
            seen = {source}
            while queue:
                node, path = queue.popleft()
                depth = len(path) - 1
                if depth >= max_hops:
                    continue
                for neighbor in self._sorted_adjacency.get(node, ()):
                    if neighbor in seen:
                        continue
                    next_path = [*path, neighbor]
                    if neighbor in target_set:
                        self._shortest_path_nodes_cache[cache_key] = tuple(next_path)
                        return next_path
                    seen.add(neighbor)
                    queue.append((neighbor, next_path))
        self._shortest_path_nodes_cache[cache_key] = None
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

    def exact_path_nodes_between(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        hops: int,
    ) -> set[str]:
        source_set = {cui for cui in sources if cui}
        out: set[str] = set()
        for target in sorted({cui for cui in targets if cui}):
            path = self.exact_path_nodes(source_set, [target], hops=hops)
            if path:
                out.update(path)
        return out

    def exact_path_edges_between(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        hops: int,
    ) -> list[tuple[str, str, str]]:
        source_set = {cui for cui in sources if cui}
        edges: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for target in sorted({cui for cui in targets if cui}):
            path = self.exact_path_nodes(source_set, [target], hops=hops)
            if not path or len(path) < 2:
                continue
            for left, right in zip(path, path[1:]):
                key = tuple(sorted((left, right)))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((left, right, self.relation_label(left, right)))
        return edges

    def clean_exact_paths_between(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        hops: int,
        forbidden_internal: Iterable[str] = (),
        max_paths_per_target: int = 2,
    ) -> dict[str, list[list[str]]]:
        """Return deterministic clean exact-hop paths for each target.

        A clean path starts at a source, ends at the requested target, has
        exactly ``hops`` edges, is simple, and does not use any
        ``forbidden_internal`` node as an internal bridge.
        """
        source_set = {cui for cui in sources if cui}
        target_set = {cui for cui in targets if cui}
        forbidden = {cui for cui in forbidden_internal if cui}
        out: dict[str, list[list[str]]] = {}
        for target in sorted(target_set):
            paths = self.clean_exact_paths(
                source_set,
                target,
                hops,
                forbidden_internal=forbidden,
                max_paths=max_paths_per_target,
            )
            if paths:
                out[target] = paths
        return out

    def clean_exact_paths(
        self,
        sources: Iterable[str],
        target: str,
        hops: int,
        forbidden_internal: Iterable[str] = (),
        max_paths: int = 2,
    ) -> list[list[str]]:
        source_set = {cui for cui in sources if cui}
        target = target.strip()
        forbidden = {cui for cui in forbidden_internal if cui}
        cache_key = (
            tuple(sorted(source_set)),
            target,
            hops,
            tuple(sorted(forbidden)),
            max_paths,
        )
        if cache_key in self._clean_exact_paths_cache:
            return [list(path) for path in self._clean_exact_paths_cache[cache_key]]
        if not source_set or not target or hops < 0 or max_paths <= 0:
            self._clean_exact_paths_cache[cache_key] = ()
            return []
        if hops == 0:
            paths = ((target,),) if target in source_set else ()
            self._clean_exact_paths_cache[cache_key] = paths
            return [list(path) for path in paths]

        # If an answer CUI is also linked in the question, do not let it become
        # the source for a path to another answer CUI.
        source_set = {
            source
            for source in source_set
            if source == target or source not in forbidden
        }
        if not source_set:
            self._clean_exact_paths_cache[cache_key] = ()
            return []

        if hops >= 5:
            fast_paths = self._target_guided_clean_exact_paths(
                source_set,
                target,
                hops,
                forbidden_internal=forbidden,
                max_paths=max_paths,
            )
            if fast_paths:
                cached = tuple(tuple(path) for path in fast_paths)
                self._clean_exact_paths_cache[cache_key] = cached
                return [list(path) for path in cached]

        if hops >= 3:
            fast_paths = self._source_guided_clean_exact_paths(
                source_set,
                target,
                hops,
                forbidden_internal=forbidden,
                max_paths=max_paths,
            )
            if fast_paths:
                cached = tuple(tuple(path) for path in fast_paths)
                self._clean_exact_paths_cache[cache_key] = cached
                return [list(path) for path in cached]

        left_depth = hops // 2
        right_depth = hops - left_depth
        left_paths = self._paths_at_depth_multi(
            source_set,
            left_depth,
            forbidden_internal=forbidden,
            max_paths_per_node=max_paths,
        )
        right_paths = self._paths_at_depth_multi(
            {target},
            right_depth,
            forbidden_internal=forbidden - {target},
            max_paths_per_node=max_paths,
        )

        paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for meet in sorted(set(left_paths) & set(right_paths)):
            for left_path in left_paths[meet]:
                for right_path in right_paths[meet]:
                    path = [*left_path, *reversed(right_path[:-1])]
                    key = tuple(path)
                    if key in seen:
                        continue
                    if len(path) != hops + 1 or len(path) != len(set(path)):
                        continue
                    if path[-1] != target or path[0] not in source_set:
                        continue
                    if set(path[1:-1]) & forbidden:
                        continue
                    seen.add(key)
                    paths.append(path)
                    if len(paths) >= max_paths:
                        self._clean_exact_paths_cache[cache_key] = tuple(tuple(item) for item in paths)
                        return paths
        self._clean_exact_paths_cache[cache_key] = tuple(tuple(item) for item in paths)
        return paths

    def _source_guided_clean_exact_paths(
        self,
        source_set: set[str],
        target: str,
        hops: int,
        forbidden_internal: set[str],
        max_paths: int,
    ) -> list[list[str]]:
        left_depth = hops // 2
        right_depth = hops - left_depth
        left_paths = self._paths_at_depth_multi(
            source_set,
            left_depth,
            forbidden_internal=forbidden_internal,
            max_paths_per_node=max_paths,
        )
        paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for meet in sorted(left_paths):
            for left_path in left_paths[meet]:
                suffixes = self._suffix_paths_to_target(
                    meet,
                    target,
                    right_depth,
                    forbidden_internal=forbidden_internal,
                    blocked=set(left_path[:-1]),
                    max_paths=max_paths - len(paths),
                )
                for suffix in suffixes:
                    path = [*left_path, *suffix[1:]]
                    key = tuple(path)
                    if key in seen:
                        continue
                    if len(path) != hops + 1 or len(path) != len(set(path)):
                        continue
                    if path[-1] != target or path[0] not in source_set:
                        continue
                    if set(path[1:-1]) & forbidden_internal:
                        continue
                    seen.add(key)
                    paths.append(path)
                    if len(paths) >= max_paths:
                        return paths
        return paths

    def _suffix_paths_to_target(
        self,
        start: str,
        target: str,
        depth: int,
        forbidden_internal: set[str],
        blocked: set[str],
        max_paths: int,
    ) -> list[tuple[str, ...]]:
        paths: list[tuple[str, ...]] = []

        def dfs(node: str, path: tuple[str, ...], remaining: int) -> None:
            if len(paths) >= max_paths:
                return
            if remaining == 0:
                if node == target:
                    paths.append(path)
                return
            for neighbor in self._sorted_adjacency.get(node, ()):
                if neighbor in blocked or neighbor in path:
                    continue
                if neighbor == target and remaining != 1:
                    continue
                if neighbor != target and neighbor in forbidden_internal:
                    continue
                dfs(neighbor, (*path, neighbor), remaining - 1)
                if len(paths) >= max_paths:
                    return

        if depth < 0 or max_paths <= 0:
            return []
        dfs(start, (start,), depth)
        return paths

    def _target_guided_clean_exact_paths(
        self,
        source_set: set[str],
        target: str,
        hops: int,
        forbidden_internal: set[str],
        max_paths: int,
        max_expansions: int = 200_000,
    ) -> list[list[str]]:
        """Find clean exact-hop paths without materializing full path frontiers."""
        target_layers = self._target_reachable_layers(
            target,
            hops,
            forbidden_internal=forbidden_internal,
        )
        paths: list[list[str]] = []
        expansions = 0

        def dfs(node: str, path: tuple[str, ...], remaining: int) -> None:
            nonlocal expansions
            if len(paths) >= max_paths or expansions >= max_expansions:
                return
            if remaining == 0:
                if node == target:
                    paths.append(list(path))
                return
            for neighbor in self._sorted_adjacency.get(node, ()):
                expansions += 1
                if neighbor in path:
                    continue
                if neighbor == target and remaining != 1:
                    continue
                if neighbor != target and neighbor in forbidden_internal:
                    continue
                if neighbor not in target_layers[remaining - 1]:
                    continue
                dfs(neighbor, (*path, neighbor), remaining - 1)
                if len(paths) >= max_paths or expansions >= max_expansions:
                    return

        for source in sorted(source_set):
            dfs(source, (source,), hops)
            if len(paths) >= max_paths or expansions >= max_expansions:
                break
        return paths

    def _target_reachable_layers(
        self,
        target: str,
        max_depth: int,
        forbidden_internal: set[str],
    ) -> list[set[str]]:
        layers: list[set[str]] = [{target}]
        for _ in range(max_depth):
            previous = layers[-1]
            current: set[str] = set()
            for node in previous:
                for neighbor in self._adjacency.get(node, set()):
                    if neighbor != target and neighbor in forbidden_internal:
                        continue
                    current.add(neighbor)
            layers.append(current)
            if not current:
                break
        while len(layers) <= max_depth:
            layers.append(set())
        return layers

    def exact_path_nodes(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        hops: int,
    ) -> list[str] | None:
        """Return a deterministic simple path with exactly ``hops`` edges."""
        source_set = {cui for cui in sources if cui}
        target_set = {cui for cui in targets if cui}
        cache_key = (tuple(sorted(source_set)), tuple(sorted(target_set)), hops)
        if cache_key in self._exact_path_nodes_cache:
            cached = self._exact_path_nodes_cache[cache_key]
            return list(cached) if cached is not None else None
        if not source_set or not target_set or hops < 0:
            self._exact_path_nodes_cache[cache_key] = None
            return None
        if hops == 0:
            shared = sorted(source_set & target_set)
            path = [shared[0]] if shared else None
            self._exact_path_nodes_cache[cache_key] = tuple(path) if path else None
            return path

        left_depth = hops // 2
        right_depth = hops - left_depth
        left_paths = self._paths_at_depth(source_set, left_depth)
        right_paths = self._paths_at_depth(target_set, right_depth)
        for meet in sorted(set(left_paths) & set(right_paths)):
            left_path = left_paths[meet]
            right_path = right_paths[meet]
            path = [*left_path, *reversed(right_path[:-1])]
            if len(path) == len(set(path)):
                self._exact_path_nodes_cache[cache_key] = tuple(path)
                return path
        self._exact_path_nodes_cache[cache_key] = None
        return None

    def _paths_at_depth(self, starts: set[str], depth: int) -> dict[str, tuple[str, ...]]:
        current = {node: (node,) for node in sorted(starts)}
        for _ in range(depth):
            next_layer: dict[str, tuple[str, ...]] = {}
            for node, path in sorted(current.items()):
                for neighbor in self._sorted_adjacency.get(node, ()):
                    if neighbor in path or neighbor in next_layer:
                        continue
                    next_layer[neighbor] = (*path, neighbor)
            current = next_layer
            if not current:
                break
        return current

    def _paths_at_depth_multi(
        self,
        starts: set[str],
        depth: int,
        forbidden_internal: set[str] | None = None,
        max_paths_per_node: int = 2,
    ) -> dict[str, list[tuple[str, ...]]]:
        forbidden_internal = forbidden_internal or set()
        current: dict[str, list[tuple[str, ...]]] = {
            node: [(node,)]
            for node in sorted(starts)
        }
        for _ in range(depth):
            next_layer: dict[str, list[tuple[str, ...]]] = {}
            for node in sorted(current):
                for path in current[node]:
                    for neighbor in self._sorted_adjacency.get(node, ()):
                        if neighbor in path or neighbor in forbidden_internal:
                            continue
                        paths = next_layer.setdefault(neighbor, [])
                        if len(paths) >= max_paths_per_node:
                            continue
                        paths.append((*path, neighbor))
            current = next_layer
            if not current:
                break
        return current

    def exact_path_edges(
        self,
        sources: Iterable[str],
        targets: Iterable[str],
        hops: int,
    ) -> list[tuple[str, str, str]]:
        path = self.exact_path_nodes(sources, targets, hops)
        if not path or len(path) < 2:
            return []
        return [
            (left, right, self.relation_label(left, right))
            for left, right in zip(path, path[1:])
        ]

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
        if self._all_nodes_sorted_cache is None:
            self.all_nodes()
        nodes = list(self._all_nodes_sorted_cache or ())
        if size <= 0 or not nodes:
            return set()
        if size >= len(nodes):
            return set(nodes)
        offset = sum(ord(char) for char in key) % len(nodes)
        rotated = nodes[offset:] + nodes[:offset]
        return set(rotated[:size])
