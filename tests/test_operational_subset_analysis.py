import unittest

from src.experiments.operational_subset_analysis import build_group_manifests


def row(item_id: str, condition: str, correct: bool) -> dict:
    return {
        "item_id": item_id,
        "dataset": "medQA",
        "split": "test",
        "linker": "metamap",
        "model": "Qwen/Qwen3.5-4B",
        "mode": "non_thinking",
        "condition": condition,
        "prediction": "A",
        "error": "",
        "is_correct": correct,
    }


class OperationalSubsetAnalysisTests(unittest.TestCase):
    def test_builds_baseline_wrong_manifest_with_helped_and_still_wrong_labels(self) -> None:
        manifests, summaries = build_group_manifests([
            row("already_known", "no_umls", True),
            row("already_known", "metadata_rich_evidence", True),
            row("helped", "no_umls", False),
            row("helped", "metadata_rich_evidence", True),
            row("still_wrong", "no_umls", False),
            row("still_wrong", "metadata_rich_evidence", False),
        ])

        manifest = next(iter(manifests.values()))
        labels = {entry["item_id"]: entry["subset_label"] for entry in manifest}

        self.assertEqual(labels, {"helped": "umls_helped", "still_wrong": "still_wrong"})
        self.assertEqual(summaries[0]["total_baseline_rows"], 3)
        self.assertEqual(summaries[0]["baseline_correct"], 1)
        self.assertEqual(summaries[0]["baseline_wrong"], 2)
        self.assertEqual(summaries[0]["umls_helped"], 1)
        self.assertEqual(summaries[0]["still_wrong"], 1)


if __name__ == "__main__":
    unittest.main()
