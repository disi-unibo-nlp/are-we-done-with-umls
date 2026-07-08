import tempfile
import unittest
import zipfile
from pathlib import Path

from src.experiments.curated_evidence import (
    auto_annotate_rows,
    build_curated_evidence_cache_rows,
    candidate_rows_for_items,
    validate_annotation_rows,
)
from src.experiments.schemas import QAItem
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex


def write_zip_text(path: Path, member: str, text: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)


class CuratedEvidenceTests(unittest.TestCase):
    def test_candidate_export_is_gold_blind_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_MID\tBridge concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID,RO,clinically_associated_with\n"
                "C_MID,C_A,RO,causes\n",
            )
            item = QAItem(
                item_id="medQA:test:1",
                dataset="medQA",
                split="test",
                linker="spacy",
                question="Which diagnosis is most likely?",
                options={"A": "Alpha"},
                gold="A",
                answer_text="Alpha",
                question_cuis=["C_Q"],
                option_cuis={"A": ["C_A"]},
            )
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_MID", "C_A"})

            first = candidate_rows_for_items([item], metadata, relations, depths=(2,))
            second = candidate_rows_for_items([item], metadata, relations, depths=(2,))

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["path_id"], second[0]["path_id"])
        self.assertNotIn("gold", first[0])
        self.assertNotIn("answer_idx", first[0])
        self.assertNotIn("model_prediction", first[0])
        self.assertEqual(first[0]["decision"], "")
        self.assertEqual(first[0]["path_rank"], 0)
        self.assertEqual(first[0]["option_label"], "A")
        self.assertEqual(first[0]["path_cuis"], ["C_Q", "C_MID", "C_A"])

    def test_validation_rejects_gold_leakage_and_changed_paths(self) -> None:
        candidate = {
            "item_id": "medQA:test:1",
            "depth": 2,
            "option_label": "A",
            "option_text": "Alpha",
            "target_cui": "C_A",
            "path_id": "path-1",
            "path_cuis": ["C_Q", "C_MID", "C_A"],
            "path_edges": [],
            "decision": "",
            "reviewer": "",
            "notes": "",
        }
        leaking = dict(candidate, decision="keep_clinical", reviewer="r1", gold="A")
        changed = dict(candidate, decision="keep_clinical", reviewer="r1", path_cuis=["C_Q", "C_OTHER", "C_A"])

        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_annotation_rows([leaking], candidate_rows=[candidate])
        with self.assertRaisesRegex(ValueError, "does not match candidate"):
            validate_annotation_rows([changed], candidate_rows=[candidate])

    def test_build_cache_requires_kept_path_for_every_answer_option(self) -> None:
        item = QAItem(
            item_id="medQA:test:1",
            dataset="medQA",
            split="test",
            linker="spacy",
            question="Question",
            options={"A": "Alpha", "B": "Beta"},
            gold="A",
            answer_text="Alpha",
            question_cuis=["C_Q"],
            option_cuis={"A": ["C_A"], "B": ["C_B"]},
        )
        kept_a = {
            "item_id": item.item_id,
            "dataset": item.dataset,
            "split": item.split,
            "linker": item.linker,
            "question": item.question,
            "options": item.options,
            "depth": 2,
            "option_label": "A",
            "option_text": "Alpha",
            "target_cui": "C_A",
            "path_id": "a",
            "path_cuis": ["C_Q", "C_MID_A", "C_A"],
            "path_edges": [
                {"left_cui": "C_Q", "left_name": "Question", "relation": "clinically associated with", "right_cui": "C_MID_A", "right_name": "A bridge"},
                {"left_cui": "C_MID_A", "left_name": "A bridge", "relation": "causes", "right_cui": "C_A", "right_name": "Alpha"},
            ],
            "decision": "keep_clinical",
            "reviewer": "r1",
            "notes": "",
        }
        dropped_b = {
            **kept_a,
            "option_label": "B",
            "option_text": "Beta",
            "target_cui": "C_B",
            "path_id": "b",
            "path_cuis": ["C_Q", "C_MID_B", "C_B"],
            "path_edges": [
                {"left_cui": "C_Q", "left_name": "Question", "relation": "has allowed qualifier", "right_cui": "C_MID_B", "right_name": "B bridge"},
                {"left_cui": "C_MID_B", "left_name": "B bridge", "relation": "can be qualified by", "right_cui": "C_B", "right_name": "Beta"},
            ],
            "decision": "drop_administrative",
        }
        rows = build_curated_evidence_cache_rows([item], [kept_a, dropped_b], depths=(2,))
        self.assertEqual(rows, [])

        kept_b = {**dropped_b, "decision": "keep_generic_valid"}
        rows = build_curated_evidence_cache_rows([item], [kept_a, kept_b], depths=(2,))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["condition"], "q_to_a_curated_depth_2_evidence")
        self.assertTrue(rows[0]["q_to_a_path_found"])
        self.assertEqual(rows[0]["q_to_a_answer_cui_count"], 2)
        self.assertEqual(rows[0]["q_to_a_answer_cui_covered_count"], 2)
        self.assertIn("Question -- clinically associated with --> A bridge", rows[0]["text"])
        self.assertIn("Question -- has allowed qualifier --> B bridge", rows[0]["text"])

    def test_auto_annotate_creates_silver_labels_without_answer_fields(self) -> None:
        base = {
            "item_id": "medQA:test:1",
            "dataset": "medQA",
            "split": "test",
            "linker": "spacy",
            "question": "Question",
            "options": {"A": "Alpha"},
            "depth": 2,
            "option_label": "A",
            "option_text": "Alpha",
            "target_cui": "C_A",
            "path_id": "path-1",
            "path_cuis": ["C_Q", "C_MID", "C_A"],
            "path_edges": [
                {"left_cui": "C_Q", "left_name": "Question", "relation": "finding site", "right_cui": "C_MID", "right_name": "Bridge"},
                {"left_cui": "C_MID", "left_name": "Bridge", "relation": "causes", "right_cui": "C_A", "right_name": "Alpha"},
            ],
            "decision": "",
            "reviewer": "",
            "notes": "",
        }
        admin = {
            **base,
            "path_id": "path-2",
            "path_edges": [
                {"left_cui": "C_Q", "left_name": "Question", "relation": "has allowed qualifier", "right_cui": "C_MID", "right_name": "Bridge"},
                {"left_cui": "C_MID", "left_name": "Bridge", "relation": "can be qualified by", "right_cui": "C_A", "right_name": "Alpha"},
            ],
        }

        rows = auto_annotate_rows([base, admin], reviewer="silver_test")

        self.assertEqual(rows[0]["decision"], "keep_clinical")
        self.assertEqual(rows[1]["decision"], "drop_administrative")
        self.assertEqual(rows[0]["reviewer"], "silver_test")
        self.assertNotIn("gold", rows[0])
        validate_annotation_rows(rows, candidate_rows=[base, admin])

    def test_auto_annotate_coverage_fallback_keeps_best_non_admin_path(self) -> None:
        candidate = {
            "item_id": "medQA:test:1",
            "dataset": "medQA",
            "split": "test",
            "linker": "spacy",
            "question": "Question",
            "options": {"A": "Alpha"},
            "depth": 4,
            "option_label": "A",
            "option_text": "Alpha",
            "target_cui": "C_A",
            "path_id": "fallback-path",
            "path_cuis": ["C_Q", "C_1", "C_2", "C_3", "C_A"],
            "path_edges": [
                {"left_cui": "C_Q", "left_name": "Question", "relation": "related to", "right_cui": "C_1", "right_name": "Lab:Panel"},
                {"left_cui": "C_1", "left_name": "Lab:Panel", "relation": "has subtype", "right_cui": "C_2", "right_name": "Intermediate"},
                {"left_cui": "C_2", "left_name": "Intermediate", "relation": "has child", "right_cui": "C_3", "right_name": "Clinical bridge"},
                {"left_cui": "C_3", "left_name": "Clinical bridge", "relation": "related to", "right_cui": "C_A", "right_name": "Alpha"},
            ],
            "decision": "",
            "reviewer": "",
            "notes": "",
        }

        strict_rows = auto_annotate_rows([candidate], reviewer="silver_test")
        fallback_rows = auto_annotate_rows([candidate], reviewer="silver_test", coverage_fallback=True)

        self.assertEqual(strict_rows[0]["decision"], "drop_spurious")
        self.assertEqual(fallback_rows[0]["decision"], "keep_generic_valid")
        self.assertIn("coverage fallback", fallback_rows[0]["notes"])
        self.assertGreater(fallback_rows[0]["curation_score"], 0)
        validate_annotation_rows(fallback_rows, candidate_rows=[candidate])

    def test_auto_annotate_coverage_fallback_rejects_hard_scaffold_path(self) -> None:
        candidate = {
            "item_id": "medQA:test:1",
            "dataset": "medQA",
            "split": "test",
            "linker": "spacy",
            "question": "Question",
            "options": {"A": "Alpha"},
            "depth": 4,
            "option_label": "A",
            "option_text": "Alpha",
            "target_cui": "C_A",
            "path_id": "hard-scaffold-path",
            "path_cuis": ["C_Q", "C_1", "C_2", "C_3", "C_A"],
            "path_edges": [
                {"left_cui": "C_Q", "left_name": "Question", "relation": "related to", "right_cui": "C_1", "right_name": "HL7 Vocabulary Version 2.5"},
                {"left_cui": "C_1", "left_name": "HL7 Vocabulary Version 2.5", "relation": "has child", "right_cui": "C_2", "right_name": "Intermediate"},
                {"left_cui": "C_2", "left_name": "Intermediate", "relation": "has child", "right_cui": "C_3", "right_name": "Clinical bridge"},
                {"left_cui": "C_3", "left_name": "Clinical bridge", "relation": "related to", "right_cui": "C_A", "right_name": "Alpha"},
            ],
            "decision": "",
            "reviewer": "",
            "notes": "",
        }

        rows = auto_annotate_rows([candidate], reviewer="silver_test", coverage_fallback=True)

        self.assertEqual(rows[0]["decision"], "drop_spurious")
        self.assertNotIn("curation_score", rows[0])
        validate_annotation_rows(rows, candidate_rows=[candidate])


if __name__ == "__main__":
    unittest.main()
