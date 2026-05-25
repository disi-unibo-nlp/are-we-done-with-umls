import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.experiments.diagnostics import (
    answer_grounding_metrics,
    evidence_availability_rows,
    retrieval_faithfulness_rows,
)
from src.experiments.cli_utils import load_items
from src.experiments.evidence_cards import build_evidence_card, build_evidence_card_details
from src.experiments.loaders import load_metamap_records, load_spacy_records
from src.experiments.make_paper_diagnostic_tables import build_paper_tables
from src.experiments.paper_config import PaperExperimentConfig, load_paper_experiment_config
from src.experiments.umls_index import CuiMetadataIndex, RelationIndex


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def write_zip_text(path: Path, member: str, text: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)


class LoaderTests(unittest.TestCase):
    def test_metamap_loader_preserves_gold_and_option_cuis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            write_jsonl(path, [{
                "question": "Which therapy is monitored with aPTT?",
                "options": {"A": "Warfarin", "B": "Heparin", "C": "Aspirin", "D": "Insulin"},
                "answer_idx": "B",
                "answer": "Heparin",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B1", "C_B2"],
                "option2": [],
                "option3": ["C_D"],
            }])

            item = load_metamap_records(path, dataset="medqa", split="test")[0]

            self.assertEqual(item.item_id, "medqa:test:0")
            self.assertEqual(item.gold, "B")
            self.assertEqual(item.answer_text, "Heparin")
            self.assertEqual(item.question_cuis, ["C_Q"])
            self.assertEqual(item.option_cuis["B"], ["C_B1", "C_B2"])

    def test_metamap_loader_handles_pubmedqa_answer_field_as_gold(self) -> None:
        # Real pubmedQA data has no answer_idx; gold must be derived from the answer field
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            write_jsonl(path, [{
                "question": "Does therapy improve survival?",
                "options": ["yes", "no", "maybe"],
                "answer": "yes",
                "concept": ["C_Q"],
                "option0": ["C_YES"],
                "option1": ["C_NO"],
                "option2": ["C_MAYBE"],
            }])
            item = load_metamap_records(path, dataset="pubmedQA", split="test")[0]
            self.assertEqual(item.gold, "yes")
            self.assertEqual(item.answer_text, "yes")
            self.assertEqual(item.options, {"yes": "yes", "no": "no", "maybe": "maybe"})

    def test_spacy_loader_groups_per_option_rows_using_reference_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "metamap.jsonl"
            spacy = root / "spacy.jsonl"
            write_jsonl(reference, [{
                "question": "Which cell matures in the thymus?",
                "options": {"A": "B cell", "B": "T cell", "C": "Neutrophil", "D": "Basophil"},
                "answer_idx": "B",
                "answer": "T cell",
                "concept": ["C_Q_REF"],
                "option0": [],
                "option1": [],
                "option2": [],
                "option3": [],
            }])
            write_jsonl(spacy, [
                {"sent": "q A", "ans": "B cell", "qc": ["C_Q"], "ac": ["C_A"]},
                {"sent": "q B", "ans": "T cell", "qc": ["C_Q"], "ac": ["C_B"]},
                {"sent": "q C", "ans": "Neutrophil", "qc": ["C_Q"], "ac": []},
                {"sent": "q D", "ans": "Basophil", "qc": ["C_Q"], "ac": ["C_D"]},
            ])

            item = load_spacy_records(
                spacy,
                dataset="medqa",
                split="test",
                options_per_item=4,
                reference_path=reference,
            )[0]

            self.assertEqual(item.item_id, "medqa:test:0")
            self.assertEqual(item.linker, "spacy")
            self.assertEqual(item.gold, "B")
            self.assertEqual(item.options["C"], "Neutrophil")
            self.assertEqual(item.question_cuis, ["C_Q"])
            self.assertEqual(item.option_cuis["B"], ["C_B"])

    def test_load_items_filters_by_item_id_file_before_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            item_ids = root / "ids.jsonl"
            write_jsonl(data_path, [
                {
                    "question": "First",
                    "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                    "answer_idx": "A",
                    "answer": "A",
                    "concept": [],
                    "option0": [],
                    "option1": [],
                    "option2": [],
                    "option3": [],
                },
                {
                    "question": "Second",
                    "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
                    "answer_idx": "B",
                    "answer": "B",
                    "concept": [],
                    "option0": [],
                    "option1": [],
                    "option2": [],
                    "option3": [],
                },
            ])
            write_jsonl(item_ids, [{"item_id": "medQA:test:1"}])

            items = load_items(
                "medQA",
                "test",
                "metamap",
                input_path=data_path,
                item_id_file=item_ids,
                limit=1,
            )

        self.assertEqual([item.item_id for item in items], ["medQA:test:1"])


class UMLSIndexTests(unittest.TestCase):
    def test_cui_metadata_index_reads_committed_zip_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cui_detail.tsv.zip"
            write_zip_text(
                path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_B\tHeparin\tT121|T109\tPharmacologic Substance|Organic Chemical\tMSH\tAn anticoagulant.\n"
                "C_Q\tMonitoring\tT033\tFinding\t\t\n",
            )

            index = CuiMetadataIndex.from_zip(path)

            self.assertEqual(index.get("C_B").name, "Heparin")
            self.assertEqual(index.get("C_B").semantic_types, ["Pharmacologic Substance", "Organic Chemical"])
            self.assertTrue(index.has_definition("C_B"))
            self.assertFalse(index.has_definition("C_Q"))

    def test_relation_index_detects_direct_and_two_hop_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "triplets.csv.zip"
            rows = [
                ["cui_1", "cui_2", "re", "rel_type"],
                ["C_Q", "C_MID", "RO", "associated_with"],
                ["C_MID", "C_B", "RO", "monitored_by"],
                ["C_Q", "C_A", "RO", "direct_relation"],
            ]
            text = "\n".join(",".join(row) for row in rows) + "\n"
            write_zip_text(path, "triplets.csv", text)

            index = RelationIndex.from_zip(path, keep_cuis={"C_Q", "C_MID", "C_B", "C_A"})

            self.assertEqual(index.degree("C_Q"), 2)
            self.assertEqual(index.shortest_path_length(["C_Q"], ["C_A"], max_hops=2), 1)
            self.assertEqual(index.shortest_path_length(["C_Q"], ["C_B"], max_hops=2), 2)
            self.assertIsNone(index.shortest_path_length(["C_A"], ["C_B"], max_hops=1))

    def test_relation_index_caches_repeated_shortest_path_lookups(self) -> None:
        index = RelationIndex(
            {
                "C_Q": {"C_MID", "C_NOISE"},
                "C_MID": {"C_Q", "C_B"},
                "C_B": {"C_MID"},
                "C_NOISE": {"C_Q"},
            }
        )

        self.assertEqual(index.shortest_path_nodes(["C_Q"], ["C_B"], max_hops=2), ["C_Q", "C_MID", "C_B"])
        self.assertEqual(index.shortest_path_nodes(["C_Q"], ["C_B"], max_hops=2), ["C_Q", "C_MID", "C_B"])
        self.assertEqual(index.shortest_path_length(["C_Q"], ["C_B"], max_hops=2), 2)
        self.assertEqual(index.shortest_path_length(["C_Q"], ["C_B"], max_hops=2), 2)

        self.assertEqual(len(index._shortest_path_nodes_cache), 1)
        self.assertEqual(len(index._shortest_path_length_cache), 1)

    def test_relation_index_caches_clean_exact_hop_paths(self) -> None:
        index = RelationIndex(
            {
                "C_Q": {"C_1", "C_NOISE"},
                "C_1": {"C_Q", "C_2"},
                "C_2": {"C_1", "C_3"},
                "C_3": {"C_2", "C_4"},
                "C_4": {"C_3", "C_A"},
                "C_A": {"C_4"},
                "C_NOISE": {"C_Q", "C_OTHER"},
                "C_OTHER": {"C_NOISE"},
            }
        )

        paths = index.clean_exact_paths(["C_Q"], "C_A", hops=5, max_paths=1)
        repeated = index.clean_exact_paths(["C_Q"], "C_A", hops=5, max_paths=1)

        self.assertEqual(paths, [["C_Q", "C_1", "C_2", "C_3", "C_4", "C_A"]])
        self.assertEqual(repeated, paths)
        self.assertEqual(len(index._clean_exact_paths_cache), 1)

    def test_relation_index_finds_clean_depth_four_path_with_target_fanout(self) -> None:
        adjacency = {
            "C_Q": {"C_1", "C_NOISE"},
            "C_1": {"C_Q", "C_2"},
            "C_2": {"C_1", "C_3"},
            "C_3": {"C_2", "C_A"},
            "C_A": {"C_3"},
            "C_NOISE": {"C_Q"},
        }
        for index in range(80):
            noise = f"C_A_NOISE_{index}"
            adjacency["C_A"].add(noise)
            adjacency[noise] = {"C_A"}
        index = RelationIndex(adjacency)

        paths = index.clean_exact_paths(["C_Q"], "C_A", hops=4, max_paths=1)

        self.assertEqual(paths, [["C_Q", "C_1", "C_2", "C_3", "C_A"]])


class DiagnosticTests(unittest.TestCase):
    def test_paper_config_loads_toml_without_yaml_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_experiments.toml"
            path.write_text(
                "[datasets]\n"
                'primary = ["medQA", "medMCQA"]\n'
                'secondary = ["pubmedQA"]\n'
                'splits = ["test"]\n'
                "\n"
                'linkers = ["metamap", "spacy"]\n'
                'evidence_views = ["MRCONSO_only", "full_committed_card"]\n'
                'retrieval_methods = ["pooled_1hop", "q_to_a_2hop", "random_same_size"]\n'
                'downstream_probe_conditions = ["no_umls", "q_to_a_path_evidence"]\n',
                encoding="utf-8",
            )

            config = load_paper_experiment_config(path)

        self.assertIsInstance(config, PaperExperimentConfig)
        self.assertEqual(config.primary_datasets, ("medQA", "medMCQA"))
        self.assertEqual(config.secondary_datasets, ("pubmedQA",))
        self.assertEqual(config.linkers, ("metamap", "spacy"))
        self.assertIn("q_to_a_path_evidence", config.downstream_probe_conditions)

    def test_answer_grounding_metrics_capture_gold_and_distractors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            write_jsonl(path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": [],
                "option3": ["C_D"],
            }])
            item = load_metamap_records(path, dataset="medqa", split="test")[0]

            row = answer_grounding_metrics(item)
            metrics = row["metrics"]

            self.assertEqual(row["experiment"], "answer_grounding")
            self.assertEqual(row["condition"], "item")
            self.assertTrue(metrics["question_grounded"])
            self.assertTrue(metrics["correct_answer_grounded"])
            self.assertFalse(metrics["all_options_grounded"])
            self.assertEqual(metrics["distractor_grounding_rate"], 2 / 3)
            self.assertEqual(metrics["num_question_cuis"], 1)
            self.assertEqual(metrics["num_gold_answer_cuis"], 1)
            self.assertEqual(metrics["num_total_answer_cuis"], 3)
            self.assertEqual(metrics["num_options_with_cui"], 3)
            self.assertEqual(metrics["option_cui_counts"], {"A": 1, "B": 1, "C": 0, "D": 1})

    def test_pubmedqa_answer_grounding_only_reports_question_level_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            write_jsonl(path, [{
                "question": "Does therapy improve survival?",
                "options": ["yes", "no", "maybe"],
                "answer_idx": "yes",
                "answer": "yes",
                "concept": ["C_Q"],
                "option0": ["C_YES"],
                "option1": ["C_NO"],
                "option2": ["C_MAYBE"],
            }])
            item = load_metamap_records(path, dataset="pubmedQA", split="test")[0]

            row = answer_grounding_metrics(item)
            metrics = row["metrics"]

        self.assertFalse(metrics["answer_option_metrics_applicable"])
        self.assertTrue(metrics["question_grounded"])
        self.assertEqual(metrics["num_question_cuis"], 1)
        self.assertNotIn("correct_answer_grounded", metrics)

    def test_evidence_availability_rows_report_metadata_by_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": [],
                "option3": ["C_D"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\tMSH\tGold definition.\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n"
                "C_D\tDelta\t\t\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_B,RO,associated_with\n",
            )

            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            rows = evidence_availability_rows(
                item,
                CuiMetadataIndex.from_zip(meta_path),
                RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_B"}),
            )

            full = next(row for row in rows if row["view"] == "full_committed_card")
            relation_only = next(row for row in rows if row["condition"] == "MRCONSO_MRREL")
            metrics = full["metrics"]

            self.assertEqual(full["experiment"], "evidence_availability")
            self.assertTrue(metrics["gold_answer_grounded"])
            self.assertTrue(metrics["gold_has_name"])
            self.assertTrue(metrics["gold_has_definition"])
            self.assertTrue(metrics["gold_has_semantic_type"])
            self.assertEqual(metrics["gold_relation_degree"], 1)
            self.assertEqual(metrics["distractor_metadata_coverage"], 2 / 3)
            self.assertTrue(metrics["gold_answer_evidence_available"])
            self.assertTrue(metrics["distractor_contrast_available"])
            self.assertTrue(metrics["evidence_view_adds_definition"])
            self.assertTrue(metrics["evidence_view_adds_semantic_type"])
            self.assertTrue(metrics["evidence_view_adds_new_signal_over_relation_only"])
            self.assertFalse(relation_only["metrics"]["evidence_view_adds_definition"])
            self.assertFalse(relation_only["metrics"]["evidence_view_adds_new_signal_over_relation_only"])

    def test_retrieval_faithfulness_rows_compare_path_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": ["C_C"],
                "option3": ["C_D"],
            }])
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID,RO,associated_with\n"
                "C_MID,C_B,RO,monitored_by\n"
                "C_Q,C_A,RO,noise\n",
            )

            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            rows = retrieval_faithfulness_rows(
                item,
                RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_MID", "C_B", "C_A"}),
            )
            by_method = {row["method"]: row for row in rows}
            q_to_a = by_method["q_to_a_2hop"]["metrics"]

            self.assertEqual(by_method["q_to_a_2hop"]["experiment"], "retrieval_faithfulness")
            self.assertEqual(by_method["q_to_a_2hop"]["condition"], "q_to_a_2hop")
            self.assertTrue(q_to_a["gold_path_available"])
            self.assertEqual(q_to_a["gold_min_path_length"], 2)
            self.assertEqual(q_to_a["distractor_path_rate"], 1 / 3)
            self.assertEqual(q_to_a["num_question_seed_nodes"], 1)
            self.assertEqual(q_to_a["num_answer_seed_nodes"], 4)
            self.assertEqual(q_to_a["num_gold_answer_seed_nodes"], 1)
            self.assertEqual(q_to_a["num_retrieved_nodes"], 4)
            self.assertEqual(q_to_a["answer_connected_node_count"], 4)
            self.assertEqual(q_to_a["decision_relevance_ratio_proxy"], 1.0)
            self.assertGreaterEqual(by_method["pooled_1hop"]["metrics"]["num_retrieved_nodes"], 2)

    def test_evidence_card_serializes_metadata_and_paths_by_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": [],
                "option3": ["C_D"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\tMSH\tGold definition.\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID,RO,associated_with\n"
                "C_MID,C_B,RO,monitored_by\n"
                "C_Q,C_A,RO,noise\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_MID", "C_B", "C_A"})

            metadata_card = build_evidence_card(item, "metadata_rich_evidence", metadata, relations)
            path_card = build_evidence_card(item, "q_to_a_path_evidence", metadata, relations)
            no_card = build_evidence_card(item, "no_umls", metadata, relations)

        self.assertIsNone(no_card)
        self.assertIn("Relevant concepts:", metadata_card)
        self.assertIn("Definition: Gold definition.", metadata_card)
        self.assertIn("Semantic types: Disease or Syndrome", metadata_card)
        self.assertIn("Q-to-answer paths:", metadata_card)
        self.assertIn("Question concept -- associated with --> unnamed UMLS concept", metadata_card)
        self.assertIn("unnamed UMLS concept -- monitored by --> Beta", metadata_card)
        self.assertNotIn("Question concepts:", metadata_card)
        self.assertNotIn("Answer-option concepts:", metadata_card)
        self.assertNotIn("Definition: Gold definition.", path_card)
        self.assertNotIn("Relevant concepts:", path_card)
        self.assertNotIn("Question concepts:", path_card)
        self.assertNotIn("Answer-option concepts:", path_card)
        self.assertIn("Q-to-answer paths:", path_card)
        self.assertIn("Question concept -- associated with --> unnamed UMLS concept", path_card)
        self.assertIn("unnamed UMLS concept -- monitored by --> Beta", path_card)
        self.assertNotIn("C_MID --monitored_by-- C_B", path_card)

    def test_relation_only_evidence_uses_plain_language_graph_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": [],
                "option3": [],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\t\t\n"
                "C_MID\tMiddle concept\tT033\tFinding\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID,RO,associated_with\n"
                "C_Q,C_A,RB,\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_MID", "C_A", "C_B"})

            card = build_evidence_card(item, "relation_only_evidence", metadata, relations)

        self.assertIn("UMLS graph:", card)
        self.assertIn("- Question concept -- associated with --> Middle concept", card)
        self.assertIn("- Question concept -- has broader relation to --> Alpha", card)
        self.assertNotIn("C_Q", card)
        self.assertNotIn("RO", card)
        self.assertNotIn("RB", card)
        self.assertNotIn("Question concepts:", card)
        self.assertNotIn("Answer-option concepts:", card)

    def test_metadata_rich_evidence_augments_q_to_a_paths_with_cuis_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": [],
                "option3": [],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\tMSH\tQuestion definition.\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\tMSH\tAlpha definition.\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\tMSH\tBeta definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID,RO,associated_with\n"
                "C_MID,C_B,RO,monitored_by\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_B", "C_MID"})

            card = build_evidence_card(item, "metadata_rich_evidence", metadata, relations)

        self.assertIn("Relevant concepts:", card)
        self.assertIn("- Beta (CUI: C_B)", card)
        self.assertIn("Definition: Beta definition.", card)
        self.assertIn("Semantic types: Disease or Syndrome", card)
        self.assertIn("Q-to-answer paths:", card)
        self.assertIn("UMLS graph:", card)
        self.assertIn("- Question concept -- associated with --> unnamed UMLS concept", card)
        self.assertIn("- unnamed UMLS concept -- monitored by --> Beta", card)
        self.assertNotIn("B C_B", card)
        self.assertNotIn("Question concepts:", card)
        self.assertNotIn("Answer-option concepts:", card)

    def test_evidence_card_serializes_gold_only_and_distractor_contrast_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": [],
                "option3": ["C_D"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\tMSH\tGold definition.\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\tMSH\tDistractor definition.\n"
                "C_D\tDelta\tT047\tDisease or Syndrome\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_B,RO,gold_relation\nC_Q,C_A,RO,distractor_relation\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_B", "C_A", "C_D"})

            gold_only = build_evidence_card(item, "gold_only_evidence", metadata, relations)
            contrast = build_evidence_card(item, "distractor_contrast_evidence", metadata, relations)

        self.assertIn("Gold answer evidence:", gold_only)
        self.assertIn("B C_B: Beta", gold_only)
        self.assertNotIn("A C_A: Alpha", gold_only)
        self.assertIn("Distractor contrast evidence:", contrast)
        self.assertIn("Option B evidence:", contrast)
        self.assertIn("Option A evidence:", contrast)
        self.assertNotIn("Gold option", contrast)
        self.assertNotIn("Distractor option", contrast)
        self.assertIn("Distractor definition.", contrast)

    def test_q_to_a_exact_depth_condition_uses_all_answer_paths_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": ["C_C"],
                "option3": ["C_D"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\tMSH\tAlpha definition.\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\tMSH\tBeta definition.\n"
                "C_C\tGamma\tT047\tDisease or Syndrome\tMSH\tGamma definition.\n"
                "C_D\tDelta\tT047\tDisease or Syndrome\tMSH\tDelta definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID_A,RO,alpha_bridge\n"
                "C_MID_A,C_A,RO,alpha_relation\n"
                "C_Q,C_MID_B,RO,beta_bridge\n"
                "C_MID_B,C_B,RO,beta_relation\n"
                "C_Q,C_MID_C,RO,gamma_bridge\n"
                "C_MID_C,C_C,RO,gamma_relation\n"
                "C_Q,C_MID_D,RO,delta_bridge\n"
                "C_MID_D,C_D,RO,delta_relation\n"
                "C_Q,C_NOISE,RO,noise\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(
                rel_path,
                keep_cuis={
                    "C_Q", "C_A", "C_B", "C_C", "C_D",
                    "C_MID_A", "C_MID_B", "C_MID_C", "C_MID_D", "C_NOISE",
                },
            )

            card = build_evidence_card(
                item,
                "q_to_a_exact_depth_2_evidence",
                metadata,
                relations,
            )

        self.assertTrue(card.startswith("- Question concept -- alpha bridge --> unnamed UMLS concept"))
        self.assertNotIn("Evidence condition:", card)
        self.assertNotIn("q_to_a_exact_depth_2_evidence", card)
        self.assertNotIn("depth=2", card)
        self.assertNotIn("exact-depth", card)
        self.assertNotIn("Q-to-answer", card)
        self.assertNotIn("UMLS graph:", card)
        self.assertNotIn("Paths to", card)
        self.assertNotIn("- Path", card)
        self.assertNotIn("CUI:", card)
        self.assertIn("Question concept -- alpha bridge --> unnamed UMLS concept", card)
        self.assertIn("unnamed UMLS concept -- alpha relation --> Alpha", card)
        self.assertIn("Question concept -- beta bridge --> unnamed UMLS concept", card)
        self.assertIn("unnamed UMLS concept -- beta relation --> Beta", card)
        self.assertNotIn("Question concepts:", card)
        self.assertNotIn("Answer-option concepts:", card)
        self.assertNotIn("Definition:", card)
        self.assertNotIn("Gold answer evidence:", card)
        self.assertNotIn("width", card)

    def test_q_to_a_exact_depth_card_reports_missing_exact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta"},
                "answer_idx": "A",
                "answer": "Alpha",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_A,RO,direct_relation\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_B"})

            card = build_evidence_card(item, "q_to_a_exact_depth_2_evidence", metadata, relations)

        self.assertIn("No complete UMLS path set found.", card)
        self.assertNotIn("Q-to-answer", card)
        self.assertNotIn("Evidence condition:", card)
        self.assertNotIn("q_to_a_exact_depth_2_evidence", card)
        self.assertNotIn("2-hop", card)
        self.assertNotIn("direct relation", card)

    def test_q_to_a_exact_depth_requires_every_answer_option_to_have_linked_cui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta"},
                "answer_idx": "A",
                "answer": "Alpha",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": [],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_A,RO,direct_relation\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A"})

            details = build_evidence_card_details(item, "q_to_a_exact_depth_1_evidence", metadata, relations)

        self.assertFalse(details.q_to_a_path_found)
        self.assertEqual(details.q_to_a_answer_cui_count, 1)
        self.assertEqual(details.q_to_a_answer_cui_covered_count, 1)
        self.assertIn("No complete UMLS path set found.", details.text or "")
        self.assertNotIn("direct relation", details.text or "")

    def test_q_to_a_exact_depth_requires_every_answer_cui_to_have_clean_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta"},
                "answer_idx": "A",
                "answer": "Alpha",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\t\t\n"
                "C_MID_A\tMid alpha\tT033\tFinding\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID_A,RO,alpha_bridge\n"
                "C_MID_A,C_A,RO,alpha_relation\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_B", "C_MID_A"})

            details = build_evidence_card_details(item, "q_to_a_exact_depth_2_evidence", metadata, relations)

        self.assertFalse(details.q_to_a_path_found)
        self.assertEqual(details.q_to_a_answer_cui_count, 2)
        self.assertEqual(details.q_to_a_answer_cui_covered_count, 1)
        self.assertIn("No complete UMLS path set", details.text or "")
        self.assertNotIn("Q-to-answer", details.text or "")
        self.assertNotIn("2-hop", details.text or "")
        self.assertNotIn("alpha relation", details.text or "")

    def test_q_to_a_exact_depth_rejects_paths_through_other_answer_cuis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\t\t\n"
                "C_MID_A\tMid alpha\tT033\tFinding\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID_A,RO,alpha_bridge\n"
                "C_MID_A,C_A,RO,alpha_relation\n"
                "C_Q,C_A,RO,beta_dirty_bridge\n"
                "C_A,C_B,RO,beta_dirty_relation\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_B", "C_MID_A"})

            details = build_evidence_card_details(item, "q_to_a_exact_depth_2_evidence", metadata, relations)

        self.assertFalse(details.q_to_a_path_found)
        self.assertEqual(details.q_to_a_answer_cui_count, 2)
        self.assertEqual(details.q_to_a_answer_cui_covered_count, 1)
        self.assertNotIn("beta dirty relation", details.text or "")

    def test_q_to_a_depth_four_uses_degree_ordered_clean_path_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta"},
                "answer_idx": "A",
                "answer": "Alpha",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\t\t\n"
                "C_A1\tAlpha bridge one\tT033\tFinding\t\t\n"
                "C_A2\tAlpha bridge two\tT033\tFinding\t\t\n"
                "C_A3\tAlpha bridge three\tT033\tFinding\t\t\n"
                "C_B1\tBeta bridge one\tT033\tFinding\t\t\n"
                "C_B2\tBeta bridge two\tT033\tFinding\t\t\n"
                "C_B3\tBeta bridge three\tT033\tFinding\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_A1,RO,alpha_1\n"
                "C_A1,C_A2,RO,alpha_2\n"
                "C_A2,C_A3,RO,alpha_3\n"
                "C_A3,C_A,RO,alpha_4\n"
                "C_Q,C_B1,RO,beta_1\n"
                "C_B1,C_B2,RO,beta_2\n"
                "C_B2,C_B3,RO,beta_3\n"
                "C_B3,C_B,RO,beta_4\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(
                rel_path,
                keep_cuis={"C_Q", "C_A", "C_B", "C_A1", "C_A2", "C_A3", "C_B1", "C_B2", "C_B3"},
            )

            details = build_evidence_card_details(item, "q_to_a_exact_depth_4_evidence", metadata, relations)

        self.assertTrue(details.q_to_a_path_found)
        self.assertEqual(details.q_to_a_answer_cui_count, 2)
        self.assertEqual(details.q_to_a_answer_cui_covered_count, 2)
        self.assertEqual(
            details.q_to_a_path_selection_policy,
            "degree_ordered_clean_exact_hops_all_answer_cuis_max_2_paths_per_cui_budget_5000000",
        )
        self.assertIn("alpha 4", details.text or "")
        self.assertIn("beta 4", details.text or "")

    def test_q_to_a_exact_depth_caps_clean_paths_per_answer_cui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha"},
                "answer_idx": "A",
                "answer": "Alpha",
                "concept": ["C_Q"],
                "option0": ["C_A"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_Q\tQuestion concept\tT033\tFinding\t\t\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\t\t\n"
                "C_MID_1\tMid one\tT033\tFinding\t\t\n"
                "C_MID_2\tMid two\tT033\tFinding\t\t\n"
                "C_MID_3\tMid three\tT033\tFinding\t\t\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\n"
                "C_Q,C_MID_1,RO,path_one_a\n"
                "C_MID_1,C_A,RO,path_one_b\n"
                "C_Q,C_MID_2,RO,path_two_a\n"
                "C_MID_2,C_A,RO,path_two_b\n"
                "C_Q,C_MID_3,RO,path_three_a\n"
                "C_MID_3,C_A,RO,path_three_b\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_MID_1", "C_MID_2", "C_MID_3"})

            details = build_evidence_card_details(item, "q_to_a_exact_depth_2_evidence", metadata, relations)

        self.assertTrue(details.q_to_a_path_found)
        self.assertEqual(details.q_to_a_answer_cui_count, 1)
        self.assertEqual(details.q_to_a_answer_cui_covered_count, 1)
        self.assertEqual(
            details.q_to_a_path_selection_policy,
            "degree_ordered_clean_exact_hops_all_answer_cuis_max_2_paths_per_cui_budget_5000000",
        )
        self.assertEqual(details.q_to_a_path_edge_count, 4)
        self.assertIn("path one a", details.text or "")
        self.assertIn("path two a", details.text or "")
        self.assertNotIn("path three a", details.text or "")

    def test_distractor_contrast_evidence_is_gold_label_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "test.jsonl"
            meta_path = root / "cui_detail.tsv.zip"
            rel_path = root / "triplets.csv.zip"
            write_jsonl(data_path, [{
                "question": "Question",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "answer_idx": "B",
                "answer": "Beta",
                "concept": ["C_Q"],
                "option0": ["C_A"],
                "option1": ["C_B"],
                "option2": ["C_C"],
                "option3": ["C_D"],
            }])
            write_zip_text(
                meta_path,
                "cui_detail.tsv",
                "CUI\tSTR\tTUI\tSTY\tDEF_SAB\tDEF\n"
                "C_A\tAlpha\tT047\tDisease or Syndrome\tMSH\tAlpha definition.\n"
                "C_B\tBeta\tT047\tDisease or Syndrome\tMSH\tBeta definition.\n"
                "C_C\tGamma\tT047\tDisease or Syndrome\tMSH\tGamma definition.\n"
                "C_D\tDelta\tT047\tDisease or Syndrome\tMSH\tDelta definition.\n",
            )
            write_zip_text(
                rel_path,
                "triplets.csv",
                "cui_1,cui_2,re,rel_type\nC_Q,C_A,RO,related\nC_Q,C_B,RO,related\n",
            )
            item = load_metamap_records(data_path, dataset="medqa", split="test")[0]
            alt_gold = type(item)(
                item_id=item.item_id,
                dataset=item.dataset,
                split=item.split,
                linker=item.linker,
                question=item.question,
                options=item.options,
                gold="A",
                answer_text="Alpha",
                question_cuis=item.question_cuis,
                option_cuis=item.option_cuis,
            )
            metadata = CuiMetadataIndex.from_zip(meta_path)
            relations = RelationIndex.from_zip(rel_path, keep_cuis={"C_Q", "C_A", "C_B", "C_C", "C_D"})

            details = build_evidence_card_details(item, "distractor_contrast_evidence", metadata, relations)
            alt_details = build_evidence_card_details(alt_gold, "distractor_contrast_evidence", metadata, relations)

        self.assertTrue(details.option_evidence_symmetric)
        self.assertFalse(details.uses_gold_label_for_evidence)
        self.assertEqual(details.text, alt_details.text)

    def test_build_paper_tables_groups_three_core_experiments(self) -> None:
        rows = [
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "experiment": "evidence_availability",
                "condition": "full_committed_card",
                "metrics": {
                    "gold_answer_grounded": True,
                    "gold_answer_evidence_available": True,
                    "distractor_contrast_available": False,
                    "evidence_token_estimate": 12,
                },
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "experiment": "answer_grounding",
                "condition": "item",
                "metrics": {
                    "question_grounded": True,
                    "correct_answer_grounded": True,
                    "all_options_grounded": False,
                    "any_option_grounded": True,
                    "distractor_grounding_rate": 0.5,
                },
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "experiment": "retrieval_faithfulness",
                "condition": "q_to_a_2hop",
                "metrics": {
                    "gold_path_available": True,
                    "gold_min_path_length": 2,
                    "distractor_path_rate": 0.25,
                    "num_retrieved_nodes": 3,
                    "num_retrieved_edges": 2,
                    "decision_relevance_ratio_proxy": 1.0,
                },
            },
            {
                "item_id": "medQA:test:0",
                "dataset": "medQA",
                "split": "test",
                "linker": "metamap",
                "experiment": "retrieval_faithfulness",
                "condition": "random_same_size",
                "metrics": {
                    "gold_path_available": True,
                    "gold_min_path_length": 2,
                    "distractor_path_rate": 0.25,
                    "num_retrieved_nodes": 3,
                    "num_retrieved_edges": 0,
                    "decision_relevance_ratio_proxy": 0.25,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "diagnostics.jsonl"
            write_jsonl(input_path, rows)
            outputs = build_paper_tables([input_path], root / "paper_tables", bootstrap_iterations=25)

            with outputs.evidence_views.open() as fh:
                evidence = list(csv.DictReader(fh))
            with outputs.answer_grounding.open() as fh:
                answer = list(csv.DictReader(fh))
            with outputs.retrieval_faithfulness.open() as fh:
                retrieval = list(csv.DictReader(fh))
            summary = outputs.diagnostic_summary.read_text(encoding="utf-8")

        self.assertEqual(evidence[0]["condition"], "full_committed_card")
        self.assertEqual(answer[0]["n"], "1")
        by_condition = {row["condition"]: row for row in retrieval}
        self.assertEqual(by_condition["q_to_a_2hop"]["random_control_gap"], "0.750000")
        self.assertEqual(by_condition["random_same_size"]["random_control_gap"], "0.000000")
        self.assertIn("Experiment 1", summary)
        self.assertIn("Experiment 3", summary)


if __name__ == "__main__":
    unittest.main()
