"""Check notebook citation and provenance contracts without provider calls."""

import ast
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import Mock

from pydantic import BaseModel, ValidationError


def notebook_definitions(prefix, names, namespace):
    """Load requested definitions or fixtures, not notebook demonstrations."""
    path = next(Path(__file__).parent.glob(f"{prefix}-*.ipynb"))
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        nodes: list[ast.stmt] = [
            node
            for node in ast.parse("".join(cell["source"])).body
            if (
                isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
            )
            or (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names
            )
        ]
        exec(  # noqa: S102 - Execute trusted notebook definitions and fixtures only.
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )


class NotebookContracts(unittest.TestCase):
    def setUp(self):
        self.ns: dict[str, Any] = {
            "BaseModel": BaseModel,
            "ValidationError": ValidationError,
            "Literal": Literal,
            "Any": Any,
            "re": re,
            "ReplySource": Literal["live", "simulated", "fallback"],
            "retrieve_or_abstain": lambda *a, **k: [{"doc_id": "refund-policy"}],
            "retrieve": lambda *a, **k: [{"doc_id": "refund-policy"}],
            "rag_messages": lambda *a: [],
            "dry_cited_answer": lambda *a: "",
            "style_only_messages": lambda *a: [],
            "dry_style_only": "",
        }
        for prefix, names in [
            ("03", {"CitedAnswer", "rag_answer"}),
            ("04", {"StyleOnlyAnswer", "style_only_answer"}),
            ("05", {"StrategyResult", "score_answer"}),
        ]:
            notebook_definitions(prefix, names, self.ns)

    def test_citation_candidates_are_not_silently_ignored(self):
        for extra in [
            "",
            " [INVENTED_POLICY]",
            " [invented_policy]",
            " [unknown]",
            " []",
        ]:
            for source in ["live", "simulated", "fallback"]:
                with self.subTest(extra=extra, source=source):
                    answer = "Policy [refund-policy]" + extra
                    self.ns["call_model"] = Mock(
                        return_value=SimpleNamespace(
                            text=json.dumps(
                                {
                                    "answer": answer,
                                    "source_ids": ["refund-policy"],
                                    "status": "answered",
                                    "response_source": "live",
                                }
                            ),
                            source=source,
                        )
                    )
                    result = self.ns["rag_answer"]("Policy?")
                    self.assertEqual(
                        result.status, "invalid_citations" if extra else "answered"
                    )
                    self.assertEqual(result.response_source, source)
                    score = self.ns["score_answer"](
                        {
                            "request_id": "test",
                            "question": "Policy?",
                            "expected_doc": "refund-policy",
                        },
                        "rag",
                        self.ns["StrategyResult"](
                            answer=answer, source_ids=["refund-policy"]
                        ),
                    )
                    self.assertEqual(score["citation_ids_valid"], not extra)
                    self.assertEqual(score["expected_source_cited"], not extra)

    def test_abstention_rejects_undeclared_citations(self):
        self.ns["call_model"] = Mock(
            return_value=SimpleNamespace(
                text=json.dumps(
                    {
                        "answer": "Unknown [INVENTED_POLICY]",
                        "source_ids": [],
                        "status": "abstained",
                    }
                ),
                source="fallback",
            )
        )
        self.assertEqual(self.ns["rag_answer"]("Policy?").status, "invalid_citations")

    def test_provenance_survives_parsing_and_failures(self):
        valid = json.dumps(
            {
                "answer": "Ask support",
                "predicted_tone": "concise",
                "needs_policy_evidence": True,
                "response_source": "live",
            }
        )
        for source in ["live", "simulated", "fallback"]:
            for text in [valid, "invalid JSON"]:
                with self.subTest(source=source, text=text):
                    self.ns["call_model"] = Mock(
                        return_value=SimpleNamespace(text=text, source=source)
                    )
                    self.assertEqual(
                        self.ns["style_only_answer"]("Policy?").response_source, source
                    )
                    self.assertEqual(
                        self.ns["rag_answer"]("Policy?").response_source, source
                    )

    def test_retrieval_abstention_does_not_call_a_model(self):
        self.ns["retrieve_or_abstain"] = lambda *a, **k: []
        self.ns["call_model"] = Mock(
            side_effect=AssertionError("Unexpected model call")
        )
        result = self.ns["rag_answer"]("Unknown?")
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.response_source, "not_called")
        self.ns["call_model"].assert_not_called()

    def test_id_consistency_is_not_sentence_coverage(self):
        request = {
            "request_id": "test",
            "question": "Policy?",
            "expected_doc": "refund-policy",
        }
        for parse_valid in (True, False):
            with self.subTest(parse_valid=parse_valid):
                result = self.ns["StrategyResult"](
                    answer="Refunds are available [refund-policy]. No fee applies.",
                    source_ids=["refund-policy"],
                    parse_valid=parse_valid,
                )
                score = self.ns["score_answer"](request, "rag", result)
                self.assertEqual(score["citation_ids_valid"], parse_valid)
                self.assertNotIn("citations_valid", score)

    def test_cited_examples_preserve_final_characters(self):
        self.ns["EXAMPLE_SOURCES"] = {"refund": "refund-policy"}
        notebook_definitions("05", {"cite_example"}, self.ns)
        for reply, expected in [
            ("Please contact support", "Please contact support [refund-policy]"),
            ("Please contact support.", "Please contact support [refund-policy]."),
            ("Need help?", "Need help [refund-policy]?"),
            ("Thanks!", "Thanks [refund-policy]!"),
        ]:
            with self.subTest(reply=reply):
                result = self.ns["cite_example"](
                    {"intent": "refund", "assistant": reply}
                )
                self.assertEqual(result["assistant"], expected)

    def test_report_fallback_distinguishes_missing_reviewer(self):
        for complete in (True, False):
            with self.subTest(live_comparison_complete=complete):
                model = Mock(
                    side_effect=lambda *args, **kwargs: SimpleNamespace(
                        text=kwargs["dry_response"], source="fallback"
                    )
                )
                namespace: dict[str, Any] = {
                    "json": json,
                    "BaseModel": BaseModel,
                    "Literal": Literal,
                    "ModelReply": SimpleNamespace,
                    "live_comparison_complete": complete,
                    "decision_messages": [],
                    "call_model": model,
                }
                notebook_definitions(
                    "05",
                    {"DecisionReport", "dry_decision", "decision_reply"},
                    namespace,
                )
                reply = namespace["decision_reply"]
                report = namespace["DecisionReport"].model_validate_json(reply.text)
                self.assertEqual(report.recommended_strategy, "not evaluated")
                if complete:
                    model.assert_called_once()
                    self.assertEqual(reply.source, "fallback")
                    self.assertIn("reviewer", report.rationale)
                    self.assertNotIn("not measured", report.rationale)
                    self.assertIn("manually", report.revisit_trigger)
                else:
                    model.assert_not_called()
                    self.assertEqual(reply.source, "simulated")
                    self.assertIn("not measured", report.rationale)
