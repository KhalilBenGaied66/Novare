"""Consistency of the committed golden set with the committed corpus (data, not pipeline)."""

import json
import re
from collections import Counter

import pytest

from app.core.config import REPO_ROOT
from app.core.schemas import AskRequest

GOLDEN = json.loads((REPO_ROOT / "evals" / "golden_set.json").read_text(encoding="utf-8"))
CORPUS = {
    path.name for path in (REPO_ROOT / "data" / "sample_docs").iterdir() if path.is_file()
} - {"manifest.json"}
ROUTES = {"automation", "rag", "agent", "human"}
PII_TYPES = {"EMAIL", "TEL", "IBAN", "CARTE", "NIR", "SIRET"}
FIELDS = {
    "id",
    "split",
    "category",
    "tags",
    "input",
    "expected_route",
    "expected_final_route",
    "expected_docs",
    "forbidden_docs",
    "expected_facts",
    "forbidden_facts",
    "expected_pii",
    "note",
}


def test_ids_are_unique_and_every_case_has_the_same_fields():
    ids = [case["id"] for case in GOLDEN]
    assert len(ids) == len(set(ids))
    assert all(set(case) == FIELDS for case in GOLDEN)


def test_both_splits_cover_every_category():
    by_split = {
        split: Counter(case["category"] for case in GOLDEN if case["split"] == split)
        for split in ("dev", "test")
    }
    assert {case["split"] for case in GOLDEN} == {"dev", "test"}
    assert set(by_split["dev"]) == set(by_split["test"])
    assert len(GOLDEN) >= 100


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case["id"])
def test_case_is_well_formed(case):
    AskRequest(**case["input"])  # the API would accept the request
    assert case["expected_route"] in ROUTES and case["expected_final_route"] in ROUTES
    assert set(case["expected_docs"]) <= CORPUS
    assert set(case["forbidden_docs"]) <= CORPUS
    for pattern in case["expected_facts"] + case["forbidden_facts"]:
        re.compile(pattern)
    assert set(case["expected_pii"]) <= PII_TYPES
