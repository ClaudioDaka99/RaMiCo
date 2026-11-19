"""Golden tests: the implementation must reproduce the frozen outputs.

Only the adapter :func:`_harness.run_case` may change across refactorings;
the expected values in ``tests/fixtures/golden`` never do. The one deliberate
change of the method, the summaries of numerical features on mixed data,
is left out of the comparison: see ``_harness``.
"""

from __future__ import annotations

from typing import Any

import _harness as h
import pytest

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module", params=h.CASES, ids=lambda case: case.name)
def golden(request: pytest.FixtureRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    case = request.param
    return h.run_case(case), h.load_fixture(case)


def test_synthetic_dataset_is_unchanged(golden):
    actual, expected = golden
    assert actual["data_digest"] == expected["data_digest"], "the data generator drifted"


def test_mined_factuals(golden):
    actual, expected = golden
    assert actual["mined_indices"] == expected["mined_indices"]


def test_elementary_records(golden):
    actual, expected = golden
    h.assert_frame_matches(actual["records"], expected["records"], "records")


def test_aggregated_rules(golden):
    actual, expected = golden
    rules, expected_rules = actual["rules"], expected["rules"]
    if actual["dataset"] == "mixed":
        rules, expected_rules = h.without_numerical_summaries(rules, expected_rules, actual["numerical_features"])
    h.assert_frame_matches(rules, expected_rules, "rules")


def test_implausibility(golden):
    actual, expected = golden
    for key in ("implausibility", "implausibility_std"):
        h.assert_metrics_match({key: actual["metrics"][key]}, {key: expected["metrics"][key]}, "metrics")


def test_fidelity_and_incomprehensibility(golden):
    actual, expected = golden
    if actual["dataset"] == "mixed":
        pytest.skip("mixed data: the original combined numerical values by their mode, RaMiCo by their median")
    h.assert_metrics_match(actual["metrics"]["fidelity"], expected["metrics"]["fidelity"], "fidelity")


def test_isolation_forest_configuration(golden):
    actual, expected = golden
    assert actual["isolation_forest_params"] == expected["isolation_forest_params"]
    assert expected["isolation_forest_params"] == [{"num_trees": 1000, "max_depth": -1}]
