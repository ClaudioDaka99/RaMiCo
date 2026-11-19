"""Tests of RuleSet and Rule: views, filters, display and files."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import _harness as h

from ramico import RaMiCo, Rule, RuleSet, _io, metrics
from ramico._ruleset import FACTUAL_PREFIX, RECORD_COLUMNS, RULE_COLUMNS, _chip, format_data_value, format_value
from ramico.exceptions import DataError
from ramico.viz import theme

NAN = float("nan")


@pytest.fixture(scope="module")
def data():
    return h.make_dataset(h.Case("ruleset", "mixed", 2, "random", n_samples=160))


@pytest.fixture(scope="module")
def rules(data):
    X, y = data
    explainer = RaMiCo(
        h.ToyModel(2), strategy="ensemble", n_factuals=5, n_counterfactuals=4, random_state=1, isolation_forest="native"
    )
    return explainer.fit(X, y, categorical_features=list(h.CATEGORICAL), class_names={0: "rejected", 1: "accepted"})


def make_rule(**changes) -> Rule:
    values = dict(
        strategy="boundary",
        from_class=0,
        to_class=1,
        feature="age",
        aggregation="mean",
        from_value=45.2,
        from_std=3.1,
        from_confidence=NAN,
        to_value=61.0,
        to_std=5.4,
        to_confidence=NAN,
        n_records=12,
        support=7.5,
        implausibility=0.25,
        implausibility_std=0.05,
    )
    values.update(changes)
    return Rule(**values)


class TestRule:
    def test_numerical_rule(self):
        rule = make_rule(from_name="rejected", to_name="accepted")
        assert rule.change == "45.20 ± 3.10 -> 61.00 ± 5.40"
        assert str(rule) == (
            "age: 45.20 ± 3.10 -> 61.00 ± 5.40  |  class rejected -> accepted  |  "
            "support 7.50% (12 records)  |  implausibility 0.250"
        )

    def test_categorical_rule(self):
        rule = make_rule(
            feature="color", aggregation="mode", from_value="red", from_std=NAN, from_confidence=80.0,
            to_value="blue", to_std=NAN, to_confidence=65.5,
        )  # fmt: skip
        assert rule.change == "red (80%) -> blue (65.5%)"
        assert make_rule(aggregation="mode", from_value=1.0, to_value=2.8, from_confidence=62.62).change.startswith(
            "1 (62.6%) -> 2.8"
        )  # data values as written in the data
        assert "class 0 -> 1" in str(rule)  # labels when there are no names

    def test_html_card(self):
        card = make_rule(feature="<age>")._repr_html_()
        assert "<code>&lt;age&gt;</code>" in card  # escaped
        assert "45.20 ± 3.10" in card and "0 → 1" in card and "12 records" in card
        assert "n/a" in make_rule(implausibility=NAN)._repr_html_()


@pytest.mark.parametrize(
    ("value", "text"),
    [(True, "True"), (np.bool_(False), "False"), (12345, "12,345"), (np.int64(5), "5"),
     (3.14159, "3.14"), (np.float32(2.5), "2.50"), ("red", "red")],
)  # fmt: skip
def test_format_value(value, text):
    assert format_value(value) == text


@pytest.mark.parametrize(
    ("value", "text"),
    [(1.0, "1"), (np.float64(77516.0), "77,516"), (2.8, "2.8"), (0.123456789, "0.123457"), (np.int64(5), "5"),
     ("red", "red")],
)  # fmt: skip
def test_format_data_value(value, text):
    assert format_data_value(value) == text


def test_implausibility_chip_uses_the_light_half_of_the_ramp():
    assert theme.SEQUENTIAL[0] in _chip(0.0) and theme.SEQUENTIAL[6] in _chip(1.0)
    assert theme.SEQUENTIAL[6] in _chip(7.0)  # clipped
    assert theme.INK["light"]["primary"] in _chip(0.9)


class TestRuleSet:
    def test_frame(self, rules):
        frame = rules.to_frame()
        assert list(frame.columns) == RULE_COLUMNS and len(frame) == len(rules) > 0
        assert frame["n_records"].dtype.kind == "i"
        assert frame["n_records"].sum() == len(rules.records)
        assert set(frame["strategy"]) == {"prototype", "nearest-boundary", "random-pairs"}
        assert set(frame["mined_class"]) <= set(rules.classes)

    def test_frame_order(self, rules):
        frame = rules.to_frame()
        position = {label: i for i, label in enumerate(rules.classes)}
        transitions = [(position[f], position[t]) for f, t in zip(frame["from_class"], frame["to_class"], strict=True)]
        assert transitions == sorted(transitions)
        for _, group in frame.groupby(["from_class", "to_class"]):
            assert group["support"].is_monotonic_decreasing

    def test_std_and_confidence_columns(self, rules):
        frame = rules.to_frame()
        means, modes = frame[frame["aggregation"] == "mean"], frame[frame["aggregation"] == "mode"]
        assert len(means) + len(modes) == len(frame)
        assert means["from_confidence"].isna().all() and means["to_confidence"].isna().all()
        assert modes["from_std"].isna().all() and modes["to_std"].isna().all()

    def test_records_of_a_rule(self, rules):
        for position in (0, len(rules) - 1):
            rule, behind = rules[position], rules.records_of(position)
            assert len(behind) == rule.n_records
            assert set(behind["feature"]) == {rule.feature} and set(behind["strategy"]) == {rule.strategy}
            assert set(behind["from_class"]) == {rule.from_class} and set(behind["to_class"]) == {rule.to_class}
        pd.testing.assert_frame_equal(rules.records_of(rules[0]), rules.records_of(0))

    def test_records(self, rules, data):
        X, _ = data
        records = rules.records
        assert list(records.columns) == RECORD_COLUMNS + [FACTUAL_PREFIX + c for c in X.columns]
        assert set(records["factual_index"]) <= set(rules.mined_indices) <= set(X.index)

    def test_sequence_protocol(self, rules):
        listed = list(rules)
        assert len(listed) == len(rules) and all(isinstance(rule, Rule) for rule in listed)
        # repr, not ==: the std of categorical rules is NaN, and NaN != NaN
        assert repr(rules[0]) == repr(listed[0]) and repr(rules[-1]) == repr(listed[-1])
        assert {rules[0].from_name, rules[0].to_name} == {"rejected", "accepted"}

    def test_metadata(self, rules):
        assert rules.strategy == "ensemble" and sorted(rules.classes) == [0, 1]
        assert rules.metadata["n_factuals"] == 5 and rules.metadata["isolation_forest"] == "native"
        assert rules.class_name(0) == "rejected" and rules.class_name(7) == "7"
        with pytest.raises(TypeError):
            rules.metadata["strategy"] = "other"  # read-only
        assert repr(rules).startswith(f"RuleSet(strategy='ensemble', rules={len(rules)}, records=")

    def test_pickle(self, rules):
        restored = pickle.loads(pickle.dumps(rules))
        pd.testing.assert_frame_equal(restored.to_frame(), rules.to_frame())
        pd.testing.assert_frame_equal(restored.records, rules.records)
        assert dict(restored.metadata) == dict(rules.metadata)
        with pytest.raises(TypeError):
            restored.metadata["strategy"] = "other"  # still read-only

    def test_filter(self, rules):
        source, target = rules.classes
        subset = rules.filter(from_class=source, to_class=target)
        frame = subset.to_frame()
        assert isinstance(subset, RuleSet) and 0 < len(subset) < len(rules)
        assert set(frame["from_class"]) == {source} and set(frame["to_class"]) == {target}
        assert set(subset.records["from_class"]) == {source}
        assert subset.metadata == rules.metadata

    def test_filter_with_lists(self, rules):
        features = ["age", "score"]
        frame = rules.filter(feature=features, strategy=["random-pairs"]).to_frame()
        assert set(frame["feature"]) <= set(features) and set(frame["strategy"]) <= {"random-pairs"}

    def test_filter_without_matches(self, rules):
        empty = rules.filter(feature="no such feature")
        assert len(empty) == 0 and empty.records.empty
        assert len(empty.filter(from_class=0)) == 0
        assert list(empty.to_frame().columns) == RULE_COLUMNS
        assert "No rules" in empty._repr_html_()

    def test_html(self, rules):
        page = rules._repr_html_()
        assert "RaMiCo rules" in page and "rejected → accepted" in page and "accepted → rejected" in page
        colours = theme.strategy_colors(["prototype", "nearest-boundary", "random-pairs"])
        assert all(colour in page for colour in colours.values())  # one strategy per component
        assert "more rules" in rules._repr_html_(max_rules=1)

    def test_single_strategy_html_has_no_strategy_column(self, data):
        X, y = data
        single = RaMiCo(
            h.ToyModel(2),
            strategy="random",
            n_factuals=4,
            n_counterfactuals=3,
            random_state=2,
            isolation_forest="native",
        ).fit(X, y, categorical_features=list(h.CATEGORICAL))
        assert len(single) > 0 and ">strategy</th>" not in single._repr_html_()


@pytest.fixture(scope="module")
def scores(rules, data):
    X, _ = data
    return metrics.evaluate(h.ToyModel(2), rules, X)


class TestFiles:
    @staticmethod
    def assert_same(restored: RuleSet, rules: RuleSet) -> None:
        pd.testing.assert_frame_equal(restored.to_frame(), rules.to_frame())
        pd.testing.assert_frame_equal(restored.records, rules.records)
        assert dict(restored.metadata) == dict(rules.metadata)
        assert restored._features == rules._features
        assert repr(restored) == repr(rules) and restored._repr_html_() == rules._repr_html_()
        assert [repr(rule) for rule in restored] == [repr(rule) for rule in rules]

    def test_round_trip(self, rules, data, scores, tmp_path):
        X, _ = data
        path = tmp_path / "rules.json"
        rules.save(path, scores=scores)
        restored = RuleSet.load(path)
        self.assert_same(restored, rules)
        assert restored.scores == scores and restored.scores.fidelity == scores.fidelity
        # the metrics of the restored rules are those of the original ones, to the last digit
        assert metrics.evaluate(h.ToyModel(2), restored, X) == scores
        assert len(restored.records_of(restored[0])) == restored[0].n_records

    def test_file_is_standard_json_with_the_public_columns(self, rules, tmp_path):
        path = tmp_path / "rules.json"
        rules.save(str(path))  # a str works as well as a Path

        def reject(constant):
            raise AssertionError(f"non-standard JSON constant {constant}")

        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        assert payload["format"] == "ramico.RuleSet" and payload["version"] == 1
        assert payload["rules"]["columns"] == RULE_COLUMNS
        assert payload["records"]["columns"] == list(rules.records.columns)
        assert payload["scores"] == {}
        rules.save(tmp_path / "again.json")
        assert (tmp_path / "again.json").read_bytes() == path.read_bytes()  # same rules, same file

    def test_compressed(self, rules, scores, tmp_path):
        rules.save(tmp_path / "rules.json.gz", scores=scores)
        rules.save(tmp_path / "rules.json", scores=scores)
        packed = (tmp_path / "rules.json.gz").read_bytes()
        assert packed[:2] == b"\x1f\x8b" and len(packed) < (tmp_path / "rules.json").stat().st_size / 3
        restored = RuleSet.load(tmp_path / "rules.json.gz")
        self.assert_same(restored, rules)
        assert dict(restored.scores) == scores

    def test_string_labels(self, data, tmp_path):
        X, y = data
        toy, names = h.ToyModel(2), np.array(["no", "yes"])

        def model(frame):
            return names[np.asarray(toy.predict(frame), dtype=int)]

        labelled = RaMiCo(
            model, strategy="random", n_factuals=4, n_counterfactuals=3, random_state=2, isolation_forest="native"
        ).fit(X, names[np.asarray(y)], categorical_features=list(h.CATEGORICAL), class_names={"no": "N", "yes": "Y"})
        labelled.save(tmp_path / "labelled.json")
        restored = RuleSet.load(tmp_path / "labelled.json")
        self.assert_same(restored, labelled)
        assert restored.class_name("yes") == "Y" and set(restored.to_frame()["from_class"]) <= {"no", "yes"}

    def test_datetime_index(self, data, tmp_path):
        X, y = data
        X = X.set_axis(pd.date_range("2020-01-01", periods=len(X), freq="D"))
        explainer = RaMiCo(
            h.ToyModel(2), "random", n_factuals=4, n_counterfactuals=3, random_state=2, isolation_forest="native"
        )
        dated = explainer.fit(X, np.asarray(y), categorical_features=list(h.CATEGORICAL))
        dated.save(tmp_path / "dated.json.gz")
        restored = RuleSet.load(tmp_path / "dated.json.gz")
        self.assert_same(restored, dated)
        assert restored.mined_indices and all(isinstance(i, pd.Timestamp) for i in restored.mined_indices)

    def test_file_saved_again_with_a_bom(self, rules, tmp_path):
        path = tmp_path / "rules.json"
        rules.save(path)
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())  # as some editors write UTF-8
        self.assert_same(RuleSet.load(path), rules)

    def test_empty(self, rules, tmp_path):
        empty = rules.filter(feature="no such feature")
        empty.save(tmp_path / "empty.json")
        restored = RuleSet.load(tmp_path / "empty.json")
        assert len(restored) == 0 and restored.records.empty
        assert list(restored.to_frame().columns) == RULE_COLUMNS
        assert list(restored.records.columns) == list(rules.records.columns)
        assert "No rules" in restored._repr_html_()

    def test_scores(self, rules, scores, tmp_path):
        assert dict(rules.scores) == {}
        rules.save(tmp_path / "rules.json", scores=scores)
        restored = RuleSet.load(tmp_path / "rules.json")
        with pytest.raises(TypeError):
            restored.scores["fidelity"] = 0.0  # read-only
        assert dict(pickle.loads(pickle.dumps(restored)).scores) == scores
        assert dict(restored.filter(from_class=0).scores) == {}  # they describe all the rules
        restored.save(tmp_path / "kept.json")  # without scores=: the ones it holds
        assert dict(RuleSet.load(tmp_path / "kept.json").scores) == scores
        rules.save(tmp_path / "odd.json", scores={"fidelity": NAN, "k": 7, "note": "mode or"})
        odd = RuleSet.load(tmp_path / "odd.json").scores
        assert np.isnan(odd["fidelity"]) and odd["k"] == 7 and odd["note"] == "mode or"

    def test_wrong_names(self, rules, tmp_path):
        for name in ("rules.csv", "rules.pkl", "rules.gz"):
            with pytest.raises(ValueError, match="to_csv"):
                rules.save(tmp_path / name)
            with pytest.raises(ValueError, match="json"):
                RuleSet.load(tmp_path / name)
        with pytest.raises(FileNotFoundError):
            RuleSet.load(tmp_path / "missing.json")

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b"not json", "is not a RuleSet file"),
            (b"\x1f\x8b\x08 damaged", "is not a RuleSet file"),
            (b'{"rules": []}', "written by RuleSet.save"),
            (b"[1, 2]", "written by RuleSet.save"),
            (b'{"format": "ramico.RuleSet", "version": 99, "ramico_version": "9.0"}', "RaMiCo 9.0 in a newer format"),
            (b'{"format": "ramico.RuleSet", "version": 1}', "incomplete or damaged"),
        ],
    )
    def test_bad_files(self, tmp_path, content, message):
        for name in ("bad.json", "bad.json.gz"):
            path = tmp_path / name
            path.write_bytes(content)
            with pytest.raises(DataError, match=message if name == "bad.json" else "is not a RuleSet file"):
                RuleSet.load(path)


class TestJsonValues:
    def test_values(self):
        value = {
            "classes": [1, 0],
            "names": {0: "rejected", 1: "accepted"},
            "pair": (1, "a", 2.5),
            "nested": {(0, 1): [np.int64(3), np.float32(0.5), np.bool_(True)]},
            "special": [NAN, float("inf"), -float("inf")],
            "dates": [pd.Timestamp("2026-01-01 10:30"), np.datetime64("2026-02-01"), pd.NaT],
            "__tag__": None,
        }
        decoded = _io.decode_value(json.loads(json.dumps(_io.encode_value(value), allow_nan=False)))
        assert decoded["classes"] == [1, 0] and decoded["names"] == {0: "rejected", 1: "accepted"}
        assert decoded["pair"] == (1, "a", 2.5) and decoded["nested"] == {(0, 1): [3, 0.5, True]}
        assert np.isnan(decoded["special"][0]) and decoded["special"][1:] == [float("inf"), -float("inf")]
        assert decoded["dates"] == [pd.Timestamp("2026-01-01 10:30"), pd.Timestamp("2026-02-01"), None]
        assert decoded["__tag__"] is None
        with pytest.raises(TypeError, match="complex"):
            _io.encode_value(complex(1, 2))

    def test_table(self):
        frame = pd.DataFrame(
            {
                "int": np.array([1, -2, 3], dtype=np.int32),
                "float": [0.1, NAN, float("inf")],
                "bool": [True, False, True],
                "mixed": pd.Series([1, "red", 2.5], dtype=object),
                "text": ["a", "b", "c"],
                "nullable": pd.array([1, None, 3], dtype="Int64"),
            }
        )
        table = json.loads(json.dumps(_io.encode_table(frame), allow_nan=False))
        restored = _io.decode_table(table)
        pd.testing.assert_frame_equal(restored, frame)
        assert [type(v) for v in restored["mixed"]] == [int, str, float]
        with pytest.raises(ValueError, match="repeated"):
            _io.encode_table(pd.DataFrame([[1, 2]], columns=["a", "a"]))


def test_large_counts_have_thousands_separators():
    assert "(1,234 records)" in str(make_rule(n_records=1234))
    assert "1,234 records" in make_rule(n_records=1234)._repr_html_()
