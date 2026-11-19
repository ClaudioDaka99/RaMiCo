# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

First release as an installable package. The code was previously developed, as research code, under the name
*Bocsor*.

### Added

- `RaMiCo(model, strategy, n_factuals, n_counterfactuals, ...).fit(X, y)` returning a `RuleSet`: the rules as a
  DataFrame (`to_frame`), the records behind them (`records`, `records_of`), filters, a rich display in notebooks, a
  publication table (`to_table`) and a graph of all the rules (`plot`). `RaMiCo.pairs` returns the pairs a strategy
  chooses.
- `RuleSet.save` writes the rules, their records, how they were obtained and their metrics to one JSON file
  (`.json.gz` to compress it); `RuleSet.load` reads it back without fitting or evaluating again. A `RuleSet` can also
  be pickled.
- `ramico.strategies`: the boundary, prototype, random and ensemble strategies of the paper as classes with a common
  interface, and a registry for custom strategies.
- `ramico.metrics`: fidelity, incomprehensibility and implausibility, for a `RuleSet` or a DataFrame of records;
  `evaluate` returns them as a `Scores` mapping, read by name (`scores.fidelity`) or by key.
- `ramico.benchmark`: repeated extraction and scoring over seeds, with a cache, the Mann-Whitney U test and the
  Benjamini-Hochberg correction.
- `ramico.datasets`: Heart Disease and Adult, downloaded from UCI and checked by checksum.
- `ramico.viz`: one visual theme, tables (great_tables) and interactive charts (plotly): all the rules as a graph with
  sliders on support, size of the change and implausibility, the records behind a rule, the Pareto trade-off between
  the metrics and the sensitivity to s and k. `viz.show` can add a PNG copy to a figure, in the same output, for
  viewers that do not run JavaScript.
- A native Isolation Forest following YDF's algorithm, deterministic and without dependencies; YDF is used when
  installed (`ramico[plausibility]`).
- Two tutorials in `examples/` (a quickstart on Heart Disease, the mining strategies compared on Adult),
  documentation, golden tests of the original behaviour and unit tests.

### Changed

Compared with the original implementation:

- Numerical features are summarised by mean ± standard deviation, and combined by their median in the rule chains of
  the metrics, also when the data have categorical features. The original code used their most frequent value there,
  so on those data the numbers differ slightly from the published ones.
- Ties between equal distances, and the order of the initial medoids, are broken by the order of the samples, so that
  results no longer depend on the version of numpy or on the CPU.
- The strategies use the names of the paper: `margin` is now `boundary`, `multicenter` is `prototype`, `all_groups` is
  `ensemble`.
- The k-medoids implementation is included in the package instead of depending on scikit-learn-extra.
- Gower distances between two sets of samples of the same size are correct: the `gower` package (0.1.2) computed
  half of them as if the two sets were one, which misplaced the boundary of boundary mining on data with classes
  of equal size. The original code had the same error; the datasets of the paper are not affected.
- A numerical feature whose maximum is 0 (values such as -10 to 0) counts in the Gower distance: the `gower` package
  divides by the maximum and left such a feature out. The datasets of the paper have none.
- Distances and metrics are computed in blocks: the same values, to the last digit, in a fraction of the time on
  large datasets.

### Removed

- The strategies that do not appear in the paper (`density`, `center`, `svm`, `threshold`) and the experimental
  modules not used by the paper.

[0.1.0]: https://github.com/ClaudioDaka99/RaMiCo/releases/tag/v0.1.0
