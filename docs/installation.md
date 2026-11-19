# Installation

RaMiCo supports Python 3.11, 3.12 and 3.13.

```bash
pip install ramico
```

The core depends only on numpy, pandas and gower. Optional extras add the rest:

| Extra | Adds | For |
|---|---|---|
| `ramico[viz]` | plotly, matplotlib, great_tables, kaleido | tables (`RuleSet.to_table`) and charts (`ramico.viz`); kaleido exports the PNG copies of the figures |
| `ramico[benchmark]` | scipy, tqdm | `ramico.benchmark` and its statistical tests |
| `ramico[plausibility]` | YDF | the Isolation Forest of the paper |
| `ramico[tutorials]` | the three above, scikit-learn, XGBoost, imbalanced-learn, ipykernel, Jinja2 | the notebooks in `examples/` |

Extras combine, for instance `pip install "ramico[viz,benchmark]"`. A missing extra raises
`ramico.exceptions.MissingDependencyError`, whose message names the extra to install.

## The Isolation Forest

Implausibility is the anomaly score of a switched sample under an Isolation Forest trained on the data.
`RaMiCo(isolation_forest=...)` chooses the engine:

- `"auto"` (default): YDF when it is installed, the native engine otherwise;
- `"ydf"`: YDF's `IsolationForestLearner`, the configuration of the paper; YDF scores vary slightly from run to run;
- `"native"`: RaMiCo's own implementation of the same algorithm, in numpy, deterministic for a given `random_state`.

YDF publishes Windows wheels only up to version 0.8.0 (Python ≤ 3.12), so the extra installs it only there. With
fewer than 256 training rows YDF 0.8.0 biases the scores, and RaMiCo warns about it; the native engine handles small
data correctly. On a large sample the two engines rank samples alike (Spearman correlation about 0.995), as much as
two YDF forests agree with each other.

## From the source

```bash
git clone https://github.com/ClaudioDaka99/RaMiCo
cd RaMiCo
pip install -e ".[dev]"
pre-commit install
```

See `CONTRIBUTING.md` for the tests and the tutorials.
