# Metrics

`ramico.metrics` scores a set of rules with the three metrics of the paper. They are usually computed on the data the
rules were extracted from.

```python
from ramico import metrics

scores = metrics.evaluate(model, rules, X_train)
scores.fidelity  # or scores.incomprehensibility, scores.implausibility, scores["fidelity"]
```

## Rule chains

For each sample $x$, RaMiCo takes the $k$ records (7 by default) whose factuals are closest to $x$, by Gower
distance, among the records that start from the class predicted for $x$. They form a **rule chain**: one condition
per feature and arrival class, set to the median of the switched values for a numerical feature and to their most
frequent value for a categorical one.

## Fidelity

The share of samples whose prediction the rules change:

$$
\text{fidelity} = \frac{|\{x \in X : M(x') \neq M(x)\}|}{|X|} \times 100,
$$

where $x'$ is $x$ with all the conditions of its chain applied at once (`mode="and"`, the paper's setting). With
`mode="or"`, a sample counts when at least one condition, applied on its own, brings the model to that condition's
arrival class. Higher is better.

## Incomprehensibility

The average number of conditions in the rule chains: how many features one must change, on average, to apply the
rules to a sample. Lower is better.

## Implausibility

The average, over the rules, of the mean implausibility of their records: how far from the training data the
switched samples are, from 0 (typical) to 1 (most anomalous). Lower is better.

## Rules from other explainers

The metrics also accept a DataFrame of records with the columns of `RuleSet.records`: `feature`, `to_value`,
`from_class`, `to_class`, `implausibility` and the factual's values in `factual.<feature>` columns. Rules produced by
another method can therefore be scored the same way.

## Comparing strategies

`ramico.benchmark.run_benchmark` repeats extraction and scoring over many random splits;
`compare_strategies` tests each strategy against a reference with a two-sided Mann-Whitney U test per metric and the
Benjamini-Hochberg correction, as in the paper.
