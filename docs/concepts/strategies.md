# Mining strategies

A strategy decides which pairs of samples RaMiCo switches, hence which regions of the data the rules describe. The
paper proposes three strategies and an ensemble of them.

| Name | Class | Factuals in $X_c$ | Counterfactuals in $X_{\neq c}$ |
|---|---|---|---|
| `boundary` | `BoundaryStrategy` | $s$ drawn among the `n_candidates` samples closest to $X_{\neq c}$ (by default $\lfloor 1.5\,s \rfloor$) | the $k$ nearest |
| `prototype` | `PrototypeStrategy` | $s$ medoids of $X_c$ (k-medoids on the Gower distances) | $k$ medoids of $X_{\neq c}$, the same for every factual |
| `random` | `RandomStrategy` | $s$ drawn at random | the $k$ nearest |
| `ensemble` | `EnsembleStrategy` | prototype, nearest-boundary and random-pairs mining, each keeping its own rules | |

The ensemble, the reference configuration of the paper, combines two more components:

- **nearest-boundary** (`NearestBoundaryStrategy`): the $s$ samples of $X_c$ closest to $X_{\neq c}$, without drawing;
- **random-pairs** (`RandomPairsStrategy`): $s$ random factuals, each with $k$ random counterfactuals.

Components run one after the other, each on every class, and their rules are kept apart: the `strategy` column of
`RuleSet.to_frame()` names the component that found each rule.

What the rules describe follows from the pairs: boundary mining describes small changes near the decision boundary,
prototype mining the typical samples of each class, random mining an unbiased sample. Tutorial 02 draws the pairs of
each strategy and compares their rules.

## Choosing a strategy

```python
RaMiCo(model, strategy="boundary")  # by name
RaMiCo(model, strategy=BoundaryStrategy(n_candidates=300))  # an instance with its own settings
RaMiCo(model, strategy=EnsembleStrategy([PrototypeStrategy(), RandomStrategy()]))
```

## Writing one

A strategy subclasses `MiningStrategy` and implements `select(X_c, X_not_c, s, k, distance, rng)`, which returns a
list of `Pair(factual, counterfactuals)` holding row positions in `X_c` and `X_not_c`. `register_strategy(name,
factory)` makes it available by name.

Randomness must come from `rng`, so that `random_state` makes the strategy reproducible. Ties between equal
distances are broken by the order of the samples, so that results do not depend on the version of numpy.
