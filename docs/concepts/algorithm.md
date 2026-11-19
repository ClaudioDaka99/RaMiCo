# The algorithm

RaMiCo explains a classifier $M$ trained on data $X$ with labels $y$. It needs only $M$'s predictions.

## From pairs to records

For every class $c$, $X_c$ holds the samples labelled $c$ and $X_{\neq c}$ the others. A
[mining strategy](strategies.md) chooses $s$ **factuals** in $X_c$ and, for each of them, $k$ **counterfactuals** in
$X_{\neq c}$ (`n_factuals` and `n_counterfactuals`).

For every pair and every feature in which the two samples differ, RaMiCo **switches** that feature alone: the factual
takes the counterfactual's value. When the prediction changes, $M(x') \neq M(x)$, the switch becomes an elementary
**record**: the feature, its value before and after, the class the model predicted before and after.

Each record also gets an **implausibility**: the anomaly score of the switched sample under an Isolation Forest
trained on the data (with the labels, as in the paper; the switched sample carries its new prediction), rescaled by
the lowest and highest scores of the training data. 0 is as typical as the most typical training sample, 1 as
anomalous as the most anomalous one.

## From records to rules

Records are grouped by feature, starting class, arrival class and strategy. Each group is a **rule**:

- for a **numerical** feature, the mean ± standard deviation of the values before and after the switch: the rule moves
  the feature from about $E[v] \pm \sigma[v]$ to about $E[v'] \pm \sigma[v']$;
- for a **categorical** feature, the most frequent value before and after, with the share of records holding it;
- its **support**, the share of all records in the group, and the mean ± standard deviation of their implausibility.

!!! note "The published numbers"
    The original implementation decided between mean and most frequent value from the values of each group, which on
    data with categorical features summarised the numerical features by their most frequent value too. RaMiCo decides
    from the column of each feature, so on those data its numbers differ slightly from the published ones.

## In code

```python
explainer = RaMiCo(model, strategy="ensemble", n_factuals=100, n_counterfactuals=50, random_state=0)
rules = explainer.fit(X_train, y_train, class_names={0: "no", 1: "yes"})

rules.to_frame()  # one row per rule
rules.records  # one row per record, with the factual's values
rules.filter(to_class=1)  # the rules that reach class 1
explainer.pairs(X_train, y_train)  # the pairs the strategy chooses
```

`X` may be a DataFrame or an array: non-numeric columns, and the columns named in `categorical_features`, are
categorical. Distances are Gower distances, which combine numerical and categorical features.
