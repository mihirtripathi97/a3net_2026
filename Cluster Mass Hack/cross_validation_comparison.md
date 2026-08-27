# Comparing the Two Cluster-Mass Approaches

This document compares the two approaches in `cluster_mass3_notebook.ipynb`:

1. The original approach without cross-validation.
2. The cluster-level 10-fold cross-validation approach.

## 1. Approach Without Cross-Validation

The first approach uses the original flags stored in the FITS metadata:

```python
train_ind = np.argwhere(hdul[2].data['train'] == 1)
val_ind = np.argwhere(hdul[2].data['validate'] == 1)
test_ind = np.argwhere(hdul[2].data['test'] == 1)
```

These indices create:

```text
train_X, train_Y
val_X,   val_Y
test_X,  test_Y
```

The model is trained once using the original training data:

```python
hist = model.fit(
    train_X,
    train_Y,
    batch_size=batch_size,
    epochs=epochs
)
```

In the current notebook, `val_X` and `val_Y` are not passed to `model.fit()`. The final evaluation is performed once on the original test set:

```python
prediction = model.predict(test_X, verbose=0).flatten()
```

This approach therefore consists of:

```text
One model
One fixed training set
One fixed test set
One final performance result
```

It answers the question:

> How well did this one model perform on this particular predefined test split?

## 2. Approach With Cluster-Level 10-Fold Cross-Validation

The second approach first identifies complete clusters. Each complete cluster has the expected lines of sight, such as `x`, `y`, and `z`.

The folds are created from unique cluster IDs:

```python
folds = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
```

The important point is that `KFold` splits `clusters`, not individual image rows:

```python
clusters = np.sort(complete_clusters)
folds.split(clusters)
```

For every fold:

```text
Approximately 90% of clusters -> training
Approximately 10% of clusters -> validation
```

All images belonging to a cluster are then expanded into the training or validation image arrays. Therefore, the three views of a cluster always stay in the same fold.

A new CNN is created and trained from scratch for each fold. The process consists of:

```text
Ten models
Ten training/validation arrangements
Ten validation results
One mean and standard deviation across folds
```

It answers the question:

> How consistently does the method perform when different groups of clusters are held out?

## 3. Main Differences

| Feature | Without cross-validation | With 10-fold cross-validation |
|---|---|---|
| Number of models | 1 | 10 |
| Data split | Original FITS flags | New cluster-level folds |
| Evaluation | One fixed test result | Ten validation results |
| Cluster separation guaranteed? | Not necessarily | Yes |
| Measures split-to-split variation? | No | Yes |
| Computational cost | Lower | Approximately ten times higher |

## 4. Important Fairness Issue

The current results are evaluated on different datasets:

```text
Without CV: performance on test_X
With CV:    performance on ten validation folds
```

These numbers should not be presented as if they were exactly the same type of measurement. The no-CV result is a single result on the original test set, while the CV result is an average over ten held-out cluster groups.

The no-CV result is a single-point estimate. The CV result includes both:

- The mean performance across folds.
- The variability of performance between folds.

## 5. Metrics for the Original Approach

The original approach can be summarized with:

```python
prediction = model.predict(test_X, verbose=0).flatten()

baseline_r2 = r2_score(test_Y, prediction)
baseline_mae = mean_absolute_error(test_Y, prediction)
baseline_rmse = np.sqrt(mean_squared_error(test_Y, prediction))
baseline_bias = np.mean(test_Y - prediction)

print('Without cross-validation')
print(f'R2:   {baseline_r2:.4f}')
print(f'MAE:  {baseline_mae:.4f}')
print(f'RMSE: {baseline_rmse:.4f}')
print(f'Bias: {baseline_bias:.4f}')
```

## 6. Metrics for the Cross-Validation Approach

The CV approach stores one row per fold in `cv_no_rot`:

```python
cv_no_rot[['r2', 'mae', 'rmse', 'scatter', 'bias']].agg(['mean', 'std'])
```

The `mean` row describes typical performance. The `std` row describes how much the result changes depending on which clusters are held out.

## 7. Interpreting the Metrics

### R2

`R2` measures how much of the variation in the true cluster masses is explained by the predictions.

- `1.0`: perfect predictions.
- `0.0`: no better than predicting the validation mean.
- Negative: worse than predicting the validation mean.

Higher is better.

### MAE

`MAE` is the mean absolute error:

```text
MAE = average(abs(true_mass - predicted_mass))
```

It describes the average prediction error in normalized log-mass units. Lower is better.

### RMSE

`RMSE` is the root mean squared error:

```text
RMSE = sqrt(average((true_mass - predicted_mass)^2))
```

It penalizes large errors more heavily than MAE. Lower is better.

### Scatter

The notebook calculates:

```python
scatter = np.std(val_Y - predictions)
```

This is the standard deviation of the residuals. It measures the random spread of the errors after accounting for their average offset. Lower is better.

### Bias

The notebook calculates:

```python
bias = np.mean(val_Y - predictions)
```

The sign is important:

- Positive bias: predictions are generally too low.
- Negative bias: predictions are generally too high.
- Bias near zero: little systematic over- or underprediction.

Bias should be close to zero.

## 8. Cluster Leakage Consideration

The original FITS split should be checked to determine whether different views of the same `cluster_id` appear in different original sets.

If one view of a cluster is in training and another view is in testing, the no-CV result may be optimistic. The model could learn cluster-specific information from one view and then be evaluated on another view of the same cluster.

The cluster-level CV approach prevents this problem because all views of a cluster are assigned together.

## 9. Recommended Presentation

A clear comparison table is:

| Approach | Evaluation | Mean or test R2 | MAE | RMSE |
|---|---|---:|---:|---:|
| Without CV | Original test set | baseline value | baseline value | baseline value |
| With 10-fold CV | Mean validation fold | mean value | mean value | mean value |

For the CV row, also report the standard deviation:

```text
R2   = mean +/- std
RMSE = mean +/- std
MAE  = mean +/- std
```

A suitable written conclusion is:

> The baseline model was trained using the original data split and evaluated once on the original test set. The second method used cluster-level 10-fold cross-validation, ensuring that all lines of sight from a given cluster remained in the same fold. The baseline gives one split-specific result, whereas cross-validation reports both average performance and variation across held-out cluster groups.

When comparing the methods, higher R2 is better, lower MAE and RMSE are better, and bias should be closer to zero. For cross-validation, a smaller standard deviation also indicates more consistent performance across different cluster groups.
