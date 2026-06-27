
# Evaluation Report — House Price Predictor

**Date:** 2026-06-27

## Final Model
**CatBoost_tuned**

## Test Set Performance (n=292 houses)

| Metric | Value | Plain English |
|--------|-------|---------------|
| RMSE   | $18,913 | Typical error, penalises big mistakes |
| MAE    | $13,594 | Average absolute prediction error |
| MedAE  | $9,143 | Median error, ignores outliers |
| MAPE   | 8.33% | Average % we are off by |
| R²     | 0.9352 | Model explains 93.5% of price variation |

## What the Numbers Mean
- On a $200,000 house, our predictions are typically off by
  around $13,594 ($6.8%)
- 50% of predictions are within $9,143
- Our worst single prediction was off by $82,819

## Residual Diagnostics
*(Fill in after reviewing plots)*
- Predicted vs Actual: [describe]
- Bias: [over or under predicts? at which price ranges?]
- Worst failures: [describe what the worst predictions have in common]

## Feature Importance
Top 5 most important features:
1. [feature] — [plain English explanation]
2. [feature]
3. [feature]
4. [feature]
5. [feature]

## Known Failure Modes
- [e.g. Model underestimates luxury properties > $400k]
- [e.g. Higher % error for unusual properties]

## Limitations
- Training data: Ames, Iowa residential sales 2006-2010
- May not generalise to other cities or time periods
- Does not account for market conditions or interest rates

## Recommendation
[State whether this model is good enough for the intended purpose]
