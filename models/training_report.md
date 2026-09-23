# Model Training Report

## 1. Price Sweetspot Regressor
- Test R2: 0.3505
- Test RMSE: $7.07
- Test MAE: $4.72
- Train size: 86,648 | Test size: 21,662

## 2. Review Score Regressor
- Test R2: 0.1446
- Test RMSE: 22.19%
- Test MAE: 16.54%
- Train size: 65,881 | Test size: 16,471

## 3. Value Score Regressor
- Test R2: 0.1638
- Test RMSE: 20.91
- Train size: 59,277 | Test size: 14,820

## 4. Ownership Regressor
- Test R2: 0.7776
- Test RMSE: 0.4529
- Train size: 91,873 | Test size: 22,969

## 5. Price Tier Classifier
- Test Accuracy: 0.8059
- Train size: 59,277 | Test size: 14,820

### Classification Report
```
              precision    recall  f1-score   support

      Budget       0.83      0.95      0.89     10999
   Mid-range       0.67      0.40      0.50      3420
     Premium       0.57      0.25      0.34       344
         AAA       0.46      0.23      0.31        57

    accuracy                           0.81     14820
   macro avg       0.63      0.46      0.51     14820
weighted avg       0.79      0.81      0.78     14820

```

### Confusion Matrix
```
           Budget  Mid-range  Premium  AAA
Budget      10465        519       12    3
Mid-range    1986       1380       44   10
Premium        97        160       85    2
AAA            30          5        9   13
```

## 6. Fair Price Classifier
- Test Accuracy: 0.9459
- Definition: Fair = (price <= genre median) OR (value_score >= genre median)
- Algorithm: DecisionTree(max_depth=8)
- Train size: 59,217 | Test size: 14,805

### Classification Report
```
              precision    recall  f1-score   support

  Overpriced       0.94      0.93      0.93      6148
        Fair       0.95      0.96      0.95      8657

    accuracy                           0.95     14805
   macro avg       0.94      0.94      0.94     14805
weighted avg       0.95      0.95      0.95     14805

```
