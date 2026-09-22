# Model Training Report

## 1. Price Sweetspot Regressor
- Test R2: 0.3506
- Test RMSE: $7.07
- Test MAE: $4.71
- Train size: 86,648 | Test size: 21,662

## 2. Review Score Regressor
- Test R2: 0.1449
- Test RMSE: 22.18%
- Test MAE: 16.54%
- Train size: 65,881 | Test size: 16,471

## 3. Value Score Regressor
- Test R2: 0.1638
- Test RMSE: 20.91
- Train size: 59,277 | Test size: 14,820

## 4. Ownership Regressor
- Test R2: 0.7775
- Test RMSE: 0.4531
- Train size: 91,873 | Test size: 22,969

## 5. Price Tier Classifier
- Test Accuracy: 0.8050
- Train size: 59,277 | Test size: 14,820

### Classification Report
```
              precision    recall  f1-score   support

      Budget       0.83      0.95      0.89     10999
   Mid-range       0.67      0.40      0.50      3420
     Premium       0.58      0.26      0.36       344
         AAA       0.50      0.28      0.36        57

    accuracy                           0.80     14820
   macro avg       0.64      0.47      0.53     14820
weighted avg       0.79      0.80      0.78     14820

```

### Confusion Matrix
```
           Budget  Mid-range  Premium  AAA
Budget      10465        517       13    4
Mid-range    2003       1360       46   11
Premium        99        155       89    1
AAA            30          5        6   16
```
