### Table 3. Dataset distributions (raw duplicates removed: 425)
| Dataset | Benign | Phishing | Total |
|---|---|---|---|
| PhiUSIIL (raw) | 134,850 | 100,945 | 235,795 |
| PhiUSIIL after de-duplication | 134,808 | 98,714 | 233,522 |
| PhiUSIIL training set | 97,061 | 71,074 | 168,135 |
| PhiUSIIL validation set | 10,785 | 7,897 | 18,682 |
| PhiUSIIL test set | 26,962 | 19,743 | 46,705 |

### Table 5. Results on PhiUSIIL
| Model | Features | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Multinomial Naive Bayes | FG1 | 86.31 | 93.43 | 72.72 | 81.79 | 90.87 |
| Logistic Regression | FG1 | 89.40 | 96.57 | 77.69 | 86.11 | 93.08 |
| Random Forest | FG1 | 89.59 | 97.99 | 76.94 | 86.20 | 93.04 |
| eXtreme Gradient Boosting | FG1 | 89.98 | 97.38 | 78.41 | 86.87 | 93.50 |
| Multinomial Naive Bayes | FG2 | 75.62 | 95.04 | 44.66 | 60.77 | 65.86 |
| Logistic Regression | FG2 | 82.67 | 91.32 | 65.20 | 76.08 | 85.01 |
| Random Forest | FG2 | 87.47 | 95.82 | 73.57 | 83.24 | 90.51 |
| eXtreme Gradient Boosting | FG2 | 87.99 | 95.88 | 74.82 | 84.05 | 91.15 |
| Multinomial Naive Bayes | FG3 | 86.20 | 93.42 | 72.46 | 81.61 | 89.55 |
| Logistic Regression | FG3 | 89.55 | 94.84 | 79.61 | 86.56 | 93.17 |
| Random Forest | FG3 | 90.21 | 98.07 | 78.39 | 87.13 | 93.18 |
| eXtreme Gradient Boosting | FG3 | 90.32 | 97.78 | 78.89 | 87.33 | 93.79 |
| Multinomial Naive Bayes | FG4 | 82.86 | 100.00 | 59.46 | 74.58 | 85.72 |
| Logistic Regression | FG4 | 84.76 | 96.34 | 66.46 | 78.66 | 86.29 |
| Random Forest | FG4 | 86.09 | 95.40 | 70.49 | 81.07 | 88.06 |
| eXtreme Gradient Boosting | FG4 | 86.16 | 95.68 | 70.44 | 81.14 | 88.36 |

### Table 7. FG2 with different deep models on PhiUSIIL
| Model | AUC | F1 (%) | Precision (%) | Recall (%) | Accuracy (%) | Training time (min) |
|---|---|---|---|---|---|---|
| RNN | 91.53 | 85.76 | 94.63 | 78.42 | 88.99 | 24.9 |
| RCNN | 92.85 | 86.46 | 93.74 | 80.23 | 89.38 | 14.0 |
| DNN | 90.76 | 83.67 | 96.17 | 74.05 | 87.79 | 6.2 |
| VDCNN | 92.74 | 84.24 | 99.63 | 72.97 | 88.46 | 43.5 |

### Table 8. Proposed model
| Setting | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | AUC (%) | Training time (s) | Test time (s) |
|---|---|---|---|---|---|---|---|
| PhiUSIIL_stratified | 89.74 | 96.78 | 78.32 | 86.58 | 92.94 | 1547 | 82.5 |
| PhiUSIIL_domain_disjoint | 88.66 | 92.62 | 76.36 | 83.70 | 91.48 | 1841 | 93.8 |

### Table 9 row (this work, URL string only)
| Study | Method | Inputs | Reported result (%) |
|---|---|---|---|
| Proposed model (this work) | Character-level CNN | URL string only | Accuracy 89.74; F1 86.58; AUC 92.94 |
