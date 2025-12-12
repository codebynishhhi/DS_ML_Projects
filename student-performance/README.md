## Student Performance Prediction — Summary

Goal: Predict students' math scores from demographic and test features.

Dataset: "Students Performance in Exams" (1,000 rows). Key features: gender, race/ethnicity, parental level of education, lunch, test preparation course, reading score, writing score.

Key result: A linear regression baseline achieves **RMSE = 8.79** and **R² = 0.683**, meaning the model explains ~68% of score variability and is off by about ±8.8 points on average.

## Model Results & Interpretation

**Metrics (Test set)**  
- Linear Regression: MAE = X.XX, RMSE = 8.79, R² = 0.683  
- Random Forest: MAE = Y.YY, RMSE = 10.30, R² = 0.564

**What this means:**
- RMSE = 8.79 → predictions are typically within ±8.8 points of the true math score. For example, a student with true score 72 is likely predicted between ~63 and ~81.
- R² = 0.683 → the model explains ~68% of variance; remaining variance may be due to unobserved factors (attendance, coaching, motivation) or noise.
- Linear Regression performed better than Random Forest in this split, suggesting the primary relationships are fairly linear and captured well by regression. (We must also ensure no target leakage — e.g., features that include math score.)

## Business implication

An RMSE of ~9 points means the model can give a rough but useful estimate of student performance. This can be used to flag students likely to underperform (e.g., predicted score < 40) for targeted interventions. However, for high-stakes decisions (scholarships, final placement), a lower error and more features would be required.


Next experiments:
1. Remove any features that leak target information (if present).
2. Add interaction features (e.g., reading * writing) and categorical encodings.
3. Try Gradient Boosting (XGBoost/LightGBM) and small hyperparameter tuning.
4. Use cross-validation to get robust error estimates.
5. Deploy the best pipeline and add explainability (SHAP) to show per-student drivers.

## Inference 1 - 
1. Baseline linear regression gives an RMSE of ~8.8, which means predictions are off by roughly 9 points on a 0–100 scale — a useful starting point. 
2. R² is 0.68, so the model explains most of the predictable variance. 
3. The Random Forest performed worse on the hold-out split, suggesting the relationships are fairly linear or that we need to tune tree hyperparameters. 
Next I’d remove any leakage, add engineered features (interactions / group means), and try gradient boosting with CV to improve the RMSE.

# Inferences 2 -
1. No target leak as the features used as input to train the model do not have any output target computed value.
2. The prdictions are not identical - false 
3. The difference in predictions is (LR - RF): 0.44710317195183036 - small but nonzero.
4. MAE Errors in Random forest is higher than in Linear Regression.
5. R2 score is better in Linear Regression, model covers 68% of variance.
6. Feature effects: reading is dominant (RF importance ≈ 0.69; LR coef ≈ 0.576). Writing contributes less but meaningfully.
7. The relationship between (reading, writing) and math is strongly linear and mostly captured by LR. 
8. RF is underperforming here either because (a) there is little nonlinearity to learn, or (b) RF is under-tuned / overfitting without enough features. 
9. With only two highly-correlated academic scores, LR’s simplicity wins.