# 🏆 **Amini Soil Nutrient Prediction Challenge – Winning Solution Documentation**

![1_bGeWjS4ucgG0B9gkMMi_FA](https://github.com/user-attachments/assets/b49044e1-dd9a-4100-87f2-c50e7aa866b0)


## 📌 Overview and Objectives

This solution was developed for a Zindi competition aimed at predicting soil nutrient levels (11 targets: N, P, K, Ca, Mg, S, Fe, Mn, Zn, Cu, B) from environmental and spatial features. The dataset included rich soil properties per location, and the objective was to minimize the RMSE between the predicted and actual nutrient values.

The key goals were:

* Build robust models per nutrient using both classic machine learning and ensemble techniques.
* Reduce overfitting and ensure generalizability through extensive cross-validation.
* Achieve reproducibility and high leaderboard performance using efficient and interpretable models.

---

## 📐 Architecture Diagram

```plaintext
         ┌─────────────┐        ┌─────────────┐
         │ Train CSVs  │──────▶│ Feature Eng  │
         └─────────────┘        └────┬────────┘
                                     ▼
                       ┌────────────────────────┐
                       │ Cross-validated Models │◀── Extensive FE (RF) and little Feature Engineering (RF) models
                       └──────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Multiple LGBM Variants   │
                     └──────────┬───────────────┘
                                ▼
                  ┌────────────────────────────────────┐
                  │ Stacking + Weighted Ensembling     │
                  └────────────┬───────────────────────┘
                               ▼
                          ┌──────────┐
                          │  Output  │
                          └──────────┘
```

---

## 🔄 ETL Process

### ✅ Extract:

* Data provided directly from the competition page (`train.csv`, `test.csv`)
* For random forest notebooks, a multiple output regressor was used at the end, and therefore nutrients were not separated into individual dataframes.
* For lightgbm notebooks, all nutrients were separated into individual dataframes per target (`N_df`, `P_df`, ...)

### ✅ Transform:

* Features like `PID`, `wp`, and target columns were dropped from input features.
* `log1p` transformation was used on targets (except B) to reduce skew in the third notebook (first lightgbm notebook).
* Categorical object columns were converted to `category` dtype.
* Two versions of random forest used:

  * **1:** Extensive feature engineering (e.g. interaction terms, ratios, discretizations)
  * **2:** Minimal feature engineering

### ✅ Load:

* Features were stored in pandas DataFrames.
* No external data was used. All data loading and preprocessing was done within the notebooks.

---

## 🧠 Data Modeling

### Algorithms Used:

* **LightGBM** with different configuration styles:

  * `precise`, `feature_selective`, `robust`, `deep_forest`, `highly_regularized`, `fast_learner`, `balanced`, `default`
* **Random Forest**:

  * Model 1: Feature-rich with engineered variables.
  * Model 2: Simpler version for contrast and diversity.

### Feature Engineering:

* Included variable encoding, log transformations, one-hot conversion, and numerical feature generation.
* All categorical variables were explicitly cast and passed into LightGBM.

### Model Training:

* Per-nutrient models were trained with 10-fold cross-validation for both lightgbm notebooks.
* The first lightgbm notebook did not utilize the bayesian optimization technique, it was only a 10 fold cross validation with all targets (except B) being scaled (np.log1p).
* Bayesian Optimization was used to stack multiple LGBM variants using optimal weight discovery per fold in the second lightgbm notebook.
* For random forest models, a multiple output regressor was used. This was trained on a simple train test split with a test size of 0.2.
* RMSE was used as the primary evaluation metric.

### Validation:

* Cross-validation RMSEs were stored per nutrient and summarized.
* Models stored per nutrient for both analysis and inference.

---

## 📤 Inference

### Deployment:

* Inference was done using ensemble of trained CV models.
* For LGBM stacks: predictions were weighted using Bayesian-optimized coefficients.
* Final predictions were aggregated using median across folds (to reduce outlier risk).

### Submission Strategy:

* Final `Gap` column generated from:

  * Weighted ensemble of multiple high-performing submissions:

    ```python
    lgbm_sub = (sub1 * 0.5) + (sub2 * 0.5)
    final_sub = (lgbm_sub * 0.65) + (sub3 * 0.35)
    ```
  * Tuned based on leaderboard experiments and validation behavior.

### Why these weights?

* **0.5/0.5** for sub1 and sub2: Both are strong LGBM variants with different preprocessing strategies.
* **0.65/0.35** in final ensemble: Gives more weight to stable base LGBMs while introducing additional diversity with sub3.

---

## 🕒 Runtime

| Notebook                        | Runtime Estimate |
| ------------------------------- | ---------------- |
| 1. RF (Extensive FE)            | \~5m 51s         |
| 2. RF (Simple)                  | \~2m 11s         |
| 3. LGBM with log1p targets      | \~7m 23s         |
| 4. LGBM with raw targets and Stacked Ensemble (LGBM only)| \~3h 10m 41s     |
| 6. Final Ensemble Submission    | \~22s            |

---

## 📊 Performance Metrics

| Target | CV RMSE (LGBM Stack) |
| ------ | -------------------- |
| N      | 470.06602            |
| P      | 37.64477             |
| K      | 186.19054            |
| Ca     | 1422.74364           |
| Mg     | 294.58220            |
| S      | 18.00171             |
| Fe     | 41.63802             |
| Mn     | 51.13168             |
| Zn     | 2.80881              |
| Cu     | 1.86514              |
| B      | 0.21136              |

**Leaderboard scores:**

* Public: **976.0499871**
* Private (Final): **🥇 1055.167929**

---

## ⚠️ Error Handling and Logging

* Early stopping is used in all model training to avoid overfitting.
* Evaluation metrics logged per fold and model.
* Cross-validation seeds fixed (`random_state=42`) for reproducibility.
* Errors in column types handled via `change_object_to_cat` utility function.

---

## 🔧 Maintenance and Monitoring

* Easy to retrain models using the `train_stacked_lgbm_cv()` function.
* Each fold saves models with weights for robust reusability.
* Updating data requires minimal refactoring.
* `cat_list` automatically tracks categorical columns—ensuring consistent handling.

---

## 💡 Insights and Notes

* Using `log1p` transformation on most targets significantly stabilized model training and improved leaderboard RMSE.
* The extensive feature engineering in RF provided valuable diversity for final ensembling.
* Even simple models (RF with default settings) contributed useful signals due to their structural variance.
* Median ensembling across folds outperformed mean for most nutrients due to better outlier resistance.
* Hundred heads are better than one. Ensembling is key in reducing the error.

---

## 📦 Environment and Libraries

* **Environment**: Kaggle notebooks (vCPU + 16GB RAM)
* **Python**: 3.10
* **Libraries**:

  ```txt
  pandas
  numpy
  matplotlib
  seaborn
  lightgbm
  scikit-learn
  bayesian-optimization
  ```

---

## 📁 Files Submitted

| File Name                                       | Description                       |
| ----------------------------------------------- | --------------------------------- |
| `1-amini-soil-random-forest-extensive-fe.ipynb` | RF with heavy feature engineering |
| `2-amini-soil-random-forest-little-fe`          | RF with little feature engineering|
| `3-amini-soil-lightgbm-logp1.ipynb`             | LGBM with log targets             |
| `4-amini-soil-lightgbm-no-logp1.ipynb`          | Stacking LGBM with raw targets    |
| `5-amini-soil-massive-ensembles`                | Weighted Ensembling               |
| `final_lgbm_rf_sub.csv`                         | Final blended submission file     |
| `requirements.txt`                              | Environment for reproducibility   |

---

## 🔚 Final Thoughts

This project showcases how stacking multiple strong base learners—paired with thoughtful feature engineering and careful ensembling—can outperform complex deep learning pipelines in structured data competitions. The modeling strategy emphasizes **simplicity**, **robustness**, and **repeatability**, leading to a solution that not only won the competition but is production-ready.

---
