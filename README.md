
# Ames Housing Price Prediction — Machine Learning Models in R

This project applies a wide range of linear and nonlinear machine learning models to predict housing prices using the **Ames Housing Dataset**, which contains 80 explanatory variables describing residential properties in Ames, Iowa.
The goal is to compare multiple predictive models and identify the most accurate model for real-estate price estimation.

---

## Project Files

* **R_code_Housing_Prices.ipynb** — All preprocessing, feature engineering, model training, tuning, and evaluation (using R code inside a notebook environment).
* **Predicting Housing Prices Using the Ames Housing** — Full written report summarizing methodology and results.
* **Appendicitis.pdf** — Supplemental model plots, tuning graphs, and variable importance figures.

---

## Dataset Overview

* **Source:** Kaggle — Ames Housing Prices Dataset
* **Total Records:** 1,460
* **Predictors:** 80 total variables (numeric, categorical, and ordinal)
* **Target Variable:** `SalePrice`

The dataset covers:
Property characteristics, construction quality, remodeling history, land features, neighborhood information, and house condition.

---

## Preprocessing Steps

* Removal of variables with over 50% missing values
* Imputation of other missing values using **KNN (k=5)**
* Conversion of categorical variables to dummy variables
* Transformation of date variables into meaningful features:

  * `HouseAge`, `RemodelAge`, `GarageAge`
* Removal of near-zero variance predictors
* Removal of highly correlated predictors (cutoff = 0.90)
* Box-Cox transformation of the target variable to reduce skewness
* Final predictor count after cleaning: **110 variables**

---

## Resampling & Data Splitting

* 80/20 train–test split
* **10-fold cross-validation** applied during training
* Box-Cox transformation applied to the target to satisfy linearity and normality assumptions

---

## Models Evaluated

### **Linear Models**

* Ordinary Least Squares (OLS)
* Ridge Regression
* Lasso Regression
* Elastic Net
* Principal Component Regression (PCR)
* Partial Least Squares (PLS)

### **Nonlinear Models**

* Multivariate Adaptive Regression Splines (MARS)
* Support Vector Machines (SVM)
* K-Nearest Neighbors (KNN)
* Neural Networks (NN)

---

## Key Results

### **Top Linear Models**

* **Elastic Net** and **PLS** achieved the best performance among linear models.
* Elastic Net Test RMSE ≈ 0.1513
* PLS Test RMSE ≈ 0.1507

### **Top Nonlinear Models**

* **Neural Network (NN)** and **Support Vector Machine (SVM)** performed best.
* SVM: RMSE ≈ 0.1460
* NN: RMSE ≈ 0.1438 (**best overall model**)

### **Overall Best Model**

✔ **Neural Network (NN)**
Lowest RMSE and strongest generalization on the test dataset.

---

## Project Structure

```
📂 Ames Housing Price Prediction
│── R_code_Housing_Prices.ipynb
│── Predicting Housing Prices Using the Ames Housing.pdf
│── Appendicitis.pdf
│── README.md
│── requirements.txt
```

---

## Skills Used

* R
* caret package
* glmnet
* pls / pcr
* MARS (earth package)
* svmRadial
* neural network models (nnet)
* dplyr, tidyr, tidyverse
* Data Cleaning & Feature Engineering
* Cross-Validation and Resampling Methods
* Box-Cox Transformation
* Model Tuning and Hyperparameter Optimization
* Regression Model Evaluation (RMSE, R²)

---

## Future Improvements

* Ensemble modeling (stacking: NN + SVM + Elastic Net)
* Add feature selection using recursive feature elimination
* Try XGBoost or LightGBM (tree-based methods)
* Deploy the model via Shiny dashboard or API

---
