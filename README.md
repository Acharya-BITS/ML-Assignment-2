# Breast Cancer Classification: 5 ML Models + Streamlit App

## a. Problem statement
Classify breast tumours as malignant or benign from 30 diagnostic features computed from
digitised images of fine-needle aspirate (FNA) samples. Five classification models are trained on
the same data, compared on six evaluation metrics, and deployed in a Streamlit app.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository
- **Instances:** 569 (min required: 500)
- **Features:** 30 numeric (mean, standard error and worst of 10 cell-nucleus measurements) (min required: 12)
- **Target:** `target` (0 = malignant, 1 = benign), binary
- **Split:** 80/20 stratified train/test, `random_state=42` (455 train / 114 test)
- **Preprocessing:** `StandardScaler` inside the pipeline for Logistic Regression, KNN and SVM
  (saved with the model, so the app needs no separate scaler)

## c. GitHub repository link
https://github.com/Acharya-BITS/ML-Assignment-2

Streamlit app: (https://ml-assignment-2-fnlmjz3ydjyvbprxcmbmww.streamlit.app/)

## d. Models used

| ML Model Name       | Accuracy |    AUC | Precision | Recall |     F1 |    MCC |
|:--------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression |   0.9825 | 0.9954 |    0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree       |   0.9123 | 0.9157 |    0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN                 |   0.9561 | 0.9788 |    0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes         |   0.9386 | 0.9878 |    0.9452 | 0.9583 | 0.9517 | 0.8676 |
| SVM                 |   0.9825 | 0.9950 |    0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name       | Observation about model performance |
|:--------------------|:------------------------------------|
| Logistic Regression | Best AUC (0.9954) and tied-best on every other metric. The classes are close to linearly separable after scaling, so a simple linear model suffices. |
| Decision Tree       | Weakest model on all metrics (accuracy 0.9123, MCC 0.8174). An unpruned tree overfits the training data and generalises worse. |
| KNN                 | Good (accuracy 0.9561) but behind the linear models. Depends on scaling and on the choice of k=5. |
| Naive Bayes         | Strong AUC (0.9878) but lower accuracy (0.9386). Its feature-independence assumption is violated here, since many features are highly correlated, which hurts the hard predictions more than the ranking. |
| SVM                 | Ties Logistic Regression on accuracy, precision, recall, F1 and MCC. AUC is marginally lower (0.9950 vs 0.9954). |

**Overall winner: Logistic Regression.** It ties SVM on all threshold-based metrics and has the
highest AUC, while being simpler and faster to train and easier to interpret.

## Repository structure
```
app.py            Streamlit app (loads saved models, no retraining)
train.py          Trains all 5 models, saves them to model/, writes test_data.csv
utils.py          Shared metric computation
requirements.txt
test_data.csv     Held-out test set (features + target) for the app upload
model/            Saved .joblib pipelines for all five classifiers
```

## Run locally
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```
