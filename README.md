# Rainfall Prediction Classifier

A binary classification model that predicts whether it will rain today in the Melbourne, Australia area, using historical weather observations available as of the previous day.

## Overview

- **Data**: ~7,500 daily weather records (2008–2017) for Melbourne, Melbourne Airport, and Watsonia, sourced from the Australian Bureau of Meteorology via [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
- **Problem framing**: The target and a same-day feature are shifted by one day so the model only uses information that would actually be available at prediction time (no data leakage from the day being predicted).
- **Features**: temperature, humidity, pressure, wind speed/direction, cloud cover, and an engineered `Season` feature derived from the date.
- **Pipeline**: `scikit-learn` `ColumnTransformer` (standard scaling for numeric features, one-hot encoding for categorical features) feeding into a classifier.
- **Models compared**:
  - Random Forest Classifier, tuned via `GridSearchCV` over tree depth, estimator count, and split thresholds
  - Logistic Regression, tuned via `GridSearchCV` over regularization type and class weighting
- **Validation**: Stratified 5-fold cross-validation to preserve class balance across folds.

## Results

The notebook reports, for each model: best cross-validated accuracy, held-out test accuracy, a full classification report (precision/recall/F1), and a confusion matrix. A feature-importance chart is included for the Random Forest model.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook rainfall_prediction.ipynb
```

The dataset is loaded directly from a public URL inside the notebook, so no manual download is needed.

## Files

- `rainfall_prediction.ipynb` — end-to-end notebook: data loading, cleaning, feature engineering, model training, evaluation, and comparison
- `requirements.txt` — Python dependencies

## Notes

This project was originally built as part of IBM's AI Engineering Professional Certificate (Coursera) and has since been cleaned up and restructured for portfolio use — course scaffolding, exercise prompts, and answer-key markdown have been removed, and the two models are organized side by side for direct comparison.
