# Credit Card Fraud Detection

This project presents a machine learning approach to detecting fraudulent credit card transactions. Due to the extreme class imbalance in real-world financial datasets, standard classification techniques are often insufficient. In this project, we compare the performance of various classification models on both imbalanced and balanced datasets using techniques such as SMOTE and undersampling.

## Project Overview

With the increasing reliance on digital payments, financial fraud has become a critical challenge. This project aims to:

- **Automatically classify** transactions as fraudulent or legitimate.
- **Improve fraud detection rates** while maintaining acceptable levels of false positives.
- **Compare model performance** on imbalanced and balanced datasets to simulate real-world behavior.

## Dataset

We used the public **Credit Card Fraud Detection** dataset from Kaggle:

[Credit Card Fraud Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Instances**: 284,807 transactions  
- **Fraud cases**: Only 492 (≈0.17%)  
- **Features**: 30 anonymized features (V1–V28), plus `Amount`, `Time`, and the target variable `Class` (0: legitimate, 1: fraud)

## Preprocessing

- Checked for missing values (none found)
- Removed duplicate entries
- Applied **MinMaxScaler** to normalize features
- Addressed class imbalance using:
  - **SMOTE**: synthetic oversampling of the minority class
  - **Random undersampling**: reducing the size of the majority class

## Models Trained

We trained and evaluated the following models:

- Random Forest
- XGBoost
- K-Nearest Neighbors (K=3, 5, 7)
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes
- Logistic Regression
- Multilayer Perceptron (MLP)
- Keras Sequential Neural Network

Each model was trained and tested on:
1. The original (imbalanced) dataset  
2. The SMOTE-balanced dataset  
3. The undersampled dataset  

## Evaluation Metrics

Given the imbalanced nature of the problem, we focused on:

- **Precision** – How many predicted frauds were actually frauds
- **Recall** – How many actual frauds were successfully detected
- **F1-score** – Harmonic mean of Precision and Recall
- **Accuracy** – Included for reference but not prioritized due to dataset imbalance

## Results Summary
### Imbalanced Dataset

| #  | Model                | Accuracy   | Precision (1) | Recall (1) | F1-score (1) |
|----|----------------------|------------|----------------|------------|--------------|
| 1  | XGBoost              | 0.999507   | 0.946667       | 0.747368   | 0.835294     |
| 2  | Random Forest        | 0.999507   | 0.985507       | 0.715789   | 0.829268     |
| 3  | KNN (k=3)            | 0.999489   | 0.945946       | 0.736842   | 0.828402     |
| 4  | KNN (k=5)            | 0.999471   | 0.957746       | 0.715789   | 0.819277     |
| 5  | Keras                | 0.999418   | 0.878049       | 0.757895   | 0.813559     |
| 6  | SVM                  | 0.999401   | 0.867470       | 0.757895   | 0.808989     |
| 7  | KNN (k=7)            | 0.999418   | 0.942857       | 0.694737   | 0.800000     |
| 8  | Logistic Regression  | 0.999295   | 0.831325       | 0.726316   | 0.775281     |
| 9  | Decision Tree        | 0.999119   | 0.758621       | 0.694737   | 0.725275     |
| 10 | MLP                  | 0.999084   | 0.820896       | 0.578947   | 0.679012     |
| 11 | Naive Bayes          | 0.976985   | 0.056410       | 0.810526   | 0.105479     |

### SMOTE (balanced dataset)

| #  | Model                | Accuracy   | Precision (1) | Recall (1) | F1-score (1) |
|----|----------------------|------------|----------------|------------|--------------|
| 1  | Random Forest        | 0.999489   | 0.912500       | 0.768421   | 0.834286     |
| 2  | XGBoost              | 0.999172   | 0.735294       | 0.789474   | 0.761421     |
| 3  | KNN (k=3)            | 0.998802   | 0.609756       | 0.789474   | 0.688073     |
| 4  | KNN (k=5)            | 0.998432   | 0.520548       | 0.800000   | 0.630705     |
| 5  | KNN (k=7)            | 0.997868   | 0.426966       | 0.800000   | 0.556777     |
| 6  | Decision Tree        | 0.997269   | 0.340426       | 0.673684   | 0.452297     |
| 7  | MLP                  | 0.995489   | 0.244444       | 0.810526   | 0.375610     |
| 8  | SVM                  | 0.981585   | 0.072842       | 0.852632   | 0.134217     |
| 9  | Logistic Regression  | 0.975734   | 0.057320       | 0.873684   | 0.107583     |
| 10 | Naive Bayes          | 0.975223   | 0.053170       | 0.821053   | 0.099872     |
| 11 | Keras                | 0.964544   | 0.039846       | 0.873684   | 0.076217     |

### Undersampling (balanced dataset)

| #  | Model                | Accuracy   | Precision (1) | Recall (1) | F1-score (1) |
|----|----------------------|------------|--------------|------------|--------------|
| 1  | Keras                | 0.998960   | 0.663636     | 0.768421   | 0.712195     |
| 2  | Logistic Regression  | 0.997727   | 0.407609     | 0.789474   | 0.537634     |
| 3  | SVM                  | 0.996793   | 0.316456     | 0.789474   | 0.451807     |
| 4  | KNN (k=7)            | 0.987929   | 0.103495     | 0.810526   | 0.183552     |
| 5  | MLP                  | 0.984651   | 0.087234     | 0.863158   | 0.158454     |
| 6  | Random Forest        | 0.983999   | 0.084780     | 0.873684   | 0.154562     |
| 7  | KNN (k=5)            | 0.983664   | 0.078947     | 0.821053   | 0.144044     |
| 8  | XGBoost              | 0.972245   | 0.049878     | 0.863158   | 0.094307     |
| 9  | KNN (k=3)            | 0.972897   | 0.048780     | 0.821053   | 0.092090     |
| 10 | Naive Bayes          | 0.968244   | 0.041376     | 0.810526   | 0.078732     |
| 11 | Decision Tree        | 0.896504   | 0.014943     | 0.936842   | 0.029417     |


> ** Note:** Accuracy is not a meaningful metric in this context, as predicting all transactions as legitimate yields 99.8% accuracy but fails to detect fraud.

## Key Takeaways

- **SMOTE balancing** significantly improved fraud detection with reasonable precision.
- **Undersampling** increased recall but produced many false positives.
- Ensemble methods like **XGBoost** and **Random Forest** outperformed simpler models.

## Future Work

To further improve performance:

- Combine **SMOTE** with cleaning techniques (e.g., SMOTEENN, Tomek links)
- Tune classification **thresholds** to better balance precision and recall
- Apply **cost-sensitive learning** to penalize misclassifying fraud more heavily
- Use **ensemble stacking** or hybrid models for better robustness

## Authors

- [Sara Kardaš](https://github.com/skardas1)

- [Amila Kukić](https://github.com/amilakukic)

> This project was developed as part of the Artificial Intelligence course at the Faculty of Electrical Engineering, University of Sarajevo.


© 2025 Sara Kardaš & Amila Kukić

Faculty of Electrical Engineering

University of Sarajevo
