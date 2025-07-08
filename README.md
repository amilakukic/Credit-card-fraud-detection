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
## Imbalanced Dataset

| Model                  | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| XGBoost                | 0.95      | 0.75   | 0.84     |
| Random Forest          | 0.99      | 0.72   | 0.83     |
| KNN (k=3)              | 0.95      | 0.74   | 0.83     |
| KNN (k=5)              | 0.96      | 0.72   | 0.82     |
| KNN (k=7)              | 0.94      | 0.69   | 0.80     |
| Logistic Regression    | 0.83      | 0.73   | 0.78     |
| SVM                    | 0.87      | 0.76   | 0.81     |
| Keras Neural Network   | 0.88      | 0.76   | 0.81     |
| Decision Tree          | 0.76      | 0.69   | 0.73     |
| MLP (Neural Network)   | 0.82      | 0.58   | 0.68     |
| Naive Bayes (threshold=0.3) | 0.06 | 0.81   | 0.11     |


> ** Note:** Accuracy is not a meaningful metric in this context, as predicting all transactions as legitimate yields 99.8% accuracy but fails to detect fraud.

## Key Takeaways

- **SMOTE balancing** significantly improved fraud detection with reasonable precision.
- **Undersampling** increased recall but produced many false positives.
- Ensemble methods like **XGBoost** and **Random Forest** outperformed simpler models.
- Deep learning approaches (Keras) can offer good performance but require careful tuning.

## Future Work

To further improve performance:

- Combine **SMOTE** with cleaning techniques (e.g., SMOTEENN, Tomek links)
- Tune classification **thresholds** to better balance precision and recall
- Apply **cost-sensitive learning** to penalize misclassifying fraud more heavily
- Use **ensemble stacking** or hybrid models for better robustness

## Authors

- [Sara Kardaš](https://github.com/skardas1)

- [Amila Kukić](https://github.com/amilakukic)

> This project was developed as part of the Artificial Intelligence course at the University of Sarajevo, Faculty of Electrical Engineering.


© 2025 Sara Kardaš & Amila Kukić

Faculty of Electrical Engineering

University of Sarajevo
