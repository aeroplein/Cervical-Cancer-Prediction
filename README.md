# Cervical Cancer Risk Prediction

## Project Overview
This project predicts the risk of cervical cancer using a **VotingClassifier** combining
Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree models.
The model handles missing values, scales features, and balances the minority class. 

## Dataset
- Source: [Kaggle - Cervical Cancer Risk Factors for Biops](https://www.kaggle.com/datasets/abdenourbenacer/cervical-cancer-risk-factors-for-biops)
- Features include patient demographics, sexual and medical history.
- Target: 'Dx:Cancer' (0 = no cancer, 1 = cancer)

## Preprocessing
- Replace missing values ('?' or empty) with median values.
- Convert all features to numeric.
- Handle class imbalance with 'class_weight='balanced'' in classifiers.

## Model
- **Voting Classifier (hard voting)**:
  - Logistic Regression ('max_iter=1000, 'class_weight='balanced'')
  - KNN (default=5)
  - Decision Tree (`random_state=42`, `class_weight='balanced'`)
 
  ## Results
  - Accuracy: ~99.6%
  - Minority class (cancer) recall improved after scaling.
  - Example classification report:
| Class | Precision | Recall | F1-score | Support |
|-------|----------|--------|----------|--------|
| 0     | 1.00     | 1.00   | 1.00     | 252    |
| 1     | 0.86     | 1.00   | 0.92     | 6      |


  ## Usage
  bash
  pip install -r requirements.txt
  python CervicsCancer.py

  
