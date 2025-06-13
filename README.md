# Smoking-Cesation

This project aims to analyze health-related data to identify patterns and build predictive models for smoking cessation. The dataset includes various demographic, physical, and biochemical health indicators, and the goal is to classify individuals based on their smoking status.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Clustering](#clustering)
8. [Modeling and Evaluation](#modeling-and-evaluation)
9. [Ensemble Learning](#ensemble-learning)
10. [Conclusion](#conclusion)

---

## Project Overview

This project involves:
- **Exploratory Data Analysis (EDA)** to understand the dataset and identify patterns.
- **Data Preprocessing** to clean and prepare the data for modeling.
- **Feature Engineering** to enhance the dataset with meaningful transformations.
- **Dimensionality Reduction** using PCA and LDA to reduce complexity while retaining variance.
- **Clustering** to explore potential groupings in the data.
- **Modeling** using KNN, Naive Bayes, and Decision Trees to classify smoking status.
- **Ensemble Learning** to combine the strengths of individual models for improved performance.

---

## Dataset Description

The dataset contains health-related features such as:
- **Demographics**: Age, height, weight, waist circumference.
- **Sensory Attributes**: Eyesight (left/right), hearing (left/right).
- **Blood Pressure**: Systolic and diastolic (relaxation).
- **Biochemical Indicators**: Cholesterol, triglycerides, HDL, LDL, fasting blood sugar, AST, ALT, serum creatinine.
- **Other Health Indicators**: Hemoglobin, urine protein, dental caries.

The target variable is `smoking`, which indicates whether an individual is a smoker or non-smoker.

---

## Exploratory Data Analysis (EDA)

### Key Findings:
1. **Smoking Status Distribution**: Visualized using a pie chart.
2. **Missing Data**: No missing values were found in the dataset.
3. **Outliers**: Identified in features like triglycerides, LDL, and waist circumference.
4. **Feature Statistics**: Summary statistics revealed distinct patterns in smokers vs. non-smokers:
   - Smokers tend to have higher triglycerides, LDL, and blood pressure.
   - Non-smokers generally exhibit healthier biochemical indicators.

### Visualizations:
- Histograms and violin plots were used to analyze feature distributions by smoking status.
- Correlation heatmaps helped identify relationships between features.

---

## Data Preprocessing

### Steps Taken:
1. **Outlier Capping**:
   - Domain-based capping for features with medically established ranges (e.g., height, blood pressure).
   - IQR-based capping for right-skewed features (e.g., triglycerides, LDL).
   - Range-based capping for limited-scale features (e.g., eyesight).
2. **Encoding Categorical Variables**:
   - One-hot encoding for multi-level categorical features like `urine protein`.
   - Binary features like `hearing` and `dental caries` were used directly.
3. **Scaling**:
   - StandardScaler was applied to normalize numerical features for machine learning models.

---

## Feature Engineering

### Feature Selection:
- Highly correlated features were dropped:
  - **Waist(cm)** was removed due to high correlation with weight.
  - **LDL** was dropped as it overlaps with cholesterol.
- Low-variance features were retained if they were categorical.

---

## Dimensionality Reduction

### Linear Discriminant Analysis (LDA):
- Reduced the dataset to a single dimension for binary classification.
- LDA showed some class separation but was insufficient for optimal classification.

### Principal Component Analysis (PCA):
- Retained 18 components, capturing ~92% of the variance.
- PCA provided a balance between dimensionality reduction and information retention.
- Scatter plots of the first two and three principal components revealed clusters corresponding to smoking status.

---

## Clustering

### Optimal Number of Clusters:
- **Elbow Method**: Suggested 3-4 clusters.
- **Silhouette Score**: Indicated 2 clusters as the most well-defined.
- **Davies-Bouldin Index**: Supported 2 clusters as optimal.

### Implementation:
- KMeans clustering with 2 clusters was applied to the PCA-transformed data.
- Clustering did not significantly improve classification performance but provided insights into potential groupings.

---

## Modeling and Evaluation

### Models Used:
1. **K-Nearest Neighbors (KNN)**:
   - Optimal `k=21` was selected using cross-validation.
   - PCA-only data performed slightly better than PCA + clustered data.
2. **Naive Bayes**:
   - PCA + clustered data outperformed PCA-only data, achieving higher recall and F1 score.
3. **Decision Tree**:
   - Hyperparameters were tuned using GridSearchCV.
   - PCA-only data provided better overall performance.

### Metrics Evaluated:
- **ROC-AUC**: Assessed the balance between true positives and false positives.
- **Precision, Recall, F1 Score**: Evaluated the trade-off between sensitivity and specificity.
- **Confusion Matrices**: Visualized model performance.

---

## Ensemble Learning

### Voting Classifiers:
1. **Soft Voting**:
   - Combined predictions from KNN, Naive Bayes, and Decision Tree.
   - Achieved higher recall and ROC-AUC compared to hard voting.
2. **Weighted Voting**:
   - Assigned weights to models based on individual performance.
   - Outperformed all other models with the highest ROC-AUC (0.8305) and F1 score (0.7464).

---

## Conclusion

### Key Insights:
- PCA effectively reduced dimensionality while retaining variance, enabling efficient modeling.
- Clustering provided exploratory insights but did not significantly enhance classification performance.
- Ensemble methods, particularly **Weighted Voting**, delivered the best overall performance.

### Final Model:
The **Weighted Voting Ensemble** is the final model, combining the strengths of KNN, Naive Bayes, and Decision Tree. It achieves a balance between precision and recall, making it suitable for deployment in smoking cessation prediction tasks.

---

## How to Run the Notebook

1. Clone the repository:
   git clone <repository-url>
   
2. Install the required dependencies:
   pip install -r requirements.txt

3. Run the notebook:
   jupyter notebook SmokingCesation.ipynb

---

## Future Work
Explore additional ensemble techniques like stacking or boosting.
Investigate the impact of feature interactions using advanced techniques like SHAP or LIME.
Incorporate external datasets to improve model generalization.
Deploy the final model as a web application or API for real-world use.
