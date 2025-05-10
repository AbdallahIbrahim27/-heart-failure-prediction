# 🧠 Heart Failure Risk Prediction

A Machine Learning Project Dedicated to Saving Lives

---

## 🎯 Project Objective

The primary goal of this project is to develop a robust machine learning model that predicts the risk of mortality due to heart failure. Special emphasis is placed on maximizing recall to ensure that high-risk patients are accurately identified and receive timely intervention.

---

## 📊 Dataset

- **Source:** [Heart Failure Clinical Records Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
- **Features:** Age, Ejection Fraction, Serum Creatinine, High Blood Pressure, Diabetes, and more.
- **Target Variable:** `DEATH_EVENT` (1 = Deceased, 0 = Survived)

---

## 🧪 Models Evaluated

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- **Random Forest** (Selected for deployment based on superior recall performance)

---

## 🔍 Focus on Recall

In the healthcare domain, minimizing false negatives is critical, as failing to identify high-risk patients can have fatal consequences. By prioritizing recall, the model aims to capture as many true positive (high-risk) cases as possible, even if it results in a higher number of false positives.

---

## 🛠 Technology Stack

- **Programming Language:** Python
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn
- **Deployment:** Streamlit (for interactive web application and user interface)
- **Version Control:** Git & GitHub

---

## 🚀 Get Started

- **Live Application:** [Try the App](https://lnkd.in/dxWHbKBp)
- **Source Code:** [GitHub Repository](https://lnkd.in/das8UdV3)

---

## 🧪 Models Trained & Evaluation

During the development of this project, several machine learning models were trained and evaluated to predict the risk of death from heart failure:

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    |   XX%    |   XX%     |  XX%   |   XX%    |
| Decision Tree          |   XX%    |   XX%     |  XX%   |   XX%    |
| K-Nearest Neighbors    |   XX%    |   XX%     |  XX%   |   XX%    |
| **Random Forest**      | **YY%**  | **YY%**   | **YY%**| **YY%**  |

> **Note:** The Random Forest model was selected for deployment due to its superior recall, which is critical in healthcare applications to minimize false negatives.

### 📈 Evaluation Metrics

- **Accuracy:** Overall correctness of the model.
- **Precision:** Proportion of positive identifications that were actually correct.
- **Recall (Sensitivity):** Proportion of actual positives that were identified correctly (prioritized in this project).
- **F1-Score:** Harmonic mean of precision and recall.

### 🔍 What is Recall?

Recall measures the proportion of actual positive cases (e.g., patients who are at risk of death from heart failure) that the model correctly identifies.

\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]

- **True Positives (TP):** Cases where the model correctly predicts a positive outcome (e.g., correctly identifies a high-risk patient).
- **False Negatives (FN):** Cases where the model fails to identify a positive outcome (e.g., misses a high-risk patient).

#### 🩺 Why is Recall Important in Healthcare?

- **Minimizing False Negatives:** In medical settings, failing to identify a high-risk patient (a false negative) can have severe or fatal consequences.
- **Prioritizing Patient Safety:** By maximizing recall, the model ensures that as many high-risk patients as possible are flagged for further attention, even if it means some false alarms (false positives).

#### 📊 Example

If your model has a recall of 0.90 (or 90%), it means it correctly identifies 90% of all actual high-risk patients.

#### 🐍 Python Example

```python
from sklearn.metrics import recall_score
# y_true: true labels, y_pred: predicted labels
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2f}")
```

### 🖼️ Infographics

Below are infographics generated to illustrate model performance and evaluation metrics:

- **Confusion Matrix (Random Forest):**  
  ![Confusion Matrix (Random Forest)]
  ![confusion_matrix_rf](https://github.com/user-attachments/assets/62583f9f-5b94-448e-96d4-a93047f9a406)


- **Recall Comparison Bar Chart:**  
  ![Recall Comparison Bar Chart]
  ![recall_comparison](https://github.com/user-attachments/assets/6f7cdb38-1eab-4b28-9ab3-448efdde33df)


- **ROC Curve (Random Forest):**  
  ![ROC Curve (Random Forest)]
  ![roc_curve_rf](https://github.com/user-attachments/assets/41c32a2b-fa39-4564-b9b5-23b0e96333ba)
