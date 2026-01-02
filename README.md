# Customer Churn Prediction & Retention Strategy

## 📌 Project Overview
ML-проект по прогнозированию оттока клиентов и разработке стратегии удержания
на основе вероятности churn.

Проект выполнен как end-to-end ML pipeline:
от EDA и обучения модели до бизнес-интерпретации и применения результатов.

---

## 🎯 Business Goal
Раннее выявление клиентов с высоким риском ухода (churn)  
и применение целевых retention-мер для снижения оттока.

---

## 📊 Dataset
Customer Churn Dataset (Kaggle)

**Target:**
- `Churn`:  
  - `0` — клиент остался  
  - `1` — клиент ушёл

**Основные признаки:**
- Customer behavior (Payment Delay, Usage Frequency)
- Customer experience (Support Calls)
- Contract & subscription features
- Demographics

---

## 🧠 Solution Approach

### 1️⃣ Exploratory Data Analysis (EDA)
- Анализ распределения churn
- Проверка баланса классов
- Анализ числовых и категориальных признаков

### 2️⃣ Feature Engineering & Preprocessing
- One-Hot Encoding для категориальных признаков
- Единый `ColumnTransformer`
- ML Pipeline для воспроизводимости

### 3️⃣ Modeling
Используемые модели:
- Logistic Regression (baseline, интерпретируемость)
- Random Forest (финальная модель)

### 4️⃣ Model Evaluation
- ROC-AUC ≈ **0.90**
- Confusion Matrix
- Precision / Recall / F1
- Подбор threshold под бизнес-цель (recall churn)

### 5️⃣ Feature Importance
Наиболее значимые факторы churn:
- Payment Delay
- Support Calls
- Tenure
- Usage Frequency

Факторы агрегированы в бизнес-группы:
- Payment Behavior
- Customer Experience
- Engagement
- Contract
- Demographics

### 6️⃣ Risk Segmentation
Клиенты сегментированы по вероятности churn:
- **Low Risk**
- **Medium Risk**
- **High Risk**

Фактический churn по сегментам:
- High Risk → ~99%
- Medium Risk → ~32%
- Low Risk → ~0%

### 7️⃣ Retention Strategy
Для каждого сегмента предложены действия:
- Low Risk — no action
- Medium Risk — promo / email / discount
- High Risk — персональный контакт, удерживающее предложение

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook
- ML Pipelines
- Random Forest, Logistic Regression

---

## 🚀 Project Structure

ml-churn-prediction/
├── data/
│ └── raw/customer_churn_dataset-testing-master.csv
├── notebooks/
│ └── 01_eda.ipynb
├── src/
│ ├── train.py
│ └── predict.py
├── models/
│ └── churn_model.pkl
├── requirements.txt
└── README.md

---

## ▶ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py

## 📈 Output

Модель возвращает:
- вероятность churn для клиента
- сегмент риска (Low / Medium / High)

## 🔍 Key Insights:
- Задержка платежей — главный индикатор churn
- Частые обращения в поддержку усиливают риск ухода
- Churn определяется комбинацией факторов, а не одним признаком
- Модель подходит для реального retention-пайплайна