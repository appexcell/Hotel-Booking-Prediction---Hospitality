# Hotel Booking Demand Analysis & Cancellation Prediction

## 1. Overview
This project analyzes hotel booking demand and predicts cancellation behavior based on historical booking data. The dataset contains information about reservation details, customer types, room types, seasonal patterns, and more. The main target variable is `is_canceled`, which indicates whether a reservation was canceled.

The project includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Classification modeling using tree-based algorithms
- Model evaluation and comparison
- Recommendations for hotel management

---

## 2. Dataset
- **Source:** [Kaggle – Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Rows:** ~75,000
- **Columns:** 32 (after cleaning)
- **Target:** `is_canceled` (1 = canceled, 0 = not canceled)

---

## 3. Key Observations
- **Data imbalance:** ~27% of bookings were canceled, 73% fulfilled.
- **Seasonality:** Peak booking months are July, August, May; lowest are November, February.
- **Pricing impact:** ADR (Average Daily Rate) affects cancellation likelihood.
- **Customer type:** Families vs individuals have different cancellation patterns.
- **Hotel type:** Resort and City Hotels show different seasonal trends.

---

## 4. Data Preprocessing
- Removed duplicated rows.
- Filled missing values for `children`, `agent`, and `country`.
- Dropped uninformative or redundant columns:
  - `reservation_status`, `company`, `adr`, `booking_changes`, `days_in_waiting_list`, etc.
- Encoded categorical features with `OneHotEncoder`.
- Scaled numerical features using `RobustScaler`.
- Split data into `train` and `test` sets (80%-20%, stratified by `is_canceled`).

---

## 5. Models Tested
The following classification models were used (tree-based ensemble focus):

| Model            | Type                  |
|-----------------|----------------------|
| DecisionTree     | Single tree          |
| RandomForest     | Ensemble, bagging    |
| ExtraTrees       | Ensemble, bagging    |
| AdaBoost         | Boosting             |
| Bagging          | Ensemble, bagging    |
| XGBoost          | Boosting             |
| LightGBM         | Boosting             |
| CatBoost         | Boosting             |

> Linear models (Logistic Regression, RidgeClassifier) and SVM were avoided due to poor performance or incompatibility with categorical-heavy data.

**Pipeline:**  
- Preprocessing → Feature encoding & scaling → Model training  
- SMOTE/resampling was removed for simplicity.

---

## 6. Evaluation Metrics
- Accuracy
- F1-score (main metric due to class imbalance)
- Precision and Recall

**Results (summary):**
- Tree-based ensembles performed best.
- RandomForest, XGBoost, LightGBM, and CatBoost achieved F1 ≈ 0.70–0.71 for the minority class.
- Single DecisionTree had lower F1 (~0.61–0.62 for canceled bookings).

**Class imbalance:**  
- Canceled bookings are underrepresented (~27%). This can be further addressed in future work with resampling, weighting, or threshold tuning.

---

## 7. Insights & Recommendations
- **Predictive models**: RandomForest or boosting models provide good cancellation prediction without complex preprocessing.  
- **Practical hotel advice**:
  - Focus on early-warning indicators for likely cancellations.
  - Adjust pricing strategies (ADR) in high-cancellation months.
  - Monitor families and group bookings closely as their cancellation patterns differ.  
- Future work: investigate imbalance mitigation (SMOTE, class weighting) and threshold optimization for better recall on cancellations.

---

## 8. Installation & Requirements
```bash
pip install -r requirements.txt
