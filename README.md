
# 📌 Loan Approval Prediction System using Decision Tree

## 🏦 Project Overview
A banking institution seeks to automate the loan approval process to reduce manual errors and eliminate bias. Traditionally, loan officers evaluate applications using various factors such as income, credit history, and marital status. However, this manual process is not only time-consuming but also inconsistent.

The objective of this project is to develop a **Decision Tree-based Machine Learning model** that can predict loan approval outcomes efficiently, fairly, and transparently.

---

## 🔍 Problem Statement
> Develop a model that automates the loan approval decision based on historical applicant data. The model should be:
- Accurate
- Easy to interpret (transparent decision rules)
- Able to handle categorical and numerical variables

---

## 💡 Features Considered
The following features are used from the dataset:
- **Gender**
- **Married**
- **Dependents**
- **Education**
- **Self_Employed**
- **ApplicantIncome**
- **CoapplicantIncome**
- **LoanAmount**
- **Loan_Amount_Term**
- **Credit_History**
- **Property_Area**

---

## 🛠️ Technologies & Libraries
- Python
- Pandas & NumPy
- Seaborn & Matplotlib
- Scikit-learn
- Google Colab

---

## ⚙️ Steps in the Project
1. **Data Loading & Cleaning**
   - Handle missing values
   - Encode categorical features
   - Convert '3+' in dependents to numerical

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Correlation matrix
   - Categorical bar plots
   - Missing value heatmaps

3. **Model Training**
   - Train a `DecisionTreeClassifier`
   - Split data into training and test sets

4. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report
   - Feature importance visualization
   - Decision tree visualization

5. **Prediction**
   - Predict loan approval for a new applicant
   - Visual display of approval result

---

## 📈 Model Output

- **Accuracy**: ~ (Varies depending on dataset)
- **Confusion Matrix**: Displays correct/incorrect predictions
- **Feature Importance**: Shows which features impact the decision most
- **Visual Decision Tree**: Helps explain the model’s logic
- **Prediction Visualization**: A colored bar shows Approved/Rejected status

---

## 📦 How to Run the Project
1. Clone or download the repository
2. Open the notebook in [Google Colab](https://colab.research.google.com/)
3. Upload the CSV dataset when prompted (LoanApprovalPrediction.csv)
4. Run all cells
5. Modify `new_applicant` data to test your own cases

---

## ✅ Sample New Applicant Prediction

```python
new_applicant = pd.DataFrame([{
    'Gender': 1,
    'Married': 1,
    'Dependents': 1,
    'Education': 0,
    'Self_Employed': 0,
    'ApplicantIncome': 4000,
    'CoapplicantIncome': 1500,
    'LoanAmount': 128,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 2
}])
```

---

## 📁 Files Included
- `LOAN_APPROVAL_PREDICTION_PROJECT.ipynb` – Main project notebook
- `LoanApprovalPrediction.csv` – Dataset (upload manually if not present)
- `README.md` – Project overview and instructions

---

## 📚 References
- [Kaggle: Loan Prediction Dataset](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
