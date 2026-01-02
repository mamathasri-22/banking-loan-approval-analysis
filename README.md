# Banking Loan Approval Analysis

## 📊 Project Overview
A comprehensive data science project for analyzing and predicting banking loan approvals using machine learning techniques. This project includes a complete data pipeline from exploration to deployment, with interactive Power BI dashboards for business insights.

## 🎯 Project Objectives
- Analyze loan approval patterns and key factors
- Build predictive models for loan approval decisions
- Identify risk factors and business insights
- Create interactive dashboards for stakeholders
- Deploy model for real-time predictions

## 📁 Project Structure

```
banking_loan_approval_analysis/
│
├── Data/
│   ├── Raw/                    # Original dataset
│   └── processed/              # Cleaned and processed data
│
├── Notebooks/                  # Jupyter notebooks for each phase
│   ├── Phase 1: Data Exploration & Cleaning
│   ├── Phase 2: Feature Engineering & Analysis
│   ├── Phase 3: Model Development
│   └── Phase 4: Deployment & Reporting
│
├── Dashboards/                 # Power BI dashboard files
├── Reports/                    # Generated reports and documentation
├── images/                     # Visualizations and charts
└── models/                     # Saved model files
```

## 🔄 Project Phases

### Phase 1: Data Exploration & Cleaning
**Notebook:** `DATA_EXPLORATION_&_CLEANING.ipynb`

**Objectives:**
- Load and inspect raw loan data
- Identify missing values and data quality issues
- Handle outliers and anomalies
- Perform initial statistical analysis
- Clean and prepare data for analysis

**Key Outputs:**
- `loan_data_cleaned.csv` - Cleaned dataset
- Data quality report
- Initial insights document

---

### Phase 2: Advanced Analysis & Feature Engineering
**Notebooks:**
- `DATA_EXPLORATION_&_CLEANING.ipynb` (continued)
- `Feature Importance.ipynb`

**Objectives:**
- Conduct exploratory data analysis (EDA)
- Create derived features for better predictions
- Analyze correlations between variables
- Perform statistical tests
- Identify key predictors for loan approval

**Key Outputs:**
- `ai_disagreements_report.csv` - AI vs manual comparison
- `ai_vs_manual_comparison.png` - Visual comparison
- `asset_analysis.png` - Asset distribution analysis
- `cibil_score_analysis.png` - Credit score patterns
- `correlation_heatmap.png` - Feature correlations
- `feature_importance.png` - Top predictive features
- `income_loan_analysis.png` - Income vs loan analysis

---

### Phase 3: Model Development & Validation
**Notebooks:**
- `The Predictive Model.ipynb`
- `Model Inference & Validation.ipynb`
- `Model Persistence.ipynb`

**Objectives:**
- Train multiple machine learning models
- Perform hyperparameter tuning
- Validate model performance
- Compare different algorithms
- Select best best-performing model
- Save model for deployment

**Key Outputs:**
- `loan_prediction_model.pkl` - Trained model
- `model_columns.pkl` - Feature columns
- `model_confusion_matrix.png` - Model performance
- `final_loan_data_with_predictions.csv` - Predictions on the dataset
- `summary_metrics.csv` - Model performance metrics
- `loan_audit_review.csv` - Audit trail

---

### Phase 4: Deployment & Business Reporting
**Notebooks:**
- `Phase 4 Model Deployment and Business Reporting.ipynb`
- `ADVANCED ANALYSIS & POWER BI PREPARATION.ipynb`
- `Loan Approval Business Insights Report.ipynb`

**Objectives:**
- Prepare data for Power BI integration
- Create interactive dashboards
- Generate business insights reports
- Document findings and recommendations
- Prepare deployment pipeline

**Key Outputs:**
- `loan_data_for_powerbi_final.csv` - Dashboard data
- `loan_data_powerbi.csv` - Processed data for BI
- `Loan_Analysis_Report.pdf` - Comprehensive analysis
- `Loan_Business_Insights_Report.pdf` - Business insights
- Power BI Dashboard (in Dashboards folder)

---

## 📈 Key Features Analyzed

1. **Applicant Information**
   - Income levels
   - Employment status
   - Education background
   - Number of dependents

2. **Credit Profile**
   - CIBIL score analysis
   - Credit history
   - Existing loans

3. **Loan Details**
   - Loan amount
   - Loan term
   - Property value
   - Loan-to-value ratio

4. **Risk Factors**
   - Debt-to-income ratio
   - Asset ownership
   - Residential status

## 🤖 Machine Learning Models

The project explores multiple algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost


**Best Model Performance:**
- Accuracy: 0.9988
- Precision: 0.9981
- Recall: 1.0000
- F1-Score: 0.9990

## 📊 Power BI Dashboard Features

Interactive dashboards include:
- Loan approval trends over time
- Demographic analysis
- Risk segmentation
- Geographic distribution
- Performance metrics
- What-if analysis tools

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Persistence:** Pickle
- **Business Intelligence:** Power BI
- **Notebook:** Jupyter

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/mamathasri-22/banking-loan-approval-analysis.git
cd banking-loan-approval-analysis
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Analysis
Navigate through notebooks in order:

1. **Phase 1:** Data Exploration & Cleaning
```bash
jupyter notebook "Notebooks/DATA_EXPLORATION_&_CLEANING.ipynb"
```

2. **Phase 2:** Feature Engineering
```bash
jupyter notebook "Notebooks/Feature Importance.ipynb"
```

3. **Phase 3:** Model Training
```bash
jupyter notebook "Notebooks/The Predictive Model.ipynb"
```

4. **Phase 4:** Deployment & Reporting
```bash
jupyter notebook "Notebooks/Phase 4 Model Deployment and Business Reporting.ipynb"
```

### Making Predictions

```python
import pickle
import pandas as pd

# Load the model
with open('models/loan_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature columns
with open('models/model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Make prediction
new_data = pd.DataFrame([your_data])
prediction = model.predict(new_data[model_columns])
```

## 📄 Key Insights

[Add your key findings here, for example:]
- CIBIL score is the strongest predictor of loan approval
- Income-to-loan ratio significantly impacts approval rates
- Property ownership increases approval probability by X%
- Optimal loan amount range for highest approval rates

## 🔮 Future Enhancements

- [ ] Real-time API for loan predictions
- [ ] Mobile application integration
- [ ] Automated model retraining pipeline
- [ ] Advanced explainability features (SHAP values)
- [ ] Integration with banking systems
- [ ] A/B testing framework

## 👥 Contributors

- **Mamatha Sri** - [GitHub Profile](https://github.com/mamathasri-22)

## 📧 Contact

For questions or collaboration:
- GitHub: [@mamathasri-22](https://github.com/mamathasri-22)
- Project Link: [https://github.com/mamathasri-22/banking-loan-approval-analysis](https://github.com/mamathasri-22/banking-loan-approval-analysis)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: https://www.kaggle.com/datasets

---

**Note:** This project is for educational and analytical purposes. All data has been anonymized and should not be used for actual loan approval decisions without proper validation and compliance checks.

## 📊 Project Status

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

*Last Updated: January 2026*
