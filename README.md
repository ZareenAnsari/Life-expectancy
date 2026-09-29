#  Life Expectancy Prediction

A machine learning regression project that predicts life expectancy from global health and socio-economic indicators. It covers exploratory data analysis, feature engineering, model training and evaluation, and an interactive app.

---

##  Problem Statement

Life expectancy differs widely between countries and over time. This project asks:

- Which health and socio-economic factors are most associated with life expectancy?
- How accurately can life expectancy be predicted from these indicators?

##  Dataset

- **Source:** Kaggle Life Expectancy dataset

##  Workflow

1. **Data cleaning:** handled missing values, removed duplicates, checked outliers.
2. **Exploratory data analysis:** distributions, correlations, and trends across year / development.
3. **Feature engineering:** Encoding categorical variables, scaling, removing highly correlated features.
4. **Modeling:** trained and compared Linear Regression, Random Forest, XGBoost.
5. **Evaluation:** compared models on a held-out test set 80/20 split using R², RMSE, and MAE.
6. **App:** built an interactive interface in `app.py` for making predictions.


##  Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** Scikit-learn , XGBoost
- **App:** Streamlit

##  Project Structure

```
Life-expectancy/
├── app.py                                  # Interactive app
├── notebooks_life expectencies dataset.ipynb   # EDA, modeling, evaluation
├── requirement.txt                         # Python dependencies
└── README.md
```

##  How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ZareenAnsari/Life-expectancy.git
cd Life-expectancy

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Run the app
streamlit run app.py            # [change if app.py is not Streamlit]
```

To explore the analysis, open the notebook in Jupyter or VS Code.


##  Future Improvements

- Hyperparameter tuning and cross-validation
- Feature importance analysis (e.g., SHAP)
- Time-aware train/test split by year
- Deploy the app online (e.g., Streamlit Community Cloud)

##  Author

**Zareen Ansari**
Computer Systems Engineering graduate | Data Analyst / Data Science
 Karachi, Pakistan
[LinkedIn](https://linkedin.com/in/zareensansari) · [GitHub](https://github.com/ZareenAnsari)
