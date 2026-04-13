import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---- Page Config ----
st.set_page_config(page_title="Life Expectancy Predictor", layout="wide", page_icon="🌍")

# ---- Load Models & Data ----
lr_model    = joblib.load('linear_regression_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
rf_model    = joblib.load('random_forest_model.pkl')
xgb_model   = joblib.load('xgboost_model.pkl')
scaler      = joblib.load('scaler.pkl')
df          = pd.read_csv("df_cleaned.csv")

# ---- Title ----
st.title("🌍 Life Expectancy Prediction Dashboard")
st.markdown("Predict life expectancy using Machine Learning models trained on WHO data.")

# ---- Sidebar ----
st.sidebar.header("🤖 Select Model")
selected_model_name = st.sidebar.selectbox("Choose a Model", [
    "Linear Regression",
    "Ridge Regression",
    "Random Forest",
    "XGBoost ⭐ Best",
])

models = {
    "Linear Regression": lr_model,
    "Ridge Regression":  ridge_model,
    "Random Forest":     rf_model,
    "XGBoost ⭐ Best":   xgb_model,
}
selected_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.header("🔧 Input Features")

# ---- All 20 Input Features (matching your X columns) ----
year          = st.sidebar.slider("Year",                          2000, 2015, 2010)
adult_mort    = st.sidebar.slider("Adult Mortality",               0,    700,  150)
infant_deaths = st.sidebar.slider("Infant Deaths",                 0,    200,  10)
alcohol       = st.sidebar.slider("Alcohol",                       0.0,  20.0, 4.0)
pct_exp       = st.sidebar.slider("Percentage Expenditure",        0.0,  20000.0, 500.0)
hepatitis_b   = st.sidebar.slider("Hepatitis B (%)",               0.0,  100.0, 80.0)
measles       = st.sidebar.slider("Measles (cases)",               0,    100000, 100)
bmi           = st.sidebar.slider("BMI",                           0.0,  80.0,  38.0)
under_five    = st.sidebar.slider("Under-Five Deaths",             0,    200,   10)
polio         = st.sidebar.slider("Polio (%)",                     0.0,  100.0, 80.0)
expenditure   = st.sidebar.slider("Total Expenditure (%)",         0.0,  20.0,  6.0)
diphtheria    = st.sidebar.slider("Diphtheria (%)",                0.0,  100.0, 80.0)
hiv           = st.sidebar.slider("HIV/AIDS",                      0.0,  50.0,  0.1)
gdp           = st.sidebar.number_input("GDP",                     0.0,  100000.0, 5000.0)
population    = st.sidebar.number_input("Population",              0.0,  2e9,   1000000.0)
thinness_1_19 = st.sidebar.slider("Thinness 1-19 years (%)",       0.0,  30.0,  5.0)
thinness_5_9  = st.sidebar.slider("Thinness 5-9 years (%)",        0.0,  30.0,  5.0)
income_comp   = st.sidebar.slider("Income Composition",            0.0,  1.0,   0.6)
schooling     = st.sidebar.slider("Schooling (years)",             0.0,  20.0,  12.0)
status        = st.sidebar.selectbox("Country Status",             ["Developed", "Developing"])
status_encoded = 1 if status == "Developed" else 0

# Add country input in sidebar
country = st.sidebar.number_input("Country (encoded number)", 0, 200, 1)

# Then add it in correct position in input_data
if st.sidebar.button("🔮 Predict Life Expectancy"):
    input_data = np.array([[
        country, year, status_encoded, adult_mort, infant_deaths, alcohol,
        pct_exp, hepatitis_b, measles, bmi, under_five,
        polio, expenditure, diphtheria, hiv, gdp,
        population, thinness_1_19, thinness_5_9, income_comp, schooling
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = selected_model.predict(input_scaled)[0]

    st.sidebar.success(f"🎯 Predicted Life Expectancy: **{prediction:.2f} years**")
    st.sidebar.info(f"Model used: **{selected_model_name}**")

# ---- Dashboard Tabs ----
tab1, tab2, tab3 = st.tabs([" EDA", " World Map", " Model Comparison"])

with tab1:
    st.subheader("Life Expectancy Distribution")
    fig1 = px.histogram(df, x='Life expectancy ', color='Status',
                        template='plotly_dark', nbins=40,
                        title='Distribution of Life Expectancy')
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Schooling vs Life Expectancy")
        fig2 = px.scatter(df, x='Schooling', y='Life expectancy ',
                          color='Status', template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("GDP vs Life Expectancy")
        fig5 = px.scatter(df, x='GDP', y='Life expectancy ',
                          color='Status', template='plotly_dark', log_x=True)
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Life Expectancy by Status")
    fig6 = px.box(df, x='Status', y='Life expectancy ',
                  color='Status', template='plotly_dark')
    st.plotly_chart(fig6, use_container_width=True)

with tab2:
    st.subheader(" World Life Expectancy Map")
    fig3 = px.choropleth(df, locations='Country', locationmode='country names',
                         color='Life expectancy ', animation_frame='Year',
                         color_continuous_scale='RdYlGn', template='plotly_dark',
                         title='Life Expectancy by Country Over Time')
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader(" Model Performance Comparison")

    # ⚠️ Replace these with your actual scores from notebook
    model_df = pd.DataFrame({
        "Model":    ["Linear Regression", "Ridge Regression", "Random Forest", "XGBoost"],
        "R² Score": [0.70, 0.72, 0.85, 0.90],
        "MAE":      [2.30, 2.10, 1.56, 1.23],
        "RMSE":     [3.10, 2.90, 2.10, 1.80],
    })
    st.dataframe(model_df, use_container_width=True)

    fig4 = px.bar(model_df, x="Model", y="R² Score", text="R² Score",
                  color="Model", template="plotly_dark",
                  color_discrete_map={"XGBoost": "gold"},
                  title="🏆 XGBoost is the Best Model")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(yaxis=dict(range=[0, 1.1]), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)