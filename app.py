
import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import plotly.express as px
from fpdf import FPDF

st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    layout="wide"
)

st.title("📈 Predictive Analytics Using Historical Data")

uploaded_file = st.file_uploader(
    "Upload Historical Dataset",
    type=["csv"]
)

if uploaded_file:

    # Load Data
    df = pd.read_csv(uploaded_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Feature Engineering
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    X = df[["Day", "Month", "Year"]]
    y = df["Sales"]

    split = int(len(df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # Linear Regression Model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Random Forest Model
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Model Evaluation
    st.header("Model Evaluation")

    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    col1, col2 = st.columns(2)

    col1.metric(
        "Linear Regression R²",
        f"{lr_r2:.4f}"
    )

    col2.metric(
        "Random Forest R²",
        f"{rf_r2:.4f}"
    )

    # Actual vs Predicted
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Linear Prediction": lr_pred,
        "Random Forest Prediction": rf_pred
    })

    fig = px.line(
        results,
        title="Actual vs Predicted"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Future Forecast
    st.header("Future Forecast")

    future_days = st.slider(
        "Forecast Days",
        min_value=7,
        max_value=90,
        value=30
    )

    last_date = df["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
    )

    future_df = pd.DataFrame({
        "Date": future_dates
    })

    future_df["Day"] = future_df["Date"].dt.day
    future_df["Month"] = future_df["Date"].dt.month
    future_df["Year"] = future_df["Date"].dt.year

    future_pred = rf.predict(
        future_df[["Day", "Month", "Year"]]
    )

    future_df["Forecast"] = future_pred

    fig2 = px.line(
        future_df,
        x="Date",
        y="Forecast",
        title="Future Sales Forecast"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Business Insights
    st.header("Business Insights")

    current_sales = df["Sales"].iloc[-1]
    average_sales = df["Sales"].mean()
    predicted_sales = future_df["Forecast"].iloc[0]

    growth_rate = (
        (predicted_sales - current_sales)
        / current_sales
    ) * 100

    accuracy = rf_r2

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Current Sales",
        f"{current_sales:,.0f}"
    )

    c2.metric(
        "Average Sales",
        f"{average_sales:,.0f}"
    )

    c3.metric(
        "Predicted Sales",
        f"{predicted_sales:,.0f}"
    )

    c4.metric(
        "Growth Rate",
        f"{growth_rate:.2f}%"
    )

    c5.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    # Download Excel
    future_df.to_excel(
        "forecast_results.xlsx",
        index=False
    )

    with open(
        "forecast_results.xlsx",
        "rb"
    ) as file:

        st.download_button(
            label="📥 Download Forecast Excel",
            data=file,
            file_name="forecast_results.xlsx"
        )

    # PDF Report
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=14
    )

    pdf.cell(
        200,
        10,
        txt="Predictive Analytics Report",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Random Forest Accuracy: {accuracy * 100:.2f}%",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Current Sales: {current_sales:,.0f}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Average Sales: {average_sales:,.0f}",
        ln=True
    )

    pdf.output(
        "forecast_report.pdf"
    )

    with open(
        "forecast_report.pdf",
        "rb"
    ) as file:

        st.download_button(
            label="📄 Download PDF Report",
            data=file,
            file_name="forecast_report.pdf"
        )

else:
    st.info("Please upload a CSV file containing Date and Sales columns.")

