import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_data(X):

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor


# =========================
# MAIN APP
# =========================
def main():
    st.title("🚀 Machine Learning App")

    # Sidebar
    data_source = st.sidebar.selectbox("Data Source", ["Example", "Upload"])

    # Load data
    if data_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Upload file", type=['csv'])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            st.stop()
    else:
        dataset_name = st.sidebar.selectbox("Dataset", ["titanic", "tips", "iris"])
        data = sns.load_dataset(dataset_name)

    # Show data
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Feature selection
    features = st.multiselect("Select feature columns", data.columns)
    target = st.selectbox("Select target column", data.columns)

    problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])

    # Slider BEFORE button (important)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    # Model selection
    if problem_type == "Regression":
        model_name = st.sidebar.selectbox("Model", [
            "Linear Regression", "Decision Tree", "Random Forest", "SVM"
        ])
    else:
        model_name = st.sidebar.selectbox("Model", [
            "Decision Tree", "Random Forest", "SVM"
        ])

    # Run button
    if st.button("Run Analysis"):

        if not features or not target:
            st.error("Please select features and target")
            return

        X = data[features]
        y = data[target]

        # Preprocessing pipeline
        preprocessor = preprocess_data(X)

        # Split first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Apply preprocessing
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # =========================
        # MODEL INITIALIZATION
        # =========================
        if model_name == "Linear Regression":
            model = LinearRegression()

        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor() if problem_type == "Regression" else DecisionTreeClassifier()

        elif model_name == "Random Forest":
            model = RandomForestRegressor() if problem_type == "Regression" else RandomForestClassifier()

        elif model_name == "SVM":
            model = SVR() if problem_type == "Regression" else SVC()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        st.subheader("📊 Results")

        # =========================
        # METRICS
        # =========================
        if problem_type == "Regression":

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("🔹 MSE:", round(mse, 4))
            st.write("🔹 MAE:", round(mae, 4))
            st.write("🔹 R2 Score:", round(r2, 4))

        else:
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            st.write("🔹 Accuracy:", round(acc, 4))
            st.write("🔹 Precision:", round(precision, 4))
            st.write("🔹 Recall:", round(recall, 4))
            st.write("🔹 F1 Score:", round(f1, 4))

        # =========================
        # SHOW SAMPLE PREDICTIONS
        # =========================
        results_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        st.subheader("🔍 Sample Predictions")
        st.dataframe(results_df.head())


if __name__ == "__main__":
    main()