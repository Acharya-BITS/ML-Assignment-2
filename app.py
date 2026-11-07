import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import compute_metrics

MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title("Breast Cancer Classification: Model Evaluation")

file = st.file_uploader("Upload test CSV (features + 'target' column)", type="csv")
choice = st.selectbox("Select model", list(MODELS))

if file:
    df = pd.read_csv(file)
    if "target" not in df.columns:
        st.error("CSV must contain a 'target' column."); st.stop()
    y, X = df["target"], df.drop(columns="target")

    model = load_model(MODELS[choice])
    missing = set(model.feature_names_in_) - set(X.columns)
    if missing:
        st.error(f"Missing features: {sorted(missing)}"); st.stop()
    X = X[model.feature_names_in_]

    st.subheader(f"Metrics: {choice}")
    st.dataframe(pd.DataFrame([compute_metrics(model, X, y)]).round(4))

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, model.predict(X)), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification report")
    st.text(classification_report(y, model.predict(X)))

    st.subheader("Comparison of all models on this data")   # ✅ CHANGED
    st.dataframe(pd.DataFrame(                               # ✅ CHANGED: no expander, indent is 4 spaces
        [{"Model": n, **compute_metrics(load_model(p), X, y)} for n, p in MODELS.items()]
    ).set_index("Model").round(4))
else:
    st.info("Upload test_data.csv to begin.")
