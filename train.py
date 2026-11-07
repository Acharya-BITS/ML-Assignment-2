import os, joblib, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utils import compute_metrics

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN":                 make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes":         GaussianNB(),
    "SVM":                 make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")
    rows.append({"Model": name, **compute_metrics(m, X_test, y_test)})

test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)

results = pd.DataFrame(rows).set_index("Model").round(4)
print(results.to_markdown())
