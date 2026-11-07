import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

def compute_metrics(model, X, y):
    pred, proba = model.predict(X), model.predict_proba(X)
    multi = len(np.unique(y)) > 2
    avg = "weighted" if multi else "binary"
    auc = (roc_auc_score(y, proba, multi_class="ovr", average="weighted")
           if multi else roc_auc_score(y, proba[:, 1]))
    return {"Accuracy": accuracy_score(y, pred), "AUC": auc,
            "Precision": precision_score(y, pred, average=avg, zero_division=0),
            "Recall": recall_score(y, pred, average=avg, zero_division=0),
            "F1": f1_score(y, pred, average=avg, zero_division=0),
            "MCC": matthews_corrcoef(y, pred)}
