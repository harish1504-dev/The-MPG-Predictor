"""Compare 4 algorithms, pick the best by CV error, refit on all data.
Run `python train.py` to regenerate model.joblib + metrics.json."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).parent
NUM = ["displacement", "horsepower", "weight", "acceleration", "model year"]
CAT = ["cylinders", "origin"]

def pipe(model):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("prep", prep), ("model", model)])

def train(save=False):
    df = pd.read_csv(HERE / "auto-mpg.csv")
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")  # '?' -> NaN
    X, y = df[NUM + CAT], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector Regressor": SVR(C=20, epsilon=0.2),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
    }
    cv, results = KFold(5, shuffle=True, random_state=42), []
    for name, m in models.items():
        p = pipe(m)
        cv_rmse = -cross_val_score(p, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error").mean()
        pred = p.fit(X_tr, y_tr).predict(X_te)
        results.append(dict(name=name, cv_rmse=round(float(cv_rmse), 3),
            rmse=round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
            mae=round(float(mean_absolute_error(y_te, pred)), 3), r2=round(float(r2_score(y_te, pred)), 4)))
    best = min(results, key=lambda r: r["cv_rmse"])["name"]
    final = pipe(models[best]).fit(X, y)
    meta = dict(best=best, results=results, mpg_sorted=sorted(y.round(1).tolist()),
        ranges={c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in NUM},
        n=len(df), n_missing_hp=int(df.horsepower.isna().sum()))
    if save:
        joblib.dump(final, HERE / "model.joblib"); json.dump(meta, open(HERE / "metrics.json", "w"))
    return final, meta

if __name__ == "__main__":
    _, m = train(save=True); print("Deployed:", m["best"])
