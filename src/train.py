import argparse, joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from utils import load_data, clean_and_select, NUMERIC_FEATURES, TARGET_COL

def build_models():
    models = {
        'linreg': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'ridge':  Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
        'lasso':  Pipeline([('scaler', StandardScaler()), ('model', Lasso(max_iter=5000))]),
        'rf':     Pipeline([('model', RandomForestRegressor(random_state=42))])
    }
    if HAS_XGB:
        models['xgb'] = Pipeline([('model', XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8))])
    return models

def param_grids():
    grids = {
        'ridge': {'model__alpha':[0.1,1.0,3.0,10.0]},
        'lasso': {'model__alpha':[0.001,0.01,0.1,1.0]},
        'rf':    {'model__n_estimators':[200,400],
                  'model__max_depth':[None,10,20],
                  'model__min_samples_split':[2,5]}
    }
    # Linear regression has no main hyperparams
    if 'xgb' in build_models():
        grids['xgb'] = {'model__n_estimators':[200,400],
                        'model__max_depth':[4,6,8],
                        'model__learning_rate':[0.03,0.05,0.1]}
    return grids

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def main(args):
    df = load_data(args.data)
    df = clean_and_select(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = build_models()
    grids = param_grids()

    results = {}
    best_model = None
    best_score = -np.inf

    for name, pipe in models.items():
        if name in grids:
            gs = GridSearchCV(pipe, grids[name], scoring='r2', cv=5, n_jobs=-1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            model = pipe

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        results[name] = metrics

        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = model

        print(f"Model: {name} => {metrics}")

    if args.model_out:
        joblib.dump(best_model, args.model_out)
        print(f"Saved best model to {args.model_out} (R2={best_score:.3f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data (e.g., data/spotify_tracks.csv)')
    parser.add_argument('--model-out', type=str, default='models/final_model.pkl', help='Output path for the best model')
    args = parser.parse_args()
    main(args)
