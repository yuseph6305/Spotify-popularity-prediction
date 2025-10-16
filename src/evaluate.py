import argparse, joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_data, clean_and_select, TARGET_COL

def main(args):
    df = load_data(args.data)
    df = clean_and_select(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = joblib.load(args.model)
    y_pred = model.predict(X)

    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print({'RMSE': rmse, 'MAE': mae, 'R2': r2})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    main(args)
