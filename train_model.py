# Script to train machine learning model.

import pandas as pd

from sklearn.model_selection import train_test_split

from ml.preprocess import process_data
from ml.model import train_model, inference, compute_model_metrics
from ml.model import model_save

import joblib
import logging

logging.basicConfig(level=logging.INFO)

def train(data_path: str, target: str, model_output: str = 'model/model.joblib'):    
    logging.info("data loading")
    df = pd.read_csv(data_path)

    logging.info("splitting data on train and test sets")
    train, test = train_test_split(df, test_size=0.20)
 
    logging.info("preprocessing data")
    categorical_cols = df.drop(target, axis=1).select_dtypes('O').columns.tolist()
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=categorical_cols,
        label=target,
        training=True,)
    X_test, y_test, _, _ = process_data(
        test, 
        categorical_features=categorical_cols,
        label=target, 
        training=False, 
        encoder=encoder,
        lb=lb)

    logging.info('training the model')
    model = train_model(X_train, y_train)

    logging.info('evaluating the model')
    y_pred = inference(model, X_test)
    metrics = compute_model_metrics(y_test, y_pred)
    report = pd.DataFrame(list(metrics), index=["precision", "recall", "fbeta"], columns=['value'])
    print(report.round(3))

    logging.info('persisting the new trained model')
    model_save(model, model_output)

    logging.info('training completed!')

if __name__ == '__main__':
    train('data/census_clean.csv', 'salary', 'model/model.joblib')