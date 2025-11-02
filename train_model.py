# Script to train machine learning model.

import pandas as pd

from sklearn.model_selection import train_test_split

from ml.preprocess import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_slice_metrics
from ml.model import model_save

import joblib
import logging

logging.basicConfig(level=logging.INFO)

def train(data_path: str, target: str, model_output: str, slice_metrics_output: str = None):    
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

    logging.info('evaluating overall model performance')
    y_pred = inference(model, X_test)
    metrics = compute_model_metrics(y_test, y_pred)
    report = pd.DataFrame(list(metrics), index=["precision", "recall", "fbeta"], columns=['value'])
    logging.info(f'model overall performance {report.round(3)}')
    
    logging.info(f'persisting the new trained model at: {model_output}')
    model_save(model, model_output)

    if slice_metrics_output:
        logging.info('computing slices metrics')
        report = []
        # get features with low cardinality
        low_cardinality = test.nunique()<10
        features = test.loc[:,low_cardinality].drop(target,  axis=1, errors='ignore').columns
        logging.info(f'\tselected features for slicing analytics: {features}')
        for f in features:
            logging.info(f'\trunning evaluation for {f}')
            report.append(compute_slice_metrics(model, encoder, lb, test, categorical_cols, f, target=target))
        pd.concat(report, axis=0, ignore_index=True).round(3).to_csv(slice_metrics_output, index=False,)
        logging.info(f'metrics saved at: {slice_metrics_output}')

    logging.info('training completed!')

if __name__ == '__main__':
    train('data/census_clean.csv', 'salary', 'model/model.joblib', 'slice_metrics.csv')