from typing import List
import pandas as pd

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier

from ml.preprocess import process_data

import joblib


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    return model

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    pred = model.predict(X)
    return pred

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def compute_slice_metrics(model, encoder, lb, test_data, categorical_cols: List, slice_feature: str, target: str):
    slice_report = []
    # run metrics for data slices of each value case of selected feature
    for fix_val in test_data[slice_feature].unique():
        # filter slice based on fixed value for selected feature
        X_slice = test_data.loc[test_data[slice_feature]==fix_val,:]
        X_slice, y_slice, _, _  = process_data(X_slice, categorical_cols, label=target, training=False, encoder=encoder, lb=lb)
        pred = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, pred)
        slice_report.append([slice_feature,fix_val,precision, recall, fbeta,X_slice.size])
    # get a dataframe compiling all metrics for each fixed parameters
    return pd.DataFrame(slice_report, columns=['feature', 'slice', 'precision', 'recall', 'f1', 'samples'])

def model_save(model, output_path):
    """
    Persist fitted model
    """
    joblib.dump(model, output_path)

def model_load(path):
    """
    Load a trained model artifact
    """
    lb = joblib.load("/".join([path,'lb.joblib']))
    encoder = joblib.load("/".join([path,'encoder.joblib']))
    model = joblib.load("/".join([path,'model.joblib']))
    return model, encoder, lb




