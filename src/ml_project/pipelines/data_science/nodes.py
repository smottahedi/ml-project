# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import mlflow
from mlflow import sklearn
from datetime import datetime
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
mlflow.sklearn.autolog()

def train_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
) -> np.ndarray:
    """Node for training a simple multi-class svm model. The parameters
    are taken from conf/project/parameters.yml. All of the data as well
    as the parameters will be provided to this function at the time of execution.
    """
    X = train_x.values
    Y = train_y.values

    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters['model_params'])

    clf.fit(X, Y)

    sklearn.log_model(sk_model=clf.best_estimator_, artifact_path="model")
    return clf.best_estimator_


def predict(model: ClassifierMixin, test_x: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    X = test_x.values

    return model.predict(X)

def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Get true class index
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == test_y) / test_y.shape[0]
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("time of prediction", str(datetime.now()))
