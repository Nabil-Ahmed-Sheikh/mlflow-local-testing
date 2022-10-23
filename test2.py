import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

code_version = "13"

os.environ["SIDETREK_PROJECT_ID"] = "a7c7184b-26c5-4e38-bb0a-cb3d89d0da74"
os.environ["SIDETREK_USER_ID"] = "1989f0db-a509-433f-8223-2a463ead81c0"
os.environ["SIDETREK_ORGANIZATION_ID"] = "e503b337-c3b5-455d-a744-9f1098c71ec2"

exp_name = "Test Experiment " + code_version
reg_model_name = "ElasticnetWineModel" + code_version

mlflow.set_tracking_uri("https://mlflow.sidetrek.com/p")
experiment_id = mlflow.create_experiment(
    exp_name,
    tags={"version": "v1", "priority": "P1"},
)

print(mlflow.get_experiment_by_name(exp_name))

os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow.get_experiment_by_name(
    exp_name).name

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = 0.5
    l1_ratio = 1

    # mlflow.set_tracking_uri("http://a39be1be3c4a64723a75119ec661f1c7-913060790.us-west-2.elb.amazonaws.com:5000")

    with mlflow.start_run():

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name=reg_model_name)
        else:
            mlflow.sklearn.log_model(lr, "model")
