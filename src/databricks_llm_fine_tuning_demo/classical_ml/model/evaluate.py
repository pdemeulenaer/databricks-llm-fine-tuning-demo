import os
from pathlib import Path
import importlib.resources as pkg_resources
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from databricks_llm_fine_tuning_demo.utils import (
    set_vars,
    runs_on_databricks,
    load_parameters,
    get_latest_mlflow_run_id,
    read_unity_catalog_to_pandas,
)

set_vars()
is_databricks = runs_on_databricks()


def main():

    # Get the path of the package
    resource = pkg_resources.files("databricks_llm_fine_tuning_demo")
    package_dir = Path(str(resource))
    print(package_dir)

    # Read the yaml configuration file
    config_file_path = os.path.join(package_dir, "classical_ml/config.yaml")
    parameters = load_parameters(config_file_path)
    print(parameters)

    if is_databricks:
        # If Databricks, then read input data from Volume
        data_dir = "/Volumes/responseosdev_catalog/volumes/databricks_llm_fine_tuning_demo_volume"
    else:
        # else read input data from package
        data_dir = os.path.join(package_dir,"classical_ml")

    # Set the MLflow experiment name
    experiment_name = parameters["training"]["experiment_name"]

    # Detect the desired MLflow location
    mlflow_location = parameters["mlflow_location"]
    if mlflow_location == "databricks":
        # Set the tracking URI to your Databricks workspace
        mlflow.set_tracking_uri("databricks")
        # Set the MLflow experiment
        experiment_name = f"/Shared/{experiment_name}"
        mlflow.set_experiment(experiment_name)
    else:
        # Set the MLflow experiment
        mlflow.set_experiment(experiment_name)

    if not is_databricks:
        # Load the test dataset from CSV file
        input_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["input_dir"])
        test_data = pd.read_csv(os.path.join(input_dir, "test.csv"))
    else:
        # Load the original and generated data from DELTA TABLES
        test_data, test_version = read_unity_catalog_to_pandas("responseosdev_catalog", "volumes", "test_data")

    print(test_data.head())

    # Split features and target
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    # Get the latest MLflow run ID
    run_id = get_latest_mlflow_run_id(experiment_name)

    # Load the trained model from MLflow
    model_name = parameters["training"]["model"]["model_name"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log the accuracy metric to the same run
    with mlflow.start_run(run_id=run_id):  # Ensures logging to the existing run
        mlflow.log_metric("accuracy", accuracy)
        if is_databricks:
            mlflow.log_param("test_delta_version", test_version)

    print(f"Model evaluated on test set. Accuracy: {accuracy:.4f} logged in run ID: {run_id}")


if __name__ == "__main__":
    main()
