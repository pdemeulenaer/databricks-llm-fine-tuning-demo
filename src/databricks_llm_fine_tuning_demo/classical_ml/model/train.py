import os
from pathlib import Path
import importlib.resources as pkg_resources
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from databricks_llm_fine_tuning_demo.utils import (
    set_vars, 
    runs_on_databricks, 
    load_parameters, 
    write_df_to_unity_catalog, 
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
        # Load the original and generated data from CSV files
        input_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["input_dir"])
        data = pd.read_csv(os.path.join(input_dir, "generated_iris_data.csv"))
    else:
        # Load the original and generated data from DELTA TABLES
        data, data_version = read_unity_catalog_to_pandas("responseosdev_catalog", "volumes", "original_iris_data")

    # Split features and target
    X = data.drop(columns=["target"])
    y = data["target"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save training and test datasets to CSV and DELTA
    output_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["output_dir"])
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    train_version = write_df_to_unity_catalog(data, "responseosdev_catalog", "volumes", "train_data")
    test_version = write_df_to_unity_catalog(data, "responseosdev_catalog", "volumes", "test_data")
    print(train_data.head())

    # Model definition
    model_name = parameters["training"]["model"]["model_name"]
    n_estimators = parameters["training"]["model"]["n_estimators"]
    max_depth = parameters["training"]["model"]["max_depth"]
    random_state = parameters["training"]["model"]["random_state"]
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Train the model
        model.fit(X_train, y_train)

        # Log model parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)
        # mlflow.log_param("data_delta_path", delta_path)
        if is_databricks:
            mlflow.log_param("data_delta_table", "responseosdev_catalog.volumes.original_iris_data")
        if is_databricks:
            mlflow.log_param("data_delta_version", data_version)
        if is_databricks:
            mlflow.log_param("train_delta_version", train_version)
        # if is_databricks: mlflow.log_param("test_delta_version", test_version) # writing this only at evaluation step

        # Infer the model signature from training data and the trained model
        signature = infer_signature(X_train, model.predict(X_train))

        # Create an input example using a sample of the training data
        input_example = X_train.head(1)

        # Log the model artifact with signature and input example
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)

        print(f"Model trained and saved with run ID: {run.info.run_id}")

    print("Training complete. Data saved to 'train.csv' and 'test.csv'.")


if __name__ == "__main__":
    main()
