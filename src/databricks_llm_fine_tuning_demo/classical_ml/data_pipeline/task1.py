import os
from pathlib import Path
import importlib.resources as pkg_resources
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from databricks_llm_fine_tuning_demo.utils import (
    set_vars,
    runs_on_databricks,
    load_parameters,
    clean_column_names,
    write_df_to_unity_catalog,
    read_unity_catalog_to_pandas,
)
from databricks_llm_fine_tuning_demo.classical_ml.data_pipeline.utils import generate_samples_from_gmm

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

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Combine features and labels into a single DataFrame
    data = pd.DataFrame(X, columns=iris.feature_names)
    data["target"] = y

    # Clean column names
    data = clean_column_names(data)
    print(data.head())

    # Split the data by class
    class_0 = data[data["target"] == 0].drop(columns="target")
    class_1 = data[data["target"] == 1].drop(columns="target")
    class_2 = data[data["target"] == 2].drop(columns="target")

    # Generate 1000 samples for each class using the GMMs
    samples_class_0 = generate_samples_from_gmm(class_0, n_samples=1000)
    samples_class_1 = generate_samples_from_gmm(class_1, n_samples=1000)
    samples_class_2 = generate_samples_from_gmm(class_2, n_samples=1000)

    # Combine generated samples and create labels
    generated_data = np.vstack([samples_class_0, samples_class_1, samples_class_2])
    generated_labels = np.array([0] * 1000 + [1] * 1000 + [2] * 1000)

    # Convert to DataFrame for easy handling
    generated_df = pd.DataFrame(generated_data, columns=iris.feature_names)
    generated_df["target"] = generated_labels

    # Display the count of samples per class
    print(generated_df["target"].value_counts())

    # Show the first few rows of the generated dataset
    generated_df = clean_column_names(generated_df)
    print(generated_df.head())

    # Save the original and generated data to CSV files
    output_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["output_dir"])
    data.to_csv(os.path.join(output_dir, "original_iris_data.csv"), index=False)
    generated_df.to_csv(os.path.join(output_dir, "generated_iris_data.csv"), index=False)

    # # Save the original and generated data to DELTA files
    if is_databricks:  # then write the output data to Unity Catalog as delta tables
        write_df_to_unity_catalog(data, "responseosdev_catalog", "volumes", "original_iris_data")
        write_df_to_unity_catalog(data, "responseosdev_catalog", "volumes", "generated_iris_data")

        # Test to see if we can read the delta table
        data2, data2_delta_version = read_unity_catalog_to_pandas(
            "responseosdev_catalog", "volumes", "original_iris_data"
        )
        print(data2.head())
        print("version of data: ", data2_delta_version)

    print("Data saved to 'original_iris_data.csv' and 'generated_iris_data.csv'")


if __name__ == "__main__":
    main()
