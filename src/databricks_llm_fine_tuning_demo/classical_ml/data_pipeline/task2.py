import os
from pathlib import Path
import importlib.resources as pkg_resources
import pandas as pd

from databricks_llm_fine_tuning_demo.utils import set_vars, runs_on_databricks, load_parameters
from databricks_llm_fine_tuning_demo.data_pipeline.utils import make_visualisation

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

    # Load the original and generated data from CSV files
    input_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["input_dir"])
    data = pd.read_csv(os.path.join(input_dir, "original_iris_data.csv"))
    generated_df = pd.read_csv(os.path.join(input_dir, "generated_iris_data.csv"))

    # Make the picture of data
    save_path = os.path.join(data_dir, "data/data_generated.png")
    make_visualisation(data, generated_df, save_path)


if __name__ == "__main__":
    main()
