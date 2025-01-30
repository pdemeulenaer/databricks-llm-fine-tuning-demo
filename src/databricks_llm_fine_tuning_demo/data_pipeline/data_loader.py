# Following https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft.html

import os
from pathlib import Path
import importlib.resources as pkg_resources
import pandas as pd
from datasets import load_dataset

# from IPython.display import HTML, display
# import torch
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

from databricks_llm_fine_tuning_demo.utils import (
    set_vars,
    runs_on_databricks,
    load_parameters,
    display_table,
    # clean_column_names,
    # write_df_to_unity_catalog,
    # read_unity_catalog_to_pandas,
)

set_vars()
is_databricks = runs_on_databricks()


def main():

    # 1. Read configuration

    # Get the path of the package
    resource = pkg_resources.files("databricks_llm_fine_tuning_demo")
    package_dir = Path(str(resource))
    print(package_dir)

    # Read the yaml configuration file
    config_file_path = os.path.join(package_dir, "config.yaml")
    parameters = load_parameters(config_file_path)
    print(parameters)

    if is_databricks:
        # If Databricks, then read input data from Volume
        data_dir = "/Volumes/responseosdev_catalog/volumes/databricks_llm_fine_tuning_demo_volume"
    else:
        # else read input data from package
        # data_dir = os.path.join(package_dir,"llm_fine_tuning")
        data_dir = package_dir

    # 2. Dataset Preparation

    # Load Dataset from HuggingFace Hub
    dataset_name = parameters["data_pipeline"]["data"]["input_dataset"]  # "b-mc2/sql-create-context"
    dataset = load_dataset(dataset_name, split="train")
    display_table(dataset.select(range(3)))

    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    print(f"Training dataset contains {len(train_dataset)} text-to-SQL pairs")
    print(f"Test dataset contains {len(test_dataset)} text-to-SQL pairs")

    # 3. Save the original and generated data to CSV files
    output_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["output_dir"])
    dataset.to_csv(os.path.join(output_dir, "hf_data.csv"), index=False)
    train_dataset.to_csv(os.path.join(output_dir, "hf_data_train.csv"), index=False)
    test_dataset.to_csv(os.path.join(output_dir, "hf_data_test.csv"), index=False)

    print(type(train_dataset))


if __name__ == "__main__":
    main()
