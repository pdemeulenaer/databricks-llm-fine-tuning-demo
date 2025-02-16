# Following https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft.html

import os
from pathlib import Path
import importlib.resources as pkg_resources

# import torch
import mlflow
from datetime import datetime
from datasets import load_dataset
import transformers

from databricks_llm_fine_tuning_demo.utils import (
    set_vars,
    runs_on_databricks,
    load_parameters,
    get_latest_mlflow_run_id,
    display_table,
)

set_vars()
is_databricks = runs_on_databricks()


def main():

    # Get the path of the package
    resource = pkg_resources.files("databricks_llm_fine_tuning_demo")
    package_dir = Path(str(resource))
    print(package_dir)

    # Read the yaml configuration file
    config_file_path = os.path.join(package_dir, "llm_fine_tuning/config.yaml")
    parameters = load_parameters(config_file_path)
    print(parameters)

    if is_databricks:
        # If Databricks, then read input data from Volume
        data_dir = "/Volumes/responseosdev_catalog/volumes/databricks_llm_fine_tuning_demo_volume"
    else:
        # else read input data from package
        data_dir = os.path.join(package_dir,"llm_fine_tuning")

    # # Load the train and test data from CSV files
    # input_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["input_dir"])
    # # train_dataset = pd.read_csv(os.path.join(input_dir, "hf_data_train.csv"))
    # # test_dataset = pd.read_csv(os.path.join(input_dir, "hf_data_test.csv"))
    # train_dataset = load_dataset("csv", data_files=os.path.join(input_dir, "hf_data_train.csv"), split="train")
    # test_dataset = load_dataset("csv", data_files=os.path.join(input_dir, "hf_data_test.csv"), split="train")

    # print(type(test_dataset))
    # print(test_dataset)
    # print(display_table(test_dataset))
    # # input()

    # 5. Kick-off the Training Job

    mlflow.set_tracking_uri("databricks")
    experiment_name = f"/Shared/MLflow PEFT Tutorial"
    mlflow.set_experiment(experiment_name)
    last_run_id = get_latest_mlflow_run_id(experiment_name)

    # 7. Load the Saved PEFT Model from MLflow

    # You can find the ID of run in the Run detail page on MLflow UI
    mlflow_model = mlflow.pyfunc.load_model(f"runs:/{last_run_id}/model")

    # We only input table and question, since system prompt is adeed in the prompt template.
    test_prompt = """
    ### Table:
    CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)

    ### Question:
    When Essendon played away; where did they play?
    """

    # Inference parameters like max_tokens_length are set to default values specified in the Model Signature
    generated_query = mlflow_model.predict(test_prompt)[0]
    display_table({"prompt": test_prompt, "generated_query": generated_query})


if __name__ == "__main__":
    main()
