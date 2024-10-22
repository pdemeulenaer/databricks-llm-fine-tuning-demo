import os
import re
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from typing import Union, Dict
from datasets import Dataset
# import torch
import yaml  # type: ignore
import mlflow  # type: ignore
from mlflow.tracking import MlflowClient  # type: ignore


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the column names of a pandas DataFrame.

    This function replaces specific characters in column names with underscores
    and removes leading/trailing underscores.

    Args:
        df: The input DataFrame whose column names need to be cleaned.

    Returns:
        A new DataFrame with cleaned column names.

    Note:
        The characters replaced with underscores are: space, comma, semicolon,
        curly braces, parentheses, newline, tab, and equals sign.
    """
    return df.rename(columns=lambda x: re.sub(r'[ ,;{}()\n\t=]', '_', x).strip('_'))


def load_parameters(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML file containing parameters.

    Args:
        path (str): The path to the YAML file.

    Returns:
        dict: The parameters.
    """
    if path is None:
        path = "parameters.yml"
    with open(path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    return params


# def get_device() -> torch.device:
#     """
#     Automatically selects the best available device (GPU, MPS, or CPU).

#     Returns:
#         torch.device: The best available device.
#     """
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         logging.info("Using %s As Device", device.type)
#         return device
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         logging.info("Using %s As Device", device.type)
#         return device
#     device = torch.device("cpu")
#     logging.info("Using %s As Device", device.type)
#     return device


def get_dbutils():
    """
    Necessary function when running on Databricks
    """
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils  # type: ignore[import-not-found]

    spark = SparkSession.builder.getOrCreate()

    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        return DBUtils(spark)

    try:
        import IPython

        return IPython.get_ipython().user_ns["dbutils"]
    except ImportError:
        raise ImportError("IPython is not available. Make sure you're not in a non-IPython environment.")


def runs_on_databricks() -> bool:
    """
    Detect if the code is running on the Databricks platform.

    This function attempts to determine whether the current execution environment
    is Databricks by checking for Databricks-specific attributes and environment
    variables.

    Returns:
        bool: True if running on Databricks, False otherwise.

    Note:
        - The function first tries to import and use the `dbutils` object, which
          is typically available in Databricks environments.
        - It also checks for the presence of the 'DATABRICKS_RUNTIME_VERSION'
          environment variable.
        - An ImportError when trying to get `dbutils` is interpreted as evidence
          that the code is not running on Databricks.
    """
    try:
        dbutils = get_dbutils()
        return "DATABRICKS_RUNTIME_VERSION" in os.environ
    except ImportError:
        return False  


def set_vars() -> None:
    """
    Set environment variables based on the execution environment.

    This function determines whether the code is running on Databricks or locally,
    and sets environment variables accordingly:
    - If running on Databricks, it reads variables from the Databricks secret scope.
    - If running locally, it loads variables from a .env file.

    The following environment variables are set:
    - AZURE_API_VERSION
    - AZURE_API_KEY
    - AZURE_API_BASE
    - HF_AUTH_TOKEN

    Returns:
        None

    Raises:
        ImportError: If running locally and python-dotenv is not installed.

    Note:
        - The function uses the `runs_on_databricks()` function to determine the 
          execution environment.
        - When running on Databricks, it uses the `dbutils` object to access 
          secrets from the "responseos" scope.
        - When running locally, it requires the python-dotenv package to be 
          installed and a .env file to be present in the working directory.
    """
    if runs_on_databricks():
        print("Running on Databricks")
        dbutils = get_dbutils()
        os.environ["AZURE_API_VERSION"] = dbutils.secrets.get(scope="responseos", key="AZURE_API_VERSION")
        os.environ["AZURE_API_KEY"] = dbutils.secrets.get(scope="responseos", key="AZURE_API_KEY")
        os.environ["AZURE_API_BASE"] = dbutils.secrets.get(scope="responseos", key="AZURE_API_BASE")
        os.environ["HF_AUTH_TOKEN"] = dbutils.secrets.get(scope="responseos", key="HF_AUTH_TOKEN")

    else:
        print("Running locally on Python")
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            raise ImportError("python-dotenv is required to run this function locally. "
                              "Please install it using 'pip install python-dotenv'.")


def write_df_to_unity_catalog(df: pd.DataFrame, catalog: str, schema: str, table: str, mode: str = "overwrite") -> int:
    """
    Write a pandas DataFrame to a Delta table in Unity Catalog when running on Databricks.
    If running locally, this function does nothing.

    Args:
        df (pd.DataFrame): The pandas DataFrame to write.
        catalog (str): The name of the Unity Catalog.
        schema (str): The name of the schema in the Unity Catalog.
        table (str): The name of the table to write to.
        mode (str, optional): The write mode ('overwrite' or 'append'). Defaults to 'overwrite'.

    Returns:
        int: the version number of the Delta table after the write operation.
             Returns -1 if not running on Databricks.

    Raises:
        ValueError: If an invalid mode is specified.
        Exception: For any issues writing to the Delta table.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        >>> version = write_df_to_unity_catalog(df, "my_catalog", "my_schema", "my_table")
        >>> print(f"New version: {version}")
    """

    # Check if running on Databricks
    is_databricks = runs_on_databricks()

    if not is_databricks:
        print("Not running on Databricks. No action taken.")
        return -1

    if mode not in ['overwrite', 'append']:
        raise ValueError("Mode must be either 'overwrite' or 'append'")

    try:
        # Import PySpark modules (these are available in Databricks environment)
        from pyspark.sql import SparkSession
        from delta.tables import DeltaTable

        # Get or create Spark session
        spark = SparkSession.builder.getOrCreate()

        # Convert pandas DataFrame to Spark DataFrame
        spark_df = spark.createDataFrame(df)

        # Write to Unity Catalog
        table_path = f"{catalog}.{schema}.{table}"
        spark_df.write.format("delta").mode(mode).saveAsTable(table_path)

        # Get the version number of the Delta table
        delta_table = DeltaTable.forName(spark, table_path)
        version = delta_table.history().first().version        

        print(f"Data successfully written to {table_path}")
        return version

    except Exception as e:
        print(f"Error writing to Delta table in Unity Catalog: {str(e)}")
        return -1


def read_unity_catalog_to_pandas(catalog: str, schema: str, table: str) -> Tuple[pd.DataFrame, int]:
    """
    Read a Delta table from Unity Catalog into a pandas DataFrame when running on Databricks.
    If running locally, this function returns an empty DataFrame.

    Args:
        catalog (str): The name of the Unity Catalog.
        schema (str): The name of the schema in the Unity Catalog.
        table (str): The name of the table to read from.

    Returns:
        Tuple[pd.DataFrame, int]: A tuple containing the data from the Delta table as a pandas DataFrame,
                                  and the version number of the Delta table.
                                  Returns an empty DataFrame and -1 if not running on Databricks.

    Raises:
        Exception: For any issues reading from the Delta table.

    Example:
        >>> df, version = read_unity_catalog_to_pandas("my_catalog", "my_schema", "my_table")
        >>> print(f"Data read from version: {version}")
        >>> print(df.head())
    """
    # Check if running on Databricks
    is_databricks = runs_on_databricks()

    if not is_databricks:
        print("Not running on Databricks. Returning an empty DataFrame.")
        return pd.DataFrame(), -1

    try:
        # Import PySpark modules (these are available in Databricks environment)
        from pyspark.sql import SparkSession
        from delta.tables import DeltaTable

        # Get or create Spark session
        spark = SparkSession.builder.getOrCreate()

        # Read from Unity Catalog
        table_path = f"{catalog}.{schema}.{table}"
        spark_df = spark.read.table(table_path)

        # Get the version number of the Delta table
        delta_table = DeltaTable.forName(spark, table_path)
        version = delta_table.history().first().version

        # Convert Spark DataFrame to pandas DataFrame
        pandas_df = spark_df.toPandas()

        print(f"Data successfully read from {table_path}")
        return pandas_df, version

    except Exception as e:
        print(f"Error reading Delta table from Unity Catalog: {str(e)}")
        return pd.DataFrame(), -1       


def get_latest_mlflow_run_id(experiment_name: str) -> int:
    """
    Get the run ID of the latest MLflow run for a given experiment.

    This function retrieves the most recent run from the specified MLflow experiment
    and returns its run ID. If no runs are found, it raises a ValueError.

    Args:
        experiment_name: The name of the MLflow experiment to query.

    Returns:
        The run ID of the latest run in the experiment, or None if no runs are found.

    Raises:
        ValueError: If no runs are found in the specified experiment.
        mlflow.exceptions.MlflowException: If the specified experiment is not found.

    Note:
        - This function assumes that MLflow is properly configured in the current environment.
        - The latest run is determined based on the start time of the runs.
        - Only the most recent run is retrieved to optimize performance.
    """

    # Retrieve the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")    

    # Get the latest run in the experiment
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )

    # Ensure we have at least one run available
    if not runs:
        raise ValueError(f"No runs found in the experiment '{experiment_name}'.")

    # Get the run_id of the latest run
    latest_run_id = runs[0].info.run_id
    return latest_run_id


def download_latest_mlflow_run(
    experiment_name: str,
    mlflow_artifacts_dir: str = "./mlflow_artifacts",
    model_name: str = "bert-base-uncased"
) -> Union[str, int]:
    """
    Download artifacts from the latest MLflow run for a given experiment.

    This function retrieves the artifacts from the most recent run of a specified MLflow
    experiment and saves them to a local directory. It creates the necessary directories
    if they don't exist.

    Args:
        experiment_name: The name of the MLflow experiment from which to download the artifacts.
        mlflow_artifacts_dir: The local directory where the artifacts will be stored.
            Defaults to "./mlflow_artifacts".
        model_name: The name of the model or subdirectory to use within the artifacts directory.
            Defaults to "bert-base-uncased".

    Returns:
        The run ID (string) of the latest run if artifacts are successfully downloaded,
        or 0 (int) if no runs are found or download fails.

    Raises:
        OSError: If there's an issue creating the artifact directory.
        mlflow.exceptions.MlflowException: If there's an issue with MLflow operations.

    Note:
        - This function assumes that MLflow is properly configured in the current environment.
        - It uses the `get_latest_mlflow_run_id` function to retrieve the latest run ID.
        - All artifacts from the latest run are downloaded, not just those related to the specified model name.
    """

    model_dir = os.path.join(mlflow_artifacts_dir, model_name)

    try:
        os.makedirs(mlflow_artifacts_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {mlflow_artifacts_dir}: {e}")
        return 0

    client = MlflowClient()

    latest_run_id = get_latest_mlflow_run_id(experiment_name)
    print(f"Latest run ID: {latest_run_id}")

    if latest_run_id:
        try:
            artifacts = client.list_artifacts(latest_run_id)
            print(f"Artifacts found: {len(artifacts)}")

            for artifact in artifacts:
                print(f"Downloading: {artifact.path}")
                client.download_artifacts(latest_run_id, artifact.path, mlflow_artifacts_dir)

            print(f"Artifacts downloaded to: {model_dir}")
            return latest_run_id
        except mlflow.exceptions.MlflowException as e:
            print(f"Error downloading artifacts: {e}")
            return 0
    else:
        print("No runs found for the experiment.")
        return 0


def display_table(dataset_or_sample: Union[Dict, Dataset]) -> None:
    """
    Display a Transformer dataset or single sample containing multi-line strings in a nicely formatted table.

    This function uses pandas to create a readable table format for the input data. It sets pandas display
    options to show full content without truncation and prints the first 5 rows of the resulting DataFrame.

    Args:
        dataset_or_sample (Union[dict, datasets.Dataset]): The input data to be displayed. Can be either a
            single sample as a dictionary or a Hugging Face datasets.Dataset object.

    Returns:
        None: This function doesn't return anything; it prints the formatted table to the console.

    Note:
        - The function sets pandas display options globally, which may affect other pandas operations
          in the same session.
        - The commented-out HTML formatting code is preserved for potential future use.
    """
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", None)
    if isinstance(dataset_or_sample, dict):
        df = pd.DataFrame(dataset_or_sample, index=[0])
    else:
        df = pd.DataFrame(dataset_or_sample)
    print(df.head(5))  