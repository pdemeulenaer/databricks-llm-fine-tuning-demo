# Introduction 
Python project for databricks-llm-fine-tuning-demo solution

# GIT Setup
This template must be used as part of a new or existing GIT repository.

# Setting up the environment

You need to have conda installed:
https://www.anaconda.com/download/

You also need to have the Databricks CLI installed. For that, follow instructions in bullet 1 of [this documentation](https://dev.azure.com/northell/Northell/_wiki/wikis/Northell.wiki/19/Anything-Databricks?anchor=1.-install-databricks-cli-and-connect-laptop-to-dbx-workspace)

## Using make
If conda has been successfully installed, project can be set up using make

```bash
$ make setup
```
This command will create conda environment, install the dependencies and create pre-commit git hook for Black

## Using create_environment.sh

If for whatever reason make command failed, `create_environment.sh` file can be used to set up the repository.
Run this command in your terminal of choice to create conda environment and activate it:

```shell
$ source ./create_environment.sh
```

## Activate environment
If the environment is not activated by default, run
```bash
$ conda activate databricks-llm-fine-tuning-demo
```

## Install
This repository uses `src` layout type. In order to have working imports, package has to be installed.
The following make command will install all requirements and also the package itself in editable version.
```bash
$ make install
```

## Env variables
Copy `.env.sample` file and rename it to `.env`. Update it with correct variables.
They will be available in the app using get_settings() function from app.dependencies.py.
This step can be skipped if `make setup` command was used to create environment.

# Running the project

## Running locally

The repo comes with a dummy project which aims to demo a data pipeline and a machine learning pipeline, both runnable either locally or from the Databricks platform. The necessary commands are available in the Makefile. 

**The Data Pipeline**: The data pipeline contained in this repo is made of 2 dummy tasks:

* First task generates data mimicking the Iris dataset, using a Gaussian Mixture Model (fitted on the Iris dataset), so that each create sample comes from the same probability distribution function as real Iris data points, but are not exactly the same. 

* Second task just generate a plot of the multiple dimensions of the generated data, vs the real Iris dataset. The data and the plot are saved in the data/ subfolder of the data_pipeline/ folder

To run the data pipeline locally: 

```bash
$ make run-data-pipeline
```

**The Machine Learning pipeline**: 

The ML pipeline again consists of 2 tasks: 

* First task creates a Random Forest classifier (Scikit-learn) on the dataset and logs the model to MLflow Tracking as an experiment

* Second task evaluates the model and logs the evaluation metric to the MLflow experiment

To run the machine learning pipeline locally:

```bash
$ make run-data-pipeline
```

## Running on the Databricks platform

Prerequisites: 

* As mentioned above, the Databricks CLI should be installed

* The data is meant to be stored in an Azure Blob storage, reachable in Databricks through a "Volume". The Volume, and any "data" folder it is meant to contain have to be created before the data and ML pipelines are executed. For the Volume creation, one has to go to the Catalog section in the Databricks UI. Inside, one can either create a Catalog, schema, volume and folder hierarchy, or simply reuse an existing catalog and schema, and just create a new volume (and a folder called "data" in the volume). As this operation requires elevated priviliges in the Databricks Workspace, it is advised to do it with your Databricks Workspace administrator. In this template, catalog is "responseosdev_catalog" and schema is "volumes". The Volume needs to be created with name "databricks-llm-fine-tuning-demo_volume", and it should contain a folder "data". See this figure for the example of such hierarchy for a project called "my_databricks_project": it is contained in a catalog "responseosdev_catalog", in a schema "volumes" and in a volume "my_databricks_project_volume" and contains the folder "data".

![volume_structure.png](/docs/img/volume_structure.png)

The data pipeline and ML pipeline are coded in such a way that they can run either locally or on the Databricks workspace defined in the databricks.yml root file. For that the code is packaged (using package definition from setup.py) and deployed to the databricks platform like this:

```bash
$ make validate-bundle # optional: this just makes sure that the databricks files are well formatted
$ make deploy-bundle # actual code packaging and deployment to the DBX workspace
```

Then the Workflows section of the Databricks workspace will then display 2 new workflows, for the data and ML pipelines. One can then trigger the execution of each pipeline either in the UI or in the command line, like this, for the data pipeline:

```bash
$ make run-dbx-data-pipeline-dev
```

and this, for the ML pipeline:

```bash
$ make run-dbx-ml-pipeline-dev
```

The execution logs can be observed in the Databricks Workflows UI. 


# Contributing
After making changes to this repos, run
```bash
$ make quality
```
This process will run Black, pylint, mypy and Behave to ensure code quality.

# Template maintenance
* This template is maintained by Philippe de Meulenaer
* This template works with conda version 24.5.0