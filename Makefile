.ONESHELL:

SHELL := $(shell which bash)
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# 0. General local commands

env-file:
	cp .env.sample .env

conda:
	conda env create --file environment.yml --yes
	$(CONDA_ACTIVATE) databricks-llm-fine-tuning-demo

install:
	pip install -r requirements.txt
	pip install -e .

pre-commit:
	pre-commit install

setup: env-file conda pre-commit

black:
	black .

lint:
	mypy src
	pylint src

test:
	behave tests/features/

doc: 
	mkdocs build	

quality: black lint test

quality-ci: lint test


# 1.1 Local commands to run the DATA pipeline

data-extraction:
	python -m src.databricks_llm_fine_tuning_demo.data_pipeline.data_loader

# data-visualization:
# 	python -m src.databricks_llm_fine_tuning_demo.data_pipeline.task2

run-data-pipeline: data-extraction # data-visualization	

# 1.2 Local commands to run the ML pipeline

train:
	python -m src.databricks_llm_fine_tuning_demo.model.train		

evaluate:
	python -m src.databricks_llm_fine_tuning_demo.model.evaluate	

run-ml-pipeline: train evaluate		

# TODO: promote model to MR

# 2. The following commands are meant to deploy & run the package to Databricks

deploy-bundle:
	databricks bundle deploy -t dev

validate-bundle:
	databricks bundle validate -t dev	

run-dbx-data-pipeline-dev:
	databricks bundle run -t dev data_pipeline_job

run-dbx-ml-pipeline-dev:
	databricks bundle run -t dev ml_pipeline_job	

# TODO: Generate a Dockerfile out of the MLflow run of choice
# See https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker
# mlflow-to-dockerfile:
# 	mlflow models build-docker -m runs:/<run_id>/model -n <image_name>	
