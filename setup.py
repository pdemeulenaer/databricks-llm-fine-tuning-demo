from setuptools import setup, find_packages, find_namespace_packages
# import codecs
import sys
sys.path.append("./src")
import datetime
import databricks_llm_fine_tuning_demo


setup(
    name='databricks_llm_fine_tuning_demo',
    version=databricks_llm_fine_tuning_demo.__version__ + "+" + datetime.datetime.utcnow().strftime("%Y%m%d.%H%M%S"),
    url="https://databricks.com",
    author="Northell",
    author_email="philippe.demeulenaer@northell.com",
    description="wheel file based on databricks-llm-fine-tuning-demo/src",
    packages=find_packages(where="./src"),
    package_dir={"": "src"},
    entry_points={
        "packages": [
            "data_extraction=databricks_llm_fine_tuning_demo.data_pipeline.task1:main",
            "data_visualization=databricks_llm_fine_tuning_demo.data_pipeline.task2:main",
            "training=databricks_llm_fine_tuning_demo.model.train:main",
            "evaluation=databricks_llm_fine_tuning_demo.model.evaluate:main",
        ]
    },
    install_requires=[
        # Dependencies in case the output wheel file is used as a library dependency.
        # For defining dependencies, when this package is used in Databricks, see:
        # https://docs.databricks.com/dev-tools/bundles/library-dependencies.html
        "setuptools"
    ],
    # package_data={'global_toolkit': ['**/*.yaml']},
    package_data={"": ["*.yaml", "*.csv", "*.pkl"]},  # Include all YAML, CSV and PKL files in the package
    include_package_data=True,  # Include package data specified by package_data
)
