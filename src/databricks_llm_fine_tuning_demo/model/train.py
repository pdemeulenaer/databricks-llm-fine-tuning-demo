# Following https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft.html

import os
from pathlib import Path
import importlib.resources as pkg_resources
import torch
import mlflow
from datetime import datetime
from datasets import load_dataset
from mlflow.models import infer_signature
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from databricks_llm_fine_tuning_demo.utils import (
    set_vars,
    runs_on_databricks,
    load_parameters,
    get_latest_mlflow_run_id,
    display_table,
    # clean_column_names,
    # write_df_to_unity_catalog,
    # read_unity_catalog_to_pandas,
)

set_vars()
is_databricks = runs_on_databricks()
# Set the tracking URI to your Databricks workspace
mlflow.set_tracking_uri("databricks")
mlflow.transformers.autolog()


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

    # 2. Dataset preparation

    # Load the train and test data from CSV files
    input_dir = os.path.join(data_dir, parameters["data_pipeline"]["data"]["input_dir"])
    # train_dataset = pd.read_csv(os.path.join(input_dir, "hf_data_train.csv"))
    # test_dataset = pd.read_csv(os.path.join(input_dir, "hf_data_test.csv"))
    train_dataset = load_dataset("csv", data_files=os.path.join(input_dir, "hf_data_train.csv"), split="train")
    test_dataset = load_dataset("csv", data_files=os.path.join(input_dir, "hf_data_test.csv"), split="train")

    # train_dataset = train_dataset.select(range(100)) # meant to be a FAST RUN ! not working yet

    print(type(test_dataset))
    print(test_dataset)
    print(display_table(test_dataset))
    # input()

    # Define Prompt Template
    PROMPT_TEMPLATE = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

    ### Table:
    {context}

    ### Question:
    {question}

    ### Response:
    {output}"""

    def apply_prompt_template(row):
        prompt = PROMPT_TEMPLATE.format(
            question=row["question"],
            context=row["context"],
            output=row["answer"],
        )
        return {"prompt": prompt}

    train_dataset = train_dataset.map(apply_prompt_template)
    test_dataset = test_dataset.map(apply_prompt_template)
    display_table(train_dataset.select(range(1)))

    # Padding the Training Dataset

    tokenizer_id = parameters["tokenizer"]["tokenizer_id"]
    # You can use a different max length if your custom dataset has shorter/longer input sequences.
    MAX_LENGTH = parameters["tokenizer"]["config"]["max_length"]
    truncation = parameters["tokenizer"]["config"]["truncation"]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        model_max_length=MAX_LENGTH,
        padding_side="left",
        add_eos_token=True,
        trust_remote_code=True,
        token=os.environ["HF_AUTH_TOKEN"],
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_and_pad_to_fixed_length(sample):
        result = tokenizer(
            sample["prompt"],
            truncation=truncation,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = train_dataset.map(tokenize_and_pad_to_fixed_length)
    tokenized_test_dataset = test_dataset.map(tokenize_and_pad_to_fixed_length)

    assert all(len(x["input_ids"]) == MAX_LENGTH for x in tokenized_train_dataset)

    display_table(tokenized_train_dataset.select(range(1)))

    # 3. Load the Base Model (with 4-bit quantization)

    quantization_config = BitsAndBytesConfig(
        # Load the model with 4-bit quantization
        load_in_4bit=True,
        # Use double quantization
        bnb_4bit_use_double_quant=True,
        # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_quant_type="nf4",
        # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model_id = parameters["model"][
        "base_model_id"
    ]  # "microsoft/Phi-3.5-mini-instruct"  # "mistralai/Mistral-7B-v0.1"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        trust_remote_code=True,  # Add this line
    )

    # How the Base model performs?
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    pipeline = transformers.pipeline(model=model, tokenizer=tokenizer, task="text-generation")

    sample = test_dataset[1]
    prompt = PROMPT_TEMPLATE.format(
        context=sample["context"], question=sample["question"], output=""
    )  # Leave the answer part blank

    with torch.no_grad():
        response = pipeline(prompt, max_new_tokens=256, repetition_penalty=1.15, return_full_text=False)

    display_table({"prompt": prompt, "generated_query": response[0]["generated_text"]})

    # 4. Define a PEFT Model

    # Enabling gradient checkpointing, to make the training further efficient
    model.gradient_checkpointing_enable()
    # Set up the model for quantization-aware training e.g. casting layers, parameter freezing, etc.
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        # This is the rank of the decomposed matrices A and B to be learned during fine-tuning. A smaller number will save more GPU memory but might result in worse performance.
        r=32,
        # This is the coefficient for the learned ΔW factor, so the larger number will typically result in a larger behavior change after fine-tuning.
        lora_alpha=64,
        # Drop out ratio for the layers in LoRA adaptors A and B.
        lora_dropout=0.1,
        # We fine-tune all linear layers in the model. It might sound a bit large, but the trainable adapter size is still only **1.16%** of the whole model.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        # Bias parameters to train. 'none' is recommended to keep the original model performing equally when turning off the adapter.
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # 5. Kick-off the Training Job

    # Comment-out this line if you are running the tutorial on Databricks
    mlflow.set_tracking_uri("databricks")
    experiment_name = parameters["training"]["experiment_name"]  # f"/Shared/MLflow PEFT Tutorial"
    mlflow.set_experiment(experiment_name)
    # experiment_name = f"/Shared/{experiment_name}"

    # Extract the 'config' section from the YAML
    config_dict = parameters["training"]["config"]

    # Define additional training parameters
    additional_params = {
        "run_name": f"phi35-SQL-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    }

    # Combine YAML config with additional parameters
    # Additional parameters will override YAML config if there are duplicates
    combined_config = {**config_dict, **additional_params}

    # training_args = TrainingArguments(**combined_config)

    training_args = TrainingArguments(
        # Set this to mlflow for logging your training
        report_to="mlflow",
        # Name the MLflow run
        run_name=f"phi35-SQL-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
        # Replace with your output destination
        output_dir="./model_output",
        # For the following arguments, refer to https://huggingface.co/docs/transformers/main_classes/trainer
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        max_steps=500,
        warmup_steps=5,
        # https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005/3
        ddp_find_unused_parameters=False,
        # # num_train_epochs=1,   # added
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,  # evaluating at every step slows down a lot the training!
    )

    trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=training_args,
    )

    # use_cache=True is incompatible with gradient checkpointing.
    peft_model.config.use_cache = False

    trainer.train()

    # 6. Save the PEFT Model to MLflow

    # Prompt template: Basically the same format as we applied to the dataset. However, the template only accepts {prompt} variable so both table and question need to be fed in there.
    prompt_template = """You are a powerful text-to-SQL model. Given the SQL tables and natural language question, your job is to write SQL query that answers the question.

    {prompt}

    ### Response:
    """

    # MLflow signature: infers schema from the provided sample input/output/params
    sample = train_dataset[1]
    signature = infer_signature(
        model_input=sample["prompt"],
        model_output=sample["answer"],
        # Parameters are saved with default values if specified
        params={"max_new_tokens": 256, "repetition_penalty": 1.15, "return_full_text": False},
    )

    # Save the model to MLflow

    # Get the ID of the MLflow Run that was automatically created above
    # last_run_id = mlflow.last_active_run().info.run_id
    last_run_id = get_latest_mlflow_run_id(experiment_name)

    # Save a tokenizer without padding because it is only needed for training
    tokenizer_no_pad = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

    # If you interrupt the training, uncomment the following line to stop the MLflow run
    # mlflow.end_run()

    with mlflow.start_run(run_id=last_run_id):
        mlflow.log_params(peft_config.to_dict())
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer_no_pad},
            prompt_template=prompt_template,
            signature=signature,
            artifact_path="model",  # This is a relative path to save model files within MLflow run
        )

    # # 7. Load the Saved PEFT Model from MLflow

    # # You can find the ID of run in the Run detail page on MLflow UI
    # mlflow_model = mlflow.pyfunc.load_model("runs:/YOUR_RUN_ID/model")

    # # We only input table and question, since system prompt is adeed in the prompt template.
    # test_prompt = """
    # ### Table:
    # CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)

    # ### Question:
    # When Essendon played away; where did they play?
    # """

    # # Inference parameters like max_tokens_length are set to default values specified in the Model Signature
    # generated_query = mlflow_model.predict(test_prompt)[0]
    # display_table({"prompt": test_prompt, "generated_query": generated_query})


if __name__ == "__main__":
    main()
