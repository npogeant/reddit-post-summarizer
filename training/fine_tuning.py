"""
This script enables fine-tuning of several text summarization models on a given dataset. 
The models that will be experimented with are T5 and BART, and the dataset used will be a Reddit summarization dataset.
All runs of the experiments can be reviewed on Weight&Biases as they are logged in. 
The training will be conducted using the Transformers library from HuggingFace and PyTorch.
"""

import os

import nltk
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

import torch
import wandb
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)

# %env WANDB_LOG_MODEL=true
os.environ["WANDB_LOG_MODEL"] = "true"


# Function to load the dataset
def load_data():
    """
    Load the dataset from HuggingFace and return the train, validation, and test datasets.
    """

    # Load the dataset
    dataset = load_dataset("reddit", split="train", streaming=True, cache_dir="cache")
    dataset = dataset.with_format("torch")

    # Shuffle the dataset
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000).take(100000)

    # Create splits from the shuffled dataset
    train_dataset = shuffled_dataset.skip(20000)
    val_dataset = shuffled_dataset.take(10000)
    test_dataset = shuffled_dataset.skip(10000).take(10000)

    return train_dataset, val_dataset, test_dataset


# Function to load the model and tokenizer
def load_model(model_name):
    """
    Load the model and tokenizer from HuggingFace and return the model, tokenizer, and data collator.

        Args:
            model_name (str): The name of the model to be used (valid names from huggingface.co/models)

        Returns:
            model (transformers.modeling_utils.PreTrainedModel): The model to be used
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): The tokenizer to be used
            data_collator (transformers.data.data_collator.DataCollator): The data collator to be used
    """

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="cache")

    # Check if gpu is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Intialize the data collator (automatic padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    return model, tokenizer, data_collator


# Function to preprocess the data
def preprocess_data(tokenizer, train_dataset, val_dataset, test_dataset):
    """
    Preprocess the data by tokenizing the text and summary columns and adding the prefix to the text column.

        Args:
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): The tokenizer to be used
            train_dataset (torch.utils.data.dataset.Dataset): The training dataset
            val_dataset (torch.utils.data.dataset.Dataset): The validation dataset
            test_dataset (torch.utils.data.dataset.Dataset): The test dataset

        Returns:
            train_dataset (torch.utils.data.dataset.Dataset): The preprocessed training dataset
            val_dataset (torch.utils.data.dataset.Dataset): The preprocessed validation dataset
            test_dataset (torch.utils.data.dataset.Dataset): The preprocessed test dataset
    """

    text_column = "content"
    summary_column = "summary"

    prefix = "summarize: "

    # Function to add the prefix to the text column and tokenize the text and summary columns
    def preprocess_function(examples):
        # Add the prefix to the text column
        inputs = [prefix + doc for doc in examples[text_column]]
        # Tokenize the text and summary columns
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        # Tokenize the summary column
        labels = tokenizer(
            text_target=examples[summary_column], max_length=128, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    return train_dataset, val_dataset, test_dataset


# Function to initialize wandb
def initialize_wandb(user, project, model_name):
    """
    Initialize wandb for logging the experiments.

        Args:
            user (str): The user name of the wandb account
            project (str): The name of the project in wandb
            model_name (str): The name of the model to be used (valid names from huggingface.co/models)
    """

    display_name = f"{model_name}-experiment"

    wandb.init(entity=user, project=project, name=display_name)


# Function to create the trainer
def create_trainer(
    model, train_dataset, val_dataset, data_collator, tokenizer, model_name
):
    """
    Create the trainer for training the model.

        Args:
            model (transformers.modeling_utils.PreTrainedModel): The model to be used
            train_dataset (torch.utils.data.dataset.Dataset): The training dataset
            val_dataset (torch.utils.data.dataset.Dataset): The validation dataset
            data_collator (transformers.data.data_collator.DataCollator): The data collator to be used
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): The tokenizer to be used
            model_name (str): The name of the model to be used (valid names from huggingface.co/models)

        Returns:
            trainer (transformers.trainer_utils.Trainer): The trainer for training the model
    """

    # Creating the metric function
    def compute_metrics(eval_pred):
        """
        Compute rouge scores for the generated summaries.

            Args:
                eval_pred (transformers.trainer_utils.EvalPrediction): The predictions and labels

            Returns:
                result (dict): The rouge scores
        """

        metric = evaluate.load("rouge")

        predictions, labels = eval_pred

        # Decode the predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Split the predictions and labels into sentences
        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]

        # Compute the rouge scores
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        return {k: round(v, 4) for k, v in result.items()}

    # Initialize the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-reddit",  # The output directory
        evaluation_strategy="steps",  # Evaluate after every x steps
        per_device_train_batch_size=8,  # Batch size
        per_device_eval_batch_size=8,  # Batch size
        predict_with_generate=True,  # Set to True for summarization tasks
        logging_steps=4,  # Log every 4 steps
        save_steps=4,  # Save every 16 steps
        eval_steps=4,  # Evaluate after every 4 steps
        # warmup_steps=1,                            # Warmup steps
        max_steps=16,  # Total number of training steps
        save_total_limit=4,  # Number of checkpoints to save
        # fp16=True,                                 # Use mixed precision
        report_to='wandb',  # Report to wandb
        load_best_model_at_end=True,  # Load the best model at the end of training
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer


# Run the training for each model
if __name__ == "__main__":
    # List of models to be used
    models = ['t5-small', 'facebook/bart-base']

    # Training loop over the models
    for model_name in models:
        # Load the data and the model
        train_dataset, val_dataset, test_dataset = load_data()
        model, tokenizer, data_collator = load_model(model_name)
        train_dataset, val_dataset, test_dataset = preprocess_data(
            tokenizer, train_dataset, val_dataset, test_dataset
        )

        # Initialize wandb
        user = "npogeant"
        project = "reddit_text_summarization"
        initialize_wandb(user, project, model_name)

        # Create the trainer
        trainer = create_trainer(
            model, train_dataset, val_dataset, data_collator, tokenizer, model_name
        )

        # Start the training
        trainer.train()
