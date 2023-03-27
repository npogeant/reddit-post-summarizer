""" 
This script aims to create a web interface for the Reddit summarization model.
It uses the Gradio library to create the interface and the WandB library to get the model artifact.
"""

import os

import wandb
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()


# Function to get the registered model from wandb
def get_registered_model(model_path):
    """
    Get the registered model from wandb.

        Args:
            model_path (str): The path of the model to be used (valid path from huggingface.co/models)
    """

    existing_path = f"./artifacts/{model_path}"

    # Check if the model is already downloaded
    if os.path.exists(existing_path):
        artifact_dir = existing_path
    else:
        # Initialize wandb
        run = wandb.init(entity='npogeant', project='reddit_text_summarization')
        artifact = run.use_artifact(model_path, type='model')
        artifact_dir = artifact.download()

    return artifact_dir


# Function to summarize the text using a fine-tuned model
def summarize(text):
    """
    Summarize the text using a fine-tuned model.

        Args:
            text (str): The text to be summarized
    """

    # Directory of the model artifact from wandb
    artifact_dir = get_registered_model("model-3ll9iraw:v0")

    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(artifact_dir)
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, cache_dir="cache")

    # Add the prefix to the text
    prefix = "summarize: "
    new_inference = f'{prefix}{text}'

    # Generate the summary
    input = tokenizer(new_inference, return_tensors="pt").input_ids
    outputs = model.generate(input, max_new_tokens=100, do_sample=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary


# Initialize the interface
demo = gr.Interface(
    fn=summarize,
    inputs="text",
    outputs="text",
    title="Reddit Post Summarizer",
    description="Enter a reddit post and get a summary of it.",
)

if __name__ == "__main__":
    demo.launch()
