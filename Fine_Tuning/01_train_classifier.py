import os
import torch
import evaluate
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import (AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    AutoConfig)

from typing import Dict, Any, Optional


class TransformerTextClassifier:
    """
    LLM model for text classification on the AG News dataset
    """

    def __init__(self,
                 model_name: str = "xlnet-base-cased",
                 dataset_name: str = "ag_news",
                 num_labels: int = 4,
                 max_length: int = 128,
                 output_dir: str = None,
                 log_dir: str = None):
        """
        Initializes the classifier with configuration parameters.

        Args:
            model_name (str): The name of the pre-trained model from Hugging Face.
            dataset_name (str): The name of the dataset from Hugging Face.
            num_labels (int): The number of unique labels in the dataset.
            max_length (int): The maximum sequence length for tokenization.
            output_dir (str): The directory to save the fine-tuned model.
            log_dir (str): The directory to save training logs.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Get current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Define the base output folder
        base_dir = "outputs"
        folder_name = timestamp + "_" + model_name.replace("/", "-") + "_finetuning_01"
        run_dir = os.path.join(base_dir, folder_name)

        # Generate output paths
        self.output_dir = os.path.join(run_dir, "model")
        self.log_dir = os.path.join(run_dir, "logs")

        # Label mappings for AG News (World, Sports, Business, Sci/Tech)
        self.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        self.label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}

        # These will be loaded by the methods
        self.dataset_dict = None
        self.tokenized_datasets = None
        self.tokenizer = None
        self.model = None
        self.trainer = None

        # Load the metric
        self.metric = evaluate.load("accuracy")

        # Check device status
        self._check_device()

    def _check_device(self):
        """Checks and prints the compute device (GPU/CPU)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[System] Using device: {device.upper()}")
        if device == "cuda":
            print(f"[System] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[System] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("[System] WARNING: Using CPU. Training will be slow.")

    def load_data(self):
        """Loads the dataset from Hugging Face."""
        print(f"Loading dataset: {self.dataset_name}...")
        self.dataset_dict = load_dataset(self.dataset_name)
        # --- Print Informative Summary ---
        print("\n" + "=" * 40)
        print(f"DATASET SUMMARY: {self.dataset_name}")
        print("=" * 40)

        # Print train/test split details
        for split, dataset in self.dataset_dict.items():
            print(f"Split: {split:<10} | Rows: {dataset.num_rows:<8} | Columns: {dataset.column_names}")

        # Print a sample example
        print("-" * 40)
        print("SAMPLE EXAMPLE (from 'train' split):")
        sample = self.dataset_dict["train"][0]

        # Truncate text if it's too long for display
        text_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']

        print(f"Text:  {text_preview}")
        print(f"Label: {sample['label']} -> {self.id2label[sample['label']]}")
        print("=" * 40 + "\n")

    def load_tokenizer(self):
        """Loads the pre-trained tokenizer."""
        print(f"Loading tokenizer for model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _tokenize_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal tokenization function to be applied to a batch of examples.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")

        return self.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

    def preprocess_data(self):
        """Tokenizes the entire dataset using the map function."""
        if not self.dataset_dict:
            self.load_data()
        if not self.tokenizer:
            self.load_tokenizer()

        small_train = self.dataset_dict["train"].shuffle(seed=42).select(range(5000))
        small_test = self.dataset_dict["test"].shuffle(seed=42).select(range(500))

        self.dataset_dict["train"] = small_train
        self.dataset_dict["test"] = small_test

        print("Tokenizing dataset ...")
        self.tokenized_datasets = self.dataset_dict.map(
            self._tokenize_batch,
            batched=True)

    def load_model(self):
        """Loads the pre-trained model with the correct classification head."""
        print(f"Loading model: {self.model_name} for sequence classification...")

        # Load config and update labels
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config)

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Computes accuracy from evaluation predictions."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self, training_args_dict: Optional[Dict[str, Any]] = None):
        """
        Initializes the Trainer and starts the fine-tuning process.

        Args:
            training_args_dict (Optional[Dict]): A dictionary of training arguments.
                                                 If None, sensible defaults are used.
        """
        if not self.tokenized_datasets or not self.model:
            raise ValueError("Data and model must be loaded. Call preprocess_data() and load_model() first.")

        if training_args_dict is None:
            # Define default training arguments if none are provided
            training_args_dict = {
                "output_dir": self.log_dir,
                "eval_strategy": "epoch",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 64,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "logging_dir": f"{self.log_dir}/logs",
                "logging_steps": 50,
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "accuracy",
            }

        training_args = TrainingArguments(**training_args_dict)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer,
        )

        # --- Start Training ---
        print("\nStarting model training...")
        self.trainer.train()

        # --- Evaluate ---
        print("\nEvaluating model on test set...")
        eval_results = self.trainer.evaluate()
        print(f"\nEvaluation Results:\n{eval_results}")

        # --- Save Model ---
        print(f"\nSaving best model to: {self.output_dir}")
        self.trainer.save_model(self.output_dir)

    def predict(self, text: str, top_k: Optional[int] = None) -> Any:
        """
        Runs inference on a single piece of text using the saved,
        fine-tuned model.

        Args:
            text (str): The input text to classify.
            top_k (Optional[int]): If set, returns the top_k predictions.

        Returns:
            Any: The prediction result from the pipeline.
        """
        print(f"\nLoading fine-tuned model from {self.output_dir}...")
        if not os.path.isdir(self.output_dir):
            raise EnvironmentError(
                f"Model directory not found at {self.output_dir}. "
                "Did you run the train() method first?"
            )

        # Load the fine-tuned model and tokenizer into a pipeline
        clf = pipeline(
            "text-classification",
            model=self.output_dir,
            tokenizer=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )

        print(f"Classifying text:\n'{text}'\n")
        result = clf(text, top_k=top_k)
        print("Model Prediction(s):")
        print(result)
        return result

    def run_full_workflow(self):
        """Runs the complete workflow from data loading to training and saving."""
        self.load_data()
        self.preprocess_data()
        self.load_model()
        self.train()


# --- Main block ---
if __name__ == "__main__":

    # Defines the available models
    MODELS = {
        "xlnet": "xlnet-base-cased",
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-uncased",
        "deberta": "microsoft/deberta-base"
    }

    SELECTED_MODEL = "distilbert"  # <--- CHANGE THIS STRING to switch models

    print(f"Selected Model: {SELECTED_MODEL.upper()}")

    model_id = MODELS.get(SELECTED_MODEL)
    if not model_id:
        raise ValueError(f"Model '{SELECTED_MODEL}' not found in options.")

    # --- 1. Initialize the Classifier ---
    classifier = TransformerTextClassifier(
        model_name= model_id,
        dataset_name="ag_news",
        num_labels=4,
        max_length=128
    )

    # --- 2. Run the Full Training Workflow ---
    # Load data, tokenize, load the model, train, evaluate, and save.
    classifier.run_full_workflow()

    # --- 3. Run a Test Prediction ---
    # Load the model you just saved from `output_dir`
    test_text = (
        "The new graphics card from Nvidia is a major breakthrough, "
        "driving stock prices up. Analysts predict it will also "
        "dominate the e-sports scene."
    )

    # Test with top_k=None to see all label scores
    classifier.predict(test_text, top_k=None)