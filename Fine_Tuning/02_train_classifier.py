import os
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    AutoConfig
)
from typing import Dict, Any, Optional


class RTEClassifier:
    """
    LLM model for Recognizing Textual Entailment (RTE) on the GLUE dataset.
    Refactored from Fine_Tuning_RTE.pdf logic.
    """

    def __init__(self,
                 model_name: str = "distilbert-base-cased",  # [cite: 100]
                 dataset_name: str = "glue",
                 subset_name: str = "rte",  # [cite: 55]
                 num_labels: int = 2,  # [cite: 74]
                 max_length: int = 128,
                 output_dir: str = None):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Get current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Define the base output folder
        base_dir = "outputs"
        folder_name = timestamp + "_" + model_name.replace("/", "-") + "_finetuning_02"
        run_dir = os.path.join(base_dir, folder_name)

        self.output_dir = os.path.join(run_dir, "model")
        self.log_dir = os.path.join(run_dir, "logs")

        # Label mappings for RTE (based on glue/rte features) [cite: 74]
        # 0: entailment, 1: not_entailment
        self.id2label = {0: "entailment", 1: "not_entailment"}
        self.label2id = {"entailment": 0, "not_entailment": 1}

        # Placeholders
        self.dataset_dict = None
        self.tokenized_datasets = None
        self.tokenizer = None
        self.model = None
        self.trainer = None

        self._check_device()

    def _check_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[System] Using device: {device.upper()}")

    def load_data(self):
        """Loads the GLUE/RTE dataset from Hugging Face."""
        print(f"Loading dataset: {self.dataset_name} ({self.subset_name})...")
        self.dataset_dict = load_dataset(self.dataset_name, self.subset_name)

        print("\n" + "=" * 40)
        print(f"DATASET SUMMARY: {self.subset_name.upper()}")
        print("=" * 40)

        # Print sample
        sample = self.dataset_dict["train"][0]
        print(f"Sentence 1: {sample['sentence1']}")
        print(f"Sentence 2: {sample['sentence2']}")
        print(f"Label:      {sample['label']} -> {self.id2label[sample['label']]}")
        print("=" * 40 + "\n")

    def load_tokenizer(self):
        """Loads the pre-trained tokenizer."""
        print(f"Loading tokenizer for model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _tokenize_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes sentence pairs as required by RTE.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded.")

        # Note: We pass BOTH sentence1 and sentence2 to the tokenizer
        return self.tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",  # Padded for consistent batching
            max_length=self.max_length
        )

    def preprocess_data(self):
        """Tokenizes the dataset."""
        if not self.dataset_dict:
            self.load_data()
        if not self.tokenizer:
            self.load_tokenizer()

        print("Tokenizing dataset (Sentence Pairs)...")
        self.tokenized_datasets = self.dataset_dict.map(
            self._tokenize_batch,
            batched=True
        )

    def load_model(self):
        """Loads the model with a 2-class head[cite: 128]."""
        print(f"Loading model: {self.model_name}...")

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Computes Accuracy and F1 Score as per the notebook.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        return {"accuracy": acc, "f1": f1}

    def train(self):
        """
        Runs the training loop with parameters from the notebook.
        """
        if not self.tokenized_datasets or not self.model:
            raise ValueError("Data and model must be loaded first.")

        # PDF Settings: 5 Epochs, Batch Size 16
        training_args = TrainingArguments(
            output_dir=self.log_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,  # [cite: 138]
            per_device_train_batch_size=16,  # [cite: 139]
            per_device_eval_batch_size=64,  # [cite: 140]
            logging_steps=150,  # [cite: 141]
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],  # PDF uses val for eval [cite: 171]
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,  # [cite: 173]
        )

        print("\nStarting training...")
        self.trainer.train()  # [cite: 176]

        print(f"\nSaving model to: {self.output_dir}")
        self.trainer.save_model(self.output_dir)  # [cite: 259]

    def predict(self, text: str, text_pair: str) -> Any:
        """
        Runs prediction on a sentence pair[cite: 263].
        """
        print(f"\nLoading fine-tuned model from {self.output_dir}...")

        clf = pipeline(
            "text-classification",
            model=self.output_dir,
            device=0 if torch.cuda.is_available() else -1
        )

        # Pipeline handles text pairs by passing a dictionary or kwargs
        # In the PDF: p({'text': ..., 'text_pair': ...}) [cite: 263]
        payload = {'text': text, 'text_pair': text_pair}

        print(f"Classifying pair:\n A: {text}\n B: {text_pair}")
        result = clf(payload)

        print("Prediction:", result)
        return result

    def run_full_workflow(self):
        self.load_data()
        self.preprocess_data()
        self.load_model()
        self.train()


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize wrapper with settings from the PDF
    classifier = RTEClassifier(
        model_name="distilbert-base-cased",  # [cite: 100]
        dataset_name="glue",
        subset_name="rte"
    )

    # Train
    classifier.run_full_workflow()

    # Test Prediction (Example from PDF) [cite: 263]
    classifier.predict(
        text="I went to the store",
        text_pair="I am a bird"
    )