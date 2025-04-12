import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Define the compute_metrics function used during evaluation.
def compute_metrics(eval_pred):
    # Unpack predictions and labels from the evaluation output.
    logits, labels = eval_pred
    # Compute predictions by taking the argmax of the logits.
    predictions = np.argmax(logits, axis=-1)
    # Calculate accuracy, precision, recall, and F1 score using sklearn.
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def transformerPipeline():
    # ---------------------------
    # 1. Load and Prepare the Data
    # ---------------------------
    # Load the parquet file into a pandas DataFrame (update the file path as necessary).
    df = pd.read_parquet("path/to/your/datafile.parquet")
    
    # Select only the two columns of interest ("lyrics" and "explicit") and rename "explicit" to "label".
    df = df[["lyrics", "explicit"]].rename(columns={"explicit": "label"})
    
    # (Optional) Convert labels to numeric if they are not already (e.g., boolean to int).
    # df["label"] = df["label"].astype(int)
    
    # Convert the pandas DataFrame to a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df)

    # ---------------------------
    # 2. Data Partitioning
    # ---------------------------
    # Split the dataset into training (80%) and evaluation (20%) sets.
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # ---------------------------
    # 3. Preprocessing and Tokenization
    # ---------------------------
    # Define the DistilBERT model name for English text.
    model_name = "distilbert-base-uncased"

    # Load the corresponding tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a function to tokenize the lyrics.
    def tokenize_function(examples):
        # Tokenize the "lyrics" field with truncation enabled.
        return tokenizer(examples["lyrics"], truncation=True)

    # Apply the tokenizer to both training and evaluation datasets (batched for speed).
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set the dataset format for PyTorch (only include necessary columns).
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Use a data collator to dynamically pad the inputs in each batch.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------------------------
    # 4. Model Preparation
    # ---------------------------
    # Load a pre-trained DistilBERT model for sequence classification with 2 labels.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Freeze the deeper layers of DistilBERT to reduce training time.
    # This freezes all parameters in the DistilBERT encoder, leaving only the classification head trainable.
    for param in model.distilbert.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ---------------------------
    # 5. Training Arguments and Trainer Setup
    # ---------------------------
    # Define training hyperparameters and other settings.
    training_args = TrainingArguments(
        output_dir="./results",                  # Directory to store model outputs and checkpoints.
        evaluation_strategy="epoch",             # Evaluate at the end of each epoch.
        learning_rate=2e-5,                      # Learning rate for the optimizer.
        per_device_train_batch_size=16,          # Batch size per GPU/CPU during training.
        per_device_eval_batch_size=16,           # Batch size for evaluation.
        num_train_epochs=3,                      # Total number of training epochs.
        weight_decay=0.01,                       # Weight decay to reduce overfitting.
        logging_dir="./logs",                    # Directory to store logs.
        logging_steps=50,                        # Log metrics every 50 steps.
        load_best_model_at_end=True,             # Save the best model during training.
        fp16=True,                               # Enable mixed precision training for faster performance on GPU.
    )

    # Instantiate the Trainer object for fine-tuning.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---------------------------
    # 6. Training and Evaluation
    # ---------------------------
    # (Optional) Move the model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Start training the model.
    trainer.train()

    # Evaluate the model on the evaluation dataset.
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    print(eval_results)

    # (Optional) Save the fine-tuned model and tokenizer for later use.
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")


if __name__ == "__main__":
    transformerPipeline()
