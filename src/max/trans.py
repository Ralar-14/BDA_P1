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
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import ConfusionMatrixDisplay

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

def plot_class_balance(df):
    # Generar gráfica de balanceo de clases
    class_counts = df['label'].value_counts()
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Distribución de Clases (Explicit vs No Explicit)')
    plt.xlabel('Clases')
    plt.ylabel('Cantidad')
    plt.xticks(ticks=[0, 1], labels=['No Explicit', 'Explicit'], rotation=0)
    plt.show()

def balance_classes(df):
    # Balancear las clases mediante sobremuestreo de la clase minoritaria
    explicit = df[df['label'] == 1]
    non_explicit = df[df['label'] == 0]

    # Sobremuestrear la clase minoritaria para igualar el tamaño de la clase mayoritaria
    if len(explicit) < len(non_explicit):
        explicit = resample(explicit, replace=True, n_samples=len(non_explicit), random_state=42)
    else:
        non_explicit = resample(non_explicit, replace=True, n_samples=len(explicit), random_state=42)

    return pd.concat([explicit, non_explicit])

def transformerPipeline():
    # ---------------------------
    # 1. Load and Prepare the Data
    # ---------------------------
    # Load the parquet file into a pandas DataFrame (update the file path as necessary).
    df = pd.read_parquet("C:/Users/maxmg/Documents/GitHub/BDA_P1/src/max/data/exploitation_zone/explicit_prediction")

    print(df.columns)
    
    # Select only the two columns of interest ("lyrics" and "explicit") and rename "explicit" to "label".
    df = df[["song_lyrics", "explicit"]].rename(columns={"explicit": "label"})
    
    # Convert labels to numeric (int) to avoid the "not implemented for 'Bool'" error
    df["label"] = df["label"].astype(int)

    # Generar gráfica de balanceo de clases antes del balanceo
    plot_class_balance(df)

    # Convert the pandas DataFrame to a Hugging Face Dataset.
    dataset = Dataset.from_pandas(df)

    # ---------------------------
    # 2. Data Partitioning
    # ---------------------------
    # Split the dataset into training (70%), validation (15%), and test (15%) sets.
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = split_dataset["train"]
    validation_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    validation_dataset = validation_test_split["train"]
    test_dataset = validation_test_split["test"]

    # Asegurarse de que el conjunto de validación tiene ejemplos de ambas clases
    validation_labels = validation_dataset["label"]
    assert 0 in validation_labels and 1 in validation_labels, "El conjunto de validación debe contener ejemplos de ambas clases."

    # Balancear solo el conjunto de entrenamiento
    train_df = train_dataset.to_pandas()
    train_df = balance_classes(train_df)
    train_dataset = Dataset.from_pandas(train_df)

    # Generar gráfica de balanceo de clases después del balanceo
    plot_class_balance(train_df)

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
        return tokenizer(examples["song_lyrics"], truncation=True)

    # Apply the tokenizer to both training and evaluation datasets (batched for speed).
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = test_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)

    # Set the dataset format for PyTorch (only include necessary columns).
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Use a data collator to dynamically pad the inputs in each batch.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------------------------
    # 4. Model Preparation
    # ---------------------------
    # Load a pre-trained DistilBERT model for sequence classification with 2 labels.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        problem_type="single_label_classification"  # Especificar el tipo de problema
    )

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
        learning_rate=2e-5,                      # Learning rate for the optimizer.
        per_device_train_batch_size=16,          # Batch size per GPU/CPU during training.
        per_device_eval_batch_size=16,           # Batch size for evaluation.
        num_train_epochs=3,                      # Total number of training epochs.
        weight_decay=0.01,                       # Weight decay to reduce overfitting.
        logging_dir="./logs",                    # Directory to store logs.
        logging_steps=50,                        # Log metrics every 50 steps.
        load_best_model_at_end=True,             # Save the best model during training.
        fp16=True,                               # Enable mixed precision training for faster performance on GPU.
        eval_strategy="steps",    # Evaluar cada ciertos steps
        eval_steps=50,                  # Evaluar cada 50 pasos (mismo que logging_steps)
        save_strategy="steps",          # Guardar cada ciertos steps
        save_steps=50,                  # Guardar cada 50 pasos
        metric_for_best_model="f1",     # Usar F1 como métrica para "el mejor modelo"
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

    # Mostrar matriz de confusión al final del entrenamiento
    eval_predictions = trainer.predict(eval_dataset)
    y_true = eval_predictions.label_ids
    y_pred = np.argmax(eval_predictions.predictions, axis=1)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['No Explicit', 'Explicit'])
    plt.title('Matriz de Confusión')
    plt.show()

    # Convertir índices a int estándar para evitar errores
    random_indices = [int(idx) for idx in np.random.choice(len(eval_dataset), size=5, replace=False)]
    print("\nEjemplos aleatorios de predicciones:")
    for idx in random_indices:
        example = eval_dataset[idx]
        input_ids = example['input_ids']
        real_label = example['label']
        pred_label = y_pred[idx]

        # Decodificar las letras desde los input_ids
        lyrics = tokenizer.decode(input_ids, skip_special_tokens=True)

        print(f"\nLetra: {lyrics}")
        print(f"Etiqueta real: {'Explicit' if real_label == 1 else 'No Explicit'}")
        print(f"Etiqueta predicha: {'Explicit' if pred_label == 1 else 'No Explicit'}")

    # (Optional) Save the fine-tuned model and tokenizer for later use.
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")


if __name__ == "__main__":
    transformerPipeline()
