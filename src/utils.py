import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from my_classes import *

def print_model_results(log_dir: str, patience: int, max_epochs: int, model_type: str):
    """
    Unifies functions to process log files, print top results, and return best parameters
    for different model types (e.g., 'classifier_only', 'lora', 'last_layer').

    Args:
        log_dir (str): The directory containing the CSV log files.
        patience (int): The number of epochs to consider for early stopping or to find the best epoch.
        max_epochs (int): The maximum number of epochs defined for the training run.
        model_type (str): The type of model, which dictates how file names are parsed.
                          Expected values: "classifier_only", "lora", or "last_layer".

    Returns:
        dict: A dictionary of the best model's parameters (learning_rate, dropout_rate,
              weight_decay, epoch, and potentially rank/alpha for LoRA) if results are found,
              otherwise None.
    """
    summary = []
    for file in os.listdir(log_dir):
        # Skip non-CSV files
        if not file.endswith(".csv"):
            continue

        path = os.path.join(log_dir, file)
        df = pd.read_csv(path)

        num_epochs = len(df)

        # Determine the best epoch index based on patience or max_epochs
        if num_epochs < max_epochs:
            best_epoch_idx = max(0, num_epochs - patience)
        else:
            # Look at the last 'patience + 1' rows to find the epoch with the minimum validation loss
            last_rows = df.tail(patience + 1)
            best_epoch_idx = last_rows['val_loss'].idxmin()

        # Retrieve the data for the best epoch
        best_row = df.loc[best_epoch_idx]

        base_name = file.replace(".csv", "")
        params = {}
        try:
            parts = base_name.split("_")
            # Parse parameters from filename based on model type
            if model_type == "classifier_only":
                if len(parts) != 4:
                    print(f"Warning: Filename '{file}' for 'classifier_only' model is not in expected format (e.g., 'model_lr_dropout_wd'). Skipping.")
                    continue
                _, lr, dropout, wd = parts
                params["learning_rate"] = float(lr)
                params["dropout_rate"] = float(dropout)
                params["weight_decay"] = float(wd)
            elif model_type == "lora":
                if len(parts) != 6:
                    print(f"Warning: Filename '{file}' for 'lora' model is not in expected format (e.g., 'model_lr_rank_alpha_dropout_wd'). Skipping.")
                    continue
                _, lr, rank, alpha, dropout, wd = parts
                params["learning_rate"] = float(lr) # Added back for consistency in summary, though not always returned in best_params for lora
                params["rank"] = int(rank)
                params["alpha"] = int(alpha)
                params["dropout_rate"] = float(dropout)
                params["weight_decay"] = float(wd)
            elif model_type == "last_layer":
                if len(parts) != 3:
                    print(f"Warning: Filename '{file}' for 'last_layer' model is not in expected format (e.g., 'model_lr_wd'). Skipping.")
                    continue
                _, lr, wd = parts
                params["learning_rate"] = float(lr)
                params["dropout_rate"] = 0.1 # Hardcoded as per the original last_layer function
                params["weight_decay"] = float(wd)
            else:
                print(f"Error: Unsupported model type '{model_type}'. Skipping file '{file}'.")
                continue
        except ValueError as e:
            print(f"Error parsing numerical values from filename '{file}': {e}. Skipping.")
            continue

        # Combine parsed parameters with performance metrics
        row_summary = {
            "epoch": best_row["epoch"] if "epoch" in best_row else best_epoch_idx,
            "train_loss": best_row["train_loss"],
            "val_loss": best_row["val_loss"],
            "train_acc": best_row.get("train_accuracy", None),
            "val_acc": best_row.get("val_accuracy", None),
        }
        row_summary.update(params)
        summary.append(row_summary)

    # If no valid results were processed, return None
    if not summary:
        print(f"No valid results found in '{log_dir}' for model type '{model_type}'.")
        return None

    results_df = pd.DataFrame(summary)
    results_df = results_df.sort_values(by="val_loss", ascending=True)

    # Reorder columns for display based on model type
    column_order_map = {
        "classifier_only": ["learning_rate", "dropout_rate", "weight_decay", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"],
        "lora": ["learning_rate", "rank", "alpha", "dropout_rate", "weight_decay", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"],
        "last_layer": ["learning_rate", "dropout_rate", "weight_decay", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    }
    # Filter columns that actually exist in the DataFrame
    if model_type in column_order_map:
        existing_columns = [col for col in column_order_map[model_type] if col in results_df.columns]
        results_df = results_df[existing_columns]
    else: # Fallback for unknown model types, though already handled above
        results_df = results_df[list(results_df.columns)]


    # Apply specific formatting to numeric columns if they exist
    if "learning_rate" in results_df.columns:
        results_df["learning_rate"] = results_df["learning_rate"].apply(lambda x: f"{x:.0e}" if x < 0.001 else f"{x:.1e}")
    if "weight_decay" in results_df.columns:
        results_df["weight_decay"] = results_df["weight_decay"].apply(lambda x: f"{x:.0e}" if x < 0.001 else f"{x:.1e}")
    if "epoch" in results_df.columns:
        results_df["epoch"] = results_df["epoch"].astype(int)

    # Print only the first 5 rows of the results table
    pd.set_option("display.max_colwidth", None)
    print("\n--- Top 5 Results ---")
    print(results_df.head(5).to_string(index=False))
    print("---------------------\n")

    # Prepare and return the parameters of the best model (first row in the sorted DataFrame)
    if not results_df.empty:
        best_params_columns = {
            "classifier_only": ["learning_rate", "dropout_rate", "weight_decay", "epoch"],
            "lora": ["learning_rate", "rank", "alpha", "dropout_rate", "weight_decay", "epoch"],
            "last_layer": ["learning_rate", "weight_decay", "epoch"]
        }
        params_to_return = [col for col in best_params_columns.get(model_type, []) if col in results_df.columns]
        best_params = results_df.iloc[0][params_to_return].to_dict()

        # Convert back formatted scientific notation to float for actual parameter use if needed
        for key, value in best_params.items():
            if isinstance(value, str) and 'e' in value:
                try:
                    best_params[key] = float(value)
                except ValueError:
                    pass # Keep as string if conversion fails

        return best_params
    else:
        return None




def evaluate_model_on_test_set(task: str, loss: str, model_type: str, lr: float, wd: float, dropout: float = None, rank: int = None, alpha: int = None):
    """
    Performs evaluation on a test set for different model types (e.g., 'classifier_only', 'lora', 'last_layer').

    Args:
        task (str): The specific task (e.g., "Sheldon_Penny", "Sheldon_Leonard").
        loss (str): The type of loss function used (e.g., "CrossEntropyLoss").
        model_type (str): The type of model to evaluate. Expected values: "classifier_only", "lora", or "last_layer".
        lr (float): Learning rate used for the model.
        wd (float): Weight decay used for the model.
        dropout (float, optional): Dropout rate. For 'classifier_only' it's the classifier dropout.
                                   For 'lora' it's the LoRA dropout (classifier still uses 0.1).
                                   Not directly used for 'last_layer's classifier (which uses 0.1). Defaults to None.
        rank (int, optional): LoRA rank. Required for 'lora' model_type. Defaults to None.
        alpha (int, optional): LoRA alpha. Required for 'lora' model_type. Defaults to None.

    Raises:
        ValueError: If required parameters for a specific model_type are missing or
                    if an unsupported model_type is provided.
        FileNotFoundError: If the model file cannot be loaded from the expected path.
    """

    # --- 1. Data Loading and Preprocessing ---
    # Load the dataframe containing embeddings or text data
    df = pd.read_pickle("../data/processed/sbert_mini_embeddings.pkl")

    # Filter DataFrame based on the specified task (e.g., character pairs)
    if task == "Sheldon_Penny":
        df = df[df['Person'].isin(['Sheldon', 'Penny'])]
    elif task == "Sheldon_Leonard":
        df = df[df['Person'].isin(['Sheldon', 'Leonard'])]

    # Encode labels (Person names) into numerical format
    label_encoder = LabelEncoder()
    y_all_encoded = label_encoder.fit_transform(df["Person"])
    all_classes = np.unique(y_all_encoded)
    num_classes = len(all_classes)
    # print(f"Number of classes: {num_classes}")

    
    # --- 2. Dataset Preparation ---
    test_loader = None
    tokenizer = None
    base_model = None
    model_load_path = "" # To store the path to the saved model state dict

    # Prepare data, tokenizer, and base model based on model_type
    if model_type == "classifier_only":
        # 'classifier_only' model uses pre-computed SBERT embeddings directly
        X = np.stack(df["Embedding"].values)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y_all_encoded, dtype=torch.long)

        # Split data into train, val, test sets. Only test set is used for evaluation.
        _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        test_dataset = EmbeddingDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        if dropout is None:
            raise ValueError("For 'classifier_only' model_type, 'dropout' must be provided.")
        model_load_path = f"../models/classifier_only/{task}/{loss}/best_classifier_{lr}_{dropout}_{wd}.pt"

    elif model_type in ["lora", "last_layer"]:
        # 'lora' and 'last_layer' models use raw text and fine-tune SBERT or its head
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        X_texts = df["Said"].tolist()
        y = y_all_encoded # This is already a numpy array

        # Split data into train, val, test sets (using raw texts for tokenization later)
        _, X_temp_texts, _, y_temp = train_test_split(X_texts, y, test_size=0.2, stratify=y, random_state=42)
        _, X_test_texts, _, y_test = train_test_split(X_temp_texts, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        test_dataset = TextDataset(X_test_texts, y_test, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32)

        base_model = AutoModel.from_pretrained(model_name)

        if model_type == "lora":
            if rank is None or alpha is None or dropout is None:
                raise ValueError("For 'lora' model_type, 'rank', 'alpha', and 'dropout' (for LoRA config) must be provided.")
            
            # Configure and apply LoRA to the base model
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=["query", "key", "value"],
                lora_dropout=dropout, # This dropout is specific to LoRA adapters
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            base_model = get_peft_model(base_model, lora_config)
            model_load_path = f"../models/LoRA/{task}/{loss}/lora_{lr}_{rank}_{alpha}_{dropout}_{wd}.pt"
        
        elif model_type == "last_layer":
            # For 'last_layer', the base_model is used as is (frozen)
            model_load_path = f"../models/last_layer/{task}/{loss}/last_layer_{lr}_{wd}.pt"
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose from 'classifier_only', 'lora', or 'last_layer'.")


    # --- 3. Model Definition ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    # Instantiate the correct model type
    if model_type == "classifier_only":
        model = EmbeddingClassifier(input_dim=384, num_classes=num_classes, dropout_rate=dropout).to(device)
    elif model_type in ["lora", "last_layer"]:
        # Classifier dropout rate is hardcoded to 0.1 in the original LoRA/last_layer functions
        model = SBERTWithClassifier(base_model, num_classes=num_classes, dropout_rate=0.1).to(device)
    
    # Load the saved model's state dictionary
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_load_path}'. Please ensure the path and filename are correct.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model from '{model_load_path}': {e}")
        return

    model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)

    all_preds = []
    all_labels = []

    # --- 4. Evaluation Loop ---
    print(f"\nStarting evaluation on test set for {model_type.capitalize()} model...")
    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch data to the correct device
            labels = batch["label"].to(device) if model_type == "classifier_only" else batch["labels"].to(device)

            outputs = None
            if model_type == "classifier_only":
                inputs = batch["embedding"].to(device)
                outputs = model(inputs)
            else: # 'lora' or 'last_layer'
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Reporting Results ---
    # Generate and display Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix on Test Set")
    plt.show()

    # Print Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))