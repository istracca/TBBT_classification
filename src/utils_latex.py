import os
import pandas as pd

def print_results_latex(log_dir: str, patience: int, max_epochs: int, model_type: str):
    """
    Unifies functions for printing results in LaTeX format for different model types.

    Args:
        log_dir (str): The directory containing CSV log files.
        patience (int): The number of epochs for early stopping or to determine the best epoch.
        max_epochs (int): The maximum number of epochs for training.
        model_type (str): The type of model ("classifier_only", "lora", or "last_layer").
    """
    summary = []
    # Iterate over files in the specified log directory
    for file in os.listdir(log_dir):
        # Skip files that are not CSVs
        if not file.endswith(".csv"):
            continue

        path = os.path.join(log_dir, file)
        df = pd.read_csv(path)

        num_epochs = len(df)

        # Determine the best epoch index based on patience and max_epochs
        if num_epochs < max_epochs:
            best_epoch_idx = max(0, num_epochs - patience)
        else:
            # Look at the last 'patience + 1' rows to find the best validation loss
            last_rows = df.tail(patience + 1)
            best_epoch_idx = last_rows['val_loss'].idxmin()

        # Get the row corresponding to the best epoch
        best_row = df.loc[best_epoch_idx]

        base_name = file.replace(".csv", "")
        params = {}
        try:
            parts = base_name.split("_")
            # Parse file name based on model type
            if model_type == "classifier_only":
                if len(parts) != 4:
                    print(f"Unrecognized file name for 'classifier_only' model: {file}. Requires 4 parts.")
                    continue
                _, lr, dropout, wd = parts
                params["learning_rate"] = float(lr)
                params["dropout_rate"] = float(dropout)
                params["weight_decay"] = float(wd)
            elif model_type == "lora":
                if len(parts) != 6:
                    print(f"Unrecognized file name for 'lora' model: {file}. Requires 6 parts.")
                    continue
                _, lr, rank, alpha, dropout, wd = parts
                params["rank"] = int(rank)
                params["alpha"] = int(alpha)
                params["weight_decay"] = float(wd)
            elif model_type == "last_layer":
                if len(parts) != 3:
                    print(f"Unrecognized file name for 'last_layer' model: {file}. Requires 3 parts.")
                    continue
                _, lr, wd = parts
                params["weight_decay"] = float(wd)
            else:
                print(f"Unsupported model type: {model_type}")
                continue
        except ValueError as e:
            print(f"Error parsing filename {file}: {e}")
            continue

        # Compile common metrics and model-specific parameters
        row_summary = {
            "epoch": best_row["epoch"] if "epoch" in best_row else best_epoch_idx,
            "train_loss": best_row["train_loss"],
            "val_loss": best_row["val_loss"],
            "train_acc": best_row.get("train_accuracy", None),
            "val_acc": best_row.get("val_accuracy", None),
        }
        row_summary.update(params) # Add model-specific parameters
        summary.append(row_summary)

    # If no valid CSV files were found, print a message and exit
    if not summary:
        print(f"No valid CSV files found in directory {log_dir} for model type {model_type}.")
        return

    results_df = pd.DataFrame(summary)

    # Reorder columns based on model type for better readability
    if model_type == "classifier_only":
        column_order = ["learning_rate", "dropout_rate", "weight_decay", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    elif model_type == "lora":
        column_order = ["rank", "alpha", "weight_decay", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    elif model_type == "last_layer":
        column_order = ["weight_decay", "epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    else:
        column_order = list(results_df.columns) # Keep original order if type is not recognized

    # Filter for columns that actually exist in the DataFrame
    existing_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[existing_columns]

    # Sort results by validation loss
    results_df = results_df.sort_values(by="val_loss", ascending=True)

    # Format specific numeric columns
    if "learning_rate" in results_df.columns:
        results_df["learning_rate"] = results_df["learning_rate"].apply(lambda x: f"{x:.0e}" if x < 0.001 else f"{x:.1e}")
    if "weight_decay" in results_df.columns:
        results_df["weight_decay"] = results_df["weight_decay"].apply(lambda x: f"{x:.0e}" if x < 0.001 else f"{x:.1e}")
    if "epoch" in results_df.columns:
        results_df["epoch"] = results_df["epoch"].astype(int)

    # Set pandas display option for full column width
    pd.set_option("display.max_colwidth", None)

    # Print the DataFrame in LaTeX format
    print(results_df.to_latex(index=False, float_format="%.4f", escape=False, column_format="l" * len(results_df.columns)))