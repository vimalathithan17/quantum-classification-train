import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Import the centralized logger
from logging_utils import log

# --- Configuration ---
# The directory containing all your source data files (configurable via env var SOURCE_DIR)
SOURCE_DIR = os.environ.get('SOURCE_DIR', 'final_processed_datasets')
# The directory where you will save your final models and outputs
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'master_label_encoder')
# The name of the column containing the categorical labels
LABEL_COL = 'class'

os.makedirs(OUTPUT_DIR, exist_ok=True)
MASTER_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'label_encoder.joblib')

def safe_load_parquet_labels(file_path, column_name):
    """Safely loads only the label column from a parquet file."""
    try:
        limit = 1 * 1024**3
        return pd.read_parquet(
            file_path,
            columns=[column_name],
            thrift_string_size_limit=limit,
            thrift_container_size_limit=limit
        )
    except Exception as e:
        log.warning(f"Could not read {file_path}. Skipping. Reason: {e}")
        return None

def create_master_encoder():
    """
    Finds all unique class labels across all data files and fits a single
    LabelEncoder on them, saving it to a file.
    """
    log.info("--- Creating Master Label Encoder ---")
    log.info(f"Reading from source directory: '{SOURCE_DIR}'")
    
    all_unique_labels = set()
    files_to_process = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.parquet')]

    # --- Step 1: Collect all unique labels from all files ---
    for filename in files_to_process:
        file_path = os.path.join(SOURCE_DIR, filename)
        labels_df = safe_load_parquet_labels(file_path, LABEL_COL)
        
        if labels_df is not None:
            unique_in_file = labels_df[LABEL_COL].unique()
            all_unique_labels.update(unique_in_file)
            log.info(f"  - Found labels in {filename}: {unique_in_file}")

    if not all_unique_labels:
        log.error("No class labels found. Cannot create an encoder.")
        return

    # --- Step 2: Fit the LabelEncoder on the complete set of unique labels ---
    master_encoder = LabelEncoder()
    # Sort the labels to ensure a consistent, deterministic mapping
    sorted_labels = sorted(list(all_unique_labels))
    master_encoder.fit(sorted_labels)
    
    n_classes = len(master_encoder.classes_)
    log.info(f"Found a total of {n_classes} unique classes across all files: {list(master_encoder.classes_)}")

    # --- Step 3: Save the fitted encoder ---
    joblib.dump(master_encoder, MASTER_ENCODER_PATH)
    log.info(f"Success! Master Label Encoder saved to '{MASTER_ENCODER_PATH}'")
    log.info("This encoder can now be loaded and used by all other scripts.")
    log.info("Example Mapping:")
    log.info(dict(zip(master_encoder.classes_, master_encoder.transform(master_encoder.classes_))))

if __name__ == "__main__":
    create_master_encoder()
