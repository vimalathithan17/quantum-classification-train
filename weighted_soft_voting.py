import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import centralized logger and assembly utilities
from logging_utils import log
from metalearner import (
    RANDOM_STATE,
    assemble_meta_data,
    ensure_writable_results_dir,
    set_seed,
)
from utils.metrics_utils import compute_metrics

# Shared local constants
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')

def apply_soft_voting(X_df, weights_dict, n_classes):
    """
    Applies hardcoded mathematical weights to the base learner probabilities.
    Returns the blended probabilities and the final discrete predictions.
    """
    # Initialize an array of zeros to hold the final blended probabilities
    blended_probs = np.zeros((len(X_df), n_classes))
    
    for mod, weight in weights_dict.items():
        if weight <= 0:
            continue
            
        # Find the probability columns for this specific modality
        mod_cols = [col for col in X_df.columns if f'pred_{mod}' in str(col)]
        
        if len(mod_cols) == n_classes:
            # Multiply this model's probabilities by its assigned weight and add to the total
            blended_probs += X_df[mod_cols].values * weight
        elif len(mod_cols) > 0:
            log.warning(f"Found {len(mod_cols)} columns for {mod}, but expected {n_classes}. Skipping.")
        else:
            log.warning(f"No columns found for modality: {mod}. Check your spelling!")

    # The final prediction is just the class (column index) with the highest blended sum
    final_predictions = np.argmax(blended_probs, axis=1)
    
    return blended_probs, final_predictions

def main():
    parser = argparse.ArgumentParser(
        description="Weighted Soft Voting Meta-Learner (No ML, just Math!)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--preds_dir', nargs='+', required=True,
                          help='Directories with base learner predictions')
    required.add_argument('--indicator_file', type=str, required=True,
                          help='Parquet file with indicator features and labels')
    
    # Base Learner Weights (Defaults set based on confusion matrix analysis)
    weight_args = parser.add_argument_group('Base Learner Weights (Must roughly sum to 1.0)')
    weight_args.add_argument('--w_transformer', type=float, default=0.70)
    weight_args.add_argument('--w_geneexpr', type=float, default=0.15)
    weight_args.add_argument('--w_mirna', type=float, default=0.10)
    weight_args.add_argument('--w_meth', type=float, default=0.05)
    weight_args.add_argument('--w_cnv', type=float, default=0.00, help="Dropped due to noise")
    weight_args.add_argument('--w_prot', type=float, default=0.00, help="Dropped due to noise")
    
    # Logging config
    log_args = parser.add_argument_group('logging')
    log_args.add_argument('--use_wandb', action='store_true', help='Enable W&B experiment tracking')
    log_args.add_argument('--wandb_project', type=str, default='soft_voting_meta')
    log_args.add_argument('--wandb_run_name', type=str, default='manual_weights_run')
                          
    args = parser.parse_args()
    set_seed(RANDOM_STATE)

    # Assemble the weights dictionary
    weights_dict = {
        'Transformer': args.w_transformer,
        'GeneExpr': args.w_geneexpr,
        'miRNA': args.w_mirna,
        'Meth': args.w_meth,
        'CNV': args.w_cnv,
        'Prot': args.w_prot
    }
    
    total_weight = sum(weights_dict.values())
    log.info(f"--- Running Weighted Soft Voting ---")
    log.info(f"Total Weight Sum: {total_weight:.2f}")
    for mod, w in weights_dict.items():
        log.info(f"  -> {mod}: {w:.2f}")

    # Load Data
    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, _ = assemble_meta_data(args.preds_dir, args.indicator_file)
    
    if X_meta_train is None:
        log.critical("Failed to assemble meta-dataset. Exiting.")
        return

    n_classes = len(le.classes_)
    global OUTPUT_DIR
    OUTPUT_DIR = ensure_writable_results_dir(OUTPUT_DIR)

    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=weights_dict)
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False

    # ---------------------------------------------------------
    # 1. Evaluate on the Meta-Training Set
    # ---------------------------------------------------------
    log.info("--- Evaluating on Training Set ---")
    _, train_preds = apply_soft_voting(X_meta_train, weights_dict, n_classes)
    train_metrics = compute_metrics(y_meta_train, train_preds, n_classes)
    
    log.info(f"  -> Train Accuracy:       {train_metrics['accuracy']:.4f}")
    log.info(f"  -> Train Weighted F1:    {train_metrics['f1_weighted']:.4f}")

    # ---------------------------------------------------------
    # 2. Evaluate on the Unseen Test Set
    # ---------------------------------------------------------
    if not X_meta_test.empty and not y_meta_test.empty:
        log.info("--- Evaluating on Final Test Set ---")
        _, test_preds = apply_soft_voting(X_meta_test, weights_dict, n_classes)
        test_metrics = compute_metrics(y_meta_test, test_preds, n_classes)
        
        # Save Test Confusion Matrix
        test_cm = confusion_matrix(y_meta_test, test_preds, labels=list(range(n_classes)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Test Set Confusion Matrix (Soft Voting)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        test_cm_path = os.path.join(OUTPUT_DIR, 'test_confusion_matrix_soft_voting.png')
        plt.savefig(test_cm_path)
        plt.close()
        log.info(f"Saved test confusion matrix diagram to {test_cm_path}")

        # Print all metrics
        log.info(f"  -> Test Accuracy:       {test_metrics['accuracy']:.4f}")
        log.info(f"  -> Test Weighted P:     {test_metrics['precision_weighted']:.4f}")
        log.info(f"  -> Test Weighted R:     {test_metrics['recall_weighted']:.4f}")
        log.info(f"  -> Test Weighted S:     {test_metrics['specificity_weighted']:.4f}")
        log.info(f"  -> Test Weighted F1:    {test_metrics['f1_weighted']:.4f}")
        log.info(f"\nClassification Report:\n{classification_report(y_meta_test, test_preds)}")
        
        if args.use_wandb:
            wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))})

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
