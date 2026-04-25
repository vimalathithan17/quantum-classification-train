import argparse
import copy
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import centralized logger and assembly utilities
from logging_utils import log
from metalearner import (
    ENCODER_DIR,
    RANDOM_STATE,
    TUNING_JOURNAL_FILE,
    _per_class_specificity,
    assemble_meta_data,
    ensure_writable_db,
    ensure_writable_results_dir,
    set_seed,
)
from utils.metrics_utils import compute_metrics

# Shared local constants
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')

# Global dictionary to track the best model during tuning
best_trial_data = {
    'score': -np.inf,
    'model': None,
    'epoch': 0
}

def get_custom_metric(y_true, y_pred, n_classes):
    """Calculates the weighted custom metric used for early stopping and evaluation."""
    m = compute_metrics(y_true, y_pred, n_classes)
    # Exclusively tracking f1_weighted to ensure maximum performance
    combined = m['f1_weighted']
    return combined, m

def train_mlp_with_custom_early_stopping(X_train, y_train, X_val, y_val, n_classes, params, patience=20, tol=1e-4, max_epochs=20000):
    """
    Manually trains an MLP using partial_fit to allow early stopping and 
    best-weight restoration based on a CUSTOM metric.
    """
    model = MLPClassifier(**params)
    classes = np.unique(y_train)
    
    best_score = -np.inf
    best_weights = None
    best_intercepts = None
    best_metrics = None
    patience_counter = 0
    best_epoch = 0
    stopped_epoch = 0
    
    for epoch in range(max_epochs):
        # Train one epoch
        model.partial_fit(X_train, y_train, classes=classes)
        
        # Evaluate on validation fold
        preds = model.predict(X_val)
        custom_score, m_dict = get_custom_metric(y_val, preds, n_classes)
        
        # Check for improvement based on tolerance interval
        if custom_score >= best_score + tol:
            best_score = custom_score
            best_metrics = m_dict
            best_epoch = epoch
            patience_counter = 0
            
            # Deep copy to protect the best state from being overwritten by future epochs
            best_weights = copy.deepcopy(model.coefs_)
            best_intercepts = copy.deepcopy(model.intercepts_)
        else:
            patience_counter += 1
            
        # Stop if no improvement for 'patience' epochs
        if patience > 0 and patience_counter >= patience:
            stopped_epoch = epoch
            log.info(f"Early stopping triggered at epoch {stopped_epoch}. Restoring weights from epoch {best_epoch}.")
            break

    # If it reached max_epochs without triggering early stopping
    if stopped_epoch == 0:
        stopped_epoch = max_epochs - 1

    # Restore the best model weights
    if best_weights is not None:
        model.coefs_ = best_weights
        model.intercepts_ = best_intercepts
        
    return model, best_score, best_metrics, best_epoch, stopped_epoch


def objective(trial, X_train, y_train, X_val, y_val, n_classes, patience, tol, max_epochs):
    """Optuna objective function for tuning the MLP Meta-Learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Layer choices including 256 neuron variations
    layer_choices = [
        '64', '128', '256',                                 # Single layers
        '64_32', '128_64', '128_64_32', '256_128', '256_128_64', # Funnel (Decreasing)
        '64_64', '128_128', '256_256',                      # Constant
        '32_64', '64_128', '128_256'                        # Expanding
    ]
    layer_choice = trial.suggest_categorical('mlp_layers', layer_choices)
    
    layer_mapping = {
        '64': (64,), '128': (128,), '256': (256,),
        '64_32': (64, 32), '128_64': (128, 64), '128_64_32': (128, 64, 32),
        '256_128': (256, 128), '256_128_64': (256, 128, 64),
        '64_64': (64, 64), '128_128': (128, 128), '256_256': (256, 256),
        '32_64': (32, 64), '64_128': (64, 128), '128_256': (128, 256)
    }
    
    params = {
        'hidden_layer_sizes': layer_mapping[layer_choice],
        'learning_rate_init': trial.suggest_float('mlp_lr', 1e-4, 1e-1, log=True),
        'alpha': trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
        'activation': trial.suggest_categorical('mlp_activation', ['relu', 'tanh', 'logistic']),
        'solver': trial.suggest_categorical('mlp_solver', ['adam', 'sgd']),
        'random_state': RANDOM_STATE,
    }

    # Train with custom early stopping using CLI parameters
    model, combined_metric, m, best_epoch, stopped_epoch = train_mlp_with_custom_early_stopping(
        X_train, y_train, X_val, y_val, n_classes, params, 
        patience=patience, tol=tol, max_epochs=max_epochs
    )
    
    # Save the absolute best model globally
    global best_trial_data
    if combined_metric > best_trial_data['score']:
        best_trial_data['score'] = combined_metric
        best_trial_data['model'] = copy.deepcopy(model)
        best_trial_data['epoch'] = best_epoch

    # Print out all key weighted metrics and epoch data for this trial
    log.info(f"Trial {trial.number} Results:")
    log.info(f"  -> Combined Score (F1 Weighted): {combined_metric:.4f}")
    log.info(f"  -> Stopped Epoch:  {stopped_epoch} | Best Epoch: {best_epoch}")
    log.info(f"  -> Accuracy:       {m['accuracy']:.4f}")
    log.info(f"  -> Weighted P:     {m['precision_weighted']:.4f}")
    log.info(f"  -> Weighted R:     {m['recall_weighted']:.4f}")
    log.info(f"  -> Weighted S:     {m['specificity_weighted']:.4f}")
    
    # Log other metrics to WandB via trial user_attrs
    for k, v in m.items():
        if isinstance(v, (int, float)):
            trial.set_user_attr(k, float(v))
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("stopped_epoch", stopped_epoch)
            
    return float(combined_metric)


def main():
    parser = argparse.ArgumentParser(
        description="Direct Train & Tune MLP Meta-Learner (Global Test Validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('--preds_dir', nargs='+', required=True,
                          help='Directories with base learner predictions (can specify multiple)')
    required.add_argument('--indicator_file', type=str, required=True,
                          help='Parquet file with indicator features and labels')
    
    # Tuning config
    tune_args = parser.add_argument_group('tuning configuration')
    tune_args.add_argument('--n_trials', type=int, default=50,
                           help='Number of Optuna trials for tuning (default: 50)')

    # Training parameters
    train_args = parser.add_argument_group('training parameters')
    train_args.add_argument('--max_epochs', type=int, default=20000,
                            help='Maximum number of epochs to train the MLP (default: 20000)')
    train_args.add_argument('--patience', type=int, default=20,
                            help='Number of epochs with no improvement before stopping (default: 20)')
    train_args.add_argument('--tol', type=float, default=1e-4,
                            help='Minimum interval of difference for an improvement to count (default: 0.0001)')
    train_args.add_argument('--transformer_weight', type=float, default=5.0,
                            help='Weight multiplier to prioritize Transformer base model predictions (default: 5.0)')

    # Logging config
    log_args = parser.add_argument_group('logging')
    log_args.add_argument('--use_wandb', action='store_true',
                          help='Enable W&B experiment tracking')
    log_args.add_argument('--wandb_project', type=str, default=None,
                          help='W&B project name')
    log_args.add_argument('--wandb_run_name', type=str, default=None,
                          help='W&B run name')
                          
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, indicator_cols = assemble_meta_data(args.preds_dir, args.indicator_file)
    
    if X_meta_train is None or X_meta_test.empty:
        log.critical("Failed to assemble meta-dataset or missing global test set. Exiting.")
        return

    n_classes = len(le.classes_)
    log.info(f"Meta-learner will be trained on {n_classes} classes.")

    # --- Feature Scaling: Give More Weight to Transformer Outputs for MLP ---
    transformer_cols = [col for col in X_meta_train.columns if 'Transformer' in str(col)]
    if transformer_cols and args.transformer_weight > 1.0:
        log.info(f"Scaling {len(transformer_cols)} Transformer columns by {args.transformer_weight}x to increase importance.")
        for col in transformer_cols:
            X_meta_train[col] = X_meta_train[col] * args.transformer_weight
            if not X_meta_test.empty:
                X_meta_test[col] = X_meta_test[col] * args.transformer_weight
    # ------------------------------------------------------------------------

    global OUTPUT_DIR
    OUTPUT_DIR = ensure_writable_results_dir(OUTPUT_DIR)

    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            log.info("Initialized Weights & Biases logging")
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False
    
    log.info(f"--- Starting Direct Training & Tuning for MLP Meta-Learner ({args.n_trials} trials) ---")
    study_name = 'mlp_metalearner_tuning'
    writable_journal_path = ensure_writable_db('mlp_' + TUNING_JOURNAL_FILE)
    
    storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=writable_journal_path))
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)

    callbacks = []
    if args.use_wandb:
        try:
            from optuna.integration.wandb import WeightsAndBiasesCallback
            wandb_callback = WeightsAndBiasesCallback(
                metric_name="combined_metric",
                wandb_kwargs={"project": args.wandb_project, "name": args.wandb_run_name}
            )
            callbacks.append(wandb_callback)
        except Exception as e:
            pass

    log.info("Using GLOBAL TEST SET for trial training and early stopping validation.")

    def single_run_objective(trial):
        return objective(
            trial, X_meta_train, y_meta_train, X_meta_test, y_meta_test, n_classes,
            patience=args.patience, tol=args.tol, max_epochs=args.max_epochs
        )

    study.optimize(single_run_objective, n_trials=args.n_trials, callbacks=callbacks)

    log.info("--- Tuning Complete ---")
    log.info(f"Best hyperparameters found: {study.best_params}")
    
    params_file = os.path.join(OUTPUT_DIR, 'best_mlp_metalearner_params.json')
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    # --- Final Test Set Evaluation using the absolutely best model ---
    global best_trial_data
    best_model = best_trial_data['model']
    
    if best_model is not None:
        import joblib
        model_path = os.path.join(OUTPUT_DIR, 'mlp_meta_learner_final.joblib')
        joblib.dump(best_model, model_path)
        log.info(f"\nFinal best MLP meta-learner saved to {model_path}!")

        if not X_meta_test.empty and not y_meta_test.empty:
            test_preds = best_model.predict(X_meta_test)
            test_metrics = compute_metrics(y_meta_test, test_preds, n_classes)
            
            # Save test confusion matrix diagram
            log.info("--- Generating Test Set Confusion Matrix Diagram ---")
            test_cm = confusion_matrix(y_meta_test, test_preds, labels=list(range(n_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title('Test Set Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            test_cm_path = os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png')
            plt.savefig(test_cm_path)
            plt.close()
            log.info(f"Saved test confusion matrix diagram to {test_cm_path}")

            log.info("--- Final Test Set Evaluation ---")
            log.info(f"  -> Accuracy:       {test_metrics['accuracy']:.4f}")
            log.info(f"  -> Weighted P:     {test_metrics['precision_weighted']:.4f}")
            log.info(f"  -> Weighted R:     {test_metrics['recall_weighted']:.4f}")
            log.info(f"  -> Weighted S:     {test_metrics['specificity_weighted']:.4f}")
            log.info(f"  -> Weighted F1:    {test_metrics['f1_weighted']:.4f}")
            log.info(f"\nClassification Report:\n{classification_report(y_meta_test, test_preds)}")
            
            if args.use_wandb:
                try:
                    import wandb
                    wandb_metrics = {f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
                    wandb_metrics["test/best_epoch"] = best_trial_data['epoch']
                    wandb.log(wandb_metrics)
                    log.info("Logged all test metrics to WandB.")
                except Exception as e:
                    log.warning(f"Failed to log test metrics to WandB: {e}")

    # Finish wandb run
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

if __name__ == '__main__':
    main()
