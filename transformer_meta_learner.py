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
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import centralized logger and assembly utilities
from logging_utils import log
from metalearner import (
    ENCODER_DIR,
    RANDOM_STATE,
    TUNING_JOURNAL_FILE,
    assemble_meta_data,
    ensure_writable_db,
    ensure_writable_results_dir,
    set_seed,
)
from utils.metrics_utils import compute_metrics

# Import the fusion model
from performance_extensions.transformer_fusion import MultimodalFusionClassifier

# Shared local constants
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')

# Global dictionary to track the best model during tuning
best_trial_data = {
    'score': -np.inf,
    'model_state': None,
    'epoch': 0,
    'params': None
}

def get_target_metric(y_true, y_pred, n_classes):
    """
    Exclusively focuses on f1_weighted to ensure we beat the base learner's baseline.
    Returns the f1_weighted as the primary score, plus the full metrics dictionary.
    """
    m = compute_metrics(y_true, y_pred, n_classes)
    return m['f1_weighted'], m

def df_to_modality_dict(X_df, device):
    """Converts the flat meta-dataset DataFrame into a dictionary of PyTorch tensors and pushes to device."""
    modalities = ['miRNA', 'CNV', 'Meth', 'Prot', 'GeneExpr', 'Transformer']
    modality_data = {}
    modality_dims = {}
    
    for mod in modalities:
        cols = [c for c in X_df.columns if f'pred_{mod}' in c]
        if cols:
            modality_data[mod] = torch.tensor(X_df[cols].values, dtype=torch.float32, device=device)
            modality_dims[mod] = len(cols)
            
    return modality_data, modality_dims

def evaluate_model(model, data_dict, labels_tensor, n_classes):
    """Evaluates the PyTorch model using full batching and returns metrics."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(data_dict)
        _, predicted = torch.max(logits, 1)
        
    preds = predicted.cpu().numpy()
    labels = labels_tensor.cpu().numpy()
    
    target_score, m_dict = get_target_metric(labels, preds, n_classes)
    return target_score, m_dict, preds

def train_transformer_meta_learner(
    train_data, y_train_tensor, val_data, y_val_tensor, modality_dims, 
    n_classes, params, device, patience=20, tol=1e-4, max_epochs=2000
):
    """Training loop with early stopping using FULL BATCHING."""
    
    # Extract learning rate
    lr = params.pop('lr', 1e-3)
    
    model = MultimodalFusionClassifier(
        modality_dims=modality_dims,
        num_classes=n_classes,
        **params
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_score = -np.inf
    best_weights = None
    best_metrics = None
    best_epoch = 0
    stopped_epoch = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # --- Full Batch Training ---
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(train_data)
        loss = criterion(logits, y_train_tensor)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
            
        # --- Full Batch Validation ---
        target_score, m_dict, _ = evaluate_model(model, val_data, y_val_tensor, n_classes)
        
        # Checkpointing: Save weights if F1 improves
        if target_score >= best_score + tol:
            best_score = target_score
            best_metrics = m_dict
            best_epoch = epoch
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience > 0 and patience_counter >= patience:
            stopped_epoch = epoch
            log.info(f"Early stopping at epoch {stopped_epoch}. Restoring best weights from epoch {best_epoch} (F1: {best_score:.4f}).")
            break
            
    if stopped_epoch == 0:
        stopped_epoch = max_epochs - 1
        
    # Restore the checkpointed weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, best_score, best_metrics, best_epoch, stopped_epoch


def objective(trial, train_data, y_train_tensor, val_data, y_val_tensor, modality_dims, n_classes, device, patience, tol, max_epochs):
    """Optuna objective for tuning the Transformer Meta-Learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Fix for Optuna dynamic space ValueError: Define the full static space once.
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64, 128, 256, 512])
    
    # Immediately prune invalid/unwanted combinations to guide Optuna
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned()
    if num_heads == 8 and embed_dim < 64:
        raise optuna.TrialPruned()
    if num_heads == 2 and embed_dim > 256:
        raise optuna.TrialPruned()
        
    params = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'use_cls_token': trial.suggest_categorical('use_cls_token', [True, False]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
    } # Batch size removed for Full Batching

    model, target_metric, m, best_epoch, stopped_epoch = train_transformer_meta_learner(
        train_data, y_train_tensor, val_data, y_val_tensor, modality_dims, n_classes, 
        copy.deepcopy(params), device, patience=patience, tol=tol, max_epochs=max_epochs
    )
    
    # Save the absolute best model globally
    global best_trial_data
    if target_metric > best_trial_data['score']:
        best_trial_data['score'] = target_metric
        best_trial_data['model_state'] = copy.deepcopy(model.state_dict())
        best_trial_data['epoch'] = best_epoch
        best_trial_data['params'] = copy.deepcopy(params)
    
    log.info(f"Trial {trial.number} Results:")
    log.info(f"  -> F1 Weighted (Target): {target_metric:.4f}")
    log.info(f"  -> Stopped Epoch:        {stopped_epoch} | Best Epoch: {best_epoch}")
    log.info(f"  -> Accuracy:             {m['accuracy']:.4f}")
    log.info(f"  -> Weighted P:           {m['precision_weighted']:.4f}")
    log.info(f"  -> Weighted R:           {m['recall_weighted']:.4f}")
    log.info(f"  -> Weighted S:           {m['specificity_weighted']:.4f}")
    
    for k, v in m.items():
        if isinstance(v, (int, float)):
            trial.set_user_attr(k, float(v))
    trial.set_user_attr("best_epoch", best_epoch)
            
    return float(target_metric)


def main():
    parser = argparse.ArgumentParser(
        description="Direct Train & Tune Transformer Meta-Learner (Full Batching)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--preds_dir', nargs='+', required=True, help='Directories with base learner predictions')
    required.add_argument('--indicator_file', type=str, required=True, help='Parquet file with indicator features and labels')
    
    tune_args = parser.add_argument_group('tuning configuration')
    tune_args.add_argument('--n_trials', type=int, default=50)

    train_args = parser.add_argument_group('training parameters')
    train_args.add_argument('--max_epochs', type=int, default=1000)
    train_args.add_argument('--patience', type=int, default=30)
    train_args.add_argument('--tol', type=float, default=1e-4)

    log_args = parser.add_argument_group('logging config')
    log_args.add_argument('--use_wandb', action='store_true')
    log_args.add_argument('--wandb_project', type=str, default=None)
    log_args.add_argument('--wandb_run_name', type=str, default=None)
                          
    args = parser.parse_args()
    set_seed(RANDOM_STATE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, indicator_cols = assemble_meta_data(args.preds_dir, args.indicator_file)
    if X_meta_train is None:
        log.critical("Failed to assemble meta-dataset. Exiting.")
        return

    n_classes = len(le.classes_)
    log.info(f"Meta-learner will be trained on {n_classes} classes.")

    global OUTPUT_DIR
    OUTPUT_DIR = ensure_writable_results_dir(OUTPUT_DIR)

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False
    
    log.info(f"--- Starting Direct Tuning for Transformer Meta-Learner ({args.n_trials} trials) ---")
    study_name = 'transformer_metalearner_tuning'
    writable_journal_path = ensure_writable_db('transformer_' + TUNING_JOURNAL_FILE)
    
    storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=writable_journal_path))
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)

    callbacks = []
    if args.use_wandb:
        try:
            from optuna.integration.wandb import WeightsAndBiasesCallback
            callbacks.append(WeightsAndBiasesCallback(metric_name="f1_weighted", wandb_kwargs={"project": args.wandb_project}))
        except Exception:
            pass

    # Create a single Train/Validation split (90/10) for trial training and early stopping
    log.info("Creating internal validation split (10%) for trial training and early stopping.")
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_meta_train, y_meta_train, test_size=0.10, stratify=y_meta_train, random_state=RANDOM_STATE
    )

    # Pre-process all data into Full Batch Tensors on the target device
    train_data, modality_dims = df_to_modality_dict(X_train_split, device)
    val_data, _ = df_to_modality_dict(X_val_split, device)
    
    y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.long, device=device)
    y_val_tensor = torch.tensor(y_val_split.values, dtype=torch.long, device=device)

    def single_run_objective(trial):
        return objective(
            trial, train_data, y_train_tensor, val_data, y_val_tensor, modality_dims,
            n_classes, device, args.patience, args.tol, args.max_epochs
        )

    study.optimize(single_run_objective, n_trials=args.n_trials, callbacks=callbacks)
    
    params_file = os.path.join(OUTPUT_DIR, 'best_transformer_metalearner_params.json')
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    # --- Final Test Set Evaluation using the absolutely best model ---
    global best_trial_data
    if best_trial_data['model_state'] is not None:
        log.info("--- Evaluating Final Best Transformer Meta-Learner ---")
        
        # Re-instantiate the best model
        best_params = best_trial_data['params']
        best_model = MultimodalFusionClassifier(
            modality_dims=modality_dims,
            num_classes=n_classes,
            **best_params
        ).to(device)
        best_model.load_state_dict(best_trial_data['model_state'])
        
        model_path = os.path.join(OUTPUT_DIR, 'transformer_meta_learner_final.pth')
        torch.save(best_model.state_dict(), model_path)
        log.info(f"Final model weights saved to {model_path}!")

        # Final Test Set Evaluation
        if not X_meta_test.empty and not y_meta_test.empty:
            test_data, _ = df_to_modality_dict(X_meta_test, device)
            y_test_tensor = torch.tensor(y_meta_test.values, dtype=torch.long, device=device)
            
            _, test_metrics, test_preds = evaluate_model(best_model, test_data, y_test_tensor, n_classes)
            
            log.info("--- Generating Test Set Confusion Matrix Diagram ---")
            test_cm = confusion_matrix(y_meta_test, test_preds, labels=list(range(n_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title('Test Set Confusion Matrix (Transformer Meta-Learner)')
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
                wandb_metrics = {f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
                wandb_metrics["test/best_epoch"] = best_trial_data['epoch']
                wandb.log(wandb_metrics)

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()import argparse
import copy
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

# Import centralized logger and assembly utilities
from logging_utils import log
from metalearner import (
    ENCODER_DIR,
    RANDOM_STATE,
    TUNING_JOURNAL_FILE,
    assemble_meta_data,
    ensure_writable_db,
    ensure_writable_results_dir,
    set_seed,
)
from utils.metrics_utils import compute_metrics

# Import the fusion model
from performance_extensions.transformer_fusion import MultimodalFusionClassifier

# Shared local constants
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'final_model_and_predictions')

def get_target_metric(y_true, y_pred, n_classes):
    """
    Exclusively focuses on f1_weighted to ensure we beat the base learner's 0.817.
    Returns the f1_weighted as the primary score, plus the full metrics dictionary.
    """
    m = compute_metrics(y_true, y_pred, n_classes)
    return m['f1_weighted'], m

def df_to_modality_dict(X_df):
    """Converts the flat meta-dataset DataFrame into a dictionary of PyTorch tensors."""
    modalities = ['miRNA', 'CNV', 'Meth', 'Prot', 'GeneExpr', 'Transformer']
    modality_data = {}
    modality_dims = {}
    
    for mod in modalities:
        # Find columns belonging to this base learner
        cols = [c for c in X_df.columns if f'pred_{mod}' in c]
        if cols:
            modality_data[mod] = torch.tensor(X_df[cols].values, dtype=torch.float32)
            modality_dims[mod] = len(cols)
            
    return modality_data, modality_dims

class MetaLearnerDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for the Transformer Meta-Learner."""
    def __init__(self, data_dict, labels):
        self.data = data_dict
        self.labels = torch.tensor(labels.values if isinstance(labels, pd.Series) else labels, dtype=torch.long)
        self.n_samples = len(self.labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sample_data = {k: v[idx] for k, v in self.data.items()}
        return sample_data, self.labels[idx]

def create_dataloader(data_dict, labels, batch_size, shuffle=True):
    dataset = MetaLearnerDataset(data_dict, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def evaluate_model(model, dataloader, n_classes, device):
    """Evaluates the PyTorch model and returns the f1_weighted metric and stats."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            for mod in batch_data:
                batch_data[mod] = batch_data[mod].to(device)
            
            logits, _ = model(batch_data)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    target_score, m_dict = get_target_metric(all_labels, all_preds, n_classes)
    return target_score, m_dict, all_preds

def train_transformer_meta_learner(
    X_train_df, y_train, X_val_df, y_val, n_classes, params, 
    device, patience=20, tol=1e-4, max_epochs=2000
):
    """Training loop with early stopping, strictly monitoring f1_weighted."""
    
    # Convert DataFrames to Tensor Dictionaries
    train_data, modality_dims = df_to_modality_dict(X_train_df)
    val_data, _ = df_to_modality_dict(X_val_df)
    
    train_loader = create_dataloader(train_data, y_train, batch_size=params.pop('batch_size', 32), shuffle=True)
    val_loader = create_dataloader(val_data, y_val, batch_size=128, shuffle=False)
    
    # Extract learning rate
    lr = params.pop('lr', 1e-3)
    
    model = MultimodalFusionClassifier(
        modality_dims=modality_dims,
        num_classes=n_classes,
        **params
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # In-memory Checkpointing Variables
    best_score = -np.inf
    best_weights = None
    best_metrics = None
    best_epoch = 0
    stopped_epoch = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            for mod in batch_data:
                batch_data[mod] = batch_data[mod].to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Validation (Monitoring strictly f1_weighted)
        target_score, m_dict, _ = evaluate_model(model, val_loader, n_classes, device)
        
        # Checkpointing: Save weights if F1 improves
        if target_score >= best_score + tol:
            best_score = target_score
            best_metrics = m_dict
            best_epoch = epoch
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience > 0 and patience_counter >= patience:
            stopped_epoch = epoch
            log.info(f"Early stopping at epoch {stopped_epoch}. Restoring best weights from epoch {best_epoch} (F1: {best_score:.4f}).")
            break
            
    if stopped_epoch == 0:
        stopped_epoch = max_epochs - 1
        
    # Restore the checkpointed weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model, best_score, best_metrics, best_epoch, stopped_epoch


def objective(trial, X_train, y_train, X_val, y_val, n_classes, device, patience, tol, max_epochs):
    """Optuna objective for tuning the Transformer Meta-Learner."""
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # Fix for Optuna dynamic space ValueError: Define the full static space once.
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64, 128, 256, 512])
    
    # Immediately prune invalid/unwanted combinations to guide Optuna
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned()
    if num_heads == 8 and embed_dim < 64:
        raise optuna.TrialPruned()
    if num_heads == 2 and embed_dim > 256:
        raise optuna.TrialPruned()
        
    params = {
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'use_cls_token': trial.suggest_categorical('use_cls_token', [True, False]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    model, target_metric, m, best_epoch, stopped_epoch = train_transformer_meta_learner(
        X_train, y_train, X_val, y_val, n_classes, params, 
        device, patience=patience, tol=tol, max_epochs=max_epochs
    )
    
    log.info(f"Trial {trial.number} Results:")
    log.info(f"  -> F1 Weighted (Target): {target_metric:.4f}")
    log.info(f"  -> Stopped Epoch:        {stopped_epoch} | Best Epoch: {best_epoch}")
    log.info(f"  -> Accuracy:             {m['accuracy']:.4f}")
    log.info(f"  -> Weighted P:           {m['precision_weighted']:.4f}")
    log.info(f"  -> Weighted R:           {m['recall_weighted']:.4f}")
    log.info(f"  -> Weighted S:           {m['specificity_weighted']:.4f}")
    
    for k, v in m.items():
        if isinstance(v, (int, float)):
            trial.set_user_attr(k, float(v))
    trial.set_user_attr("best_epoch", best_epoch)
            
    return float(target_metric)


def main():
    parser = argparse.ArgumentParser(description="Train/Tune Transformer Meta-Learner")
    
    required = parser.add_argument_group('required arguments')
    required.add_argument('--preds_dir', nargs='+', required=True, help='Directories with base learner predictions')
    required.add_argument('--indicator_file', type=str, required=True, help='Parquet file with indicator features and labels')
    
    mode_args = parser.add_argument_group('operation mode')
    mode_args.add_argument('--mode', type=str, default='tune_and_train', choices=['train', 'tune', 'tune_and_train'])
    mode_args.add_argument('--n_trials', type=int, default=50)

    train_args = parser.add_argument_group('training parameters')
    train_args.add_argument('--max_epochs', type=int, default=1000)
    train_args.add_argument('--patience', type=int, default=30)
    train_args.add_argument('--tol', type=float, default=1e-4)

    log_args = parser.add_argument_group('logging config')
    log_args.add_argument('--use_wandb', action='store_true')
    log_args.add_argument('--wandb_project', type=str, default=None)
    log_args.add_argument('--wandb_run_name', type=str, default=None)
                          
    args = parser.parse_args()
    set_seed(RANDOM_STATE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    X_meta_train, y_meta_train, X_meta_test, y_meta_test, le, indicator_cols = assemble_meta_data(args.preds_dir, args.indicator_file)
    if X_meta_train is None:
        log.critical("Failed to assemble meta-dataset. Exiting.")
        return

    n_classes = len(le.classes_)
    log.info(f"Meta-learner will be trained on {n_classes} classes.")

    global OUTPUT_DIR
    OUTPUT_DIR = ensure_writable_results_dir(OUTPUT_DIR)

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
            args.use_wandb = False
    
    # ---------------------------------------------------------
    # PHASE 1: TUNING
    # ---------------------------------------------------------
    if args.mode in ['tune', 'tune_and_train']:
        log.info(f"--- Tuning Transformer Meta-Learner ({args.n_trials} trials) ---")
        study_name = 'transformer_metalearner_tuning'
        writable_journal_path = ensure_writable_db('transformer_' + TUNING_JOURNAL_FILE)
        
        storage = JournalStorage(JournalFileBackend(lock_obj=None, file_path=writable_journal_path))
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)

        callbacks = []
        if args.use_wandb:
            try:
                from optuna.integration.wandb import WeightsAndBiasesCallback
                callbacks.append(WeightsAndBiasesCallback(metric_name="f1_weighted", wandb_kwargs={"project": args.wandb_project}))
            except Exception:
                pass

        def cv_objective(trial):
            skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            for train_idx, val_idx in skf.split(X_meta_train, y_meta_train):
                X_train_fold, y_train_fold = X_meta_train.iloc[train_idx], y_meta_train.iloc[train_idx]
                X_val_fold, y_val_fold = X_meta_train.iloc[val_idx], y_meta_train.iloc[val_idx]
                
                fold_score = objective(
                    trial, X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                    n_classes, device, args.patience, args.tol, args.max_epochs
                )
                scores.append(fold_score)
            return np.mean(scores)

        study.optimize(cv_objective, n_trials=args.n_trials, callbacks=callbacks)
        
        params_file = os.path.join(OUTPUT_DIR, 'best_transformer_metalearner_params.json')
        with open(params_file, 'w') as f:
            json.dump(study.best_params, f, indent=4)
            
    # ---------------------------------------------------------
    # PHASE 2: FINAL TRAINING
    # ---------------------------------------------------------
    if args.mode in ['train', 'tune_and_train']:
        log.info("--- Training Final Transformer Meta-Learner ---")
        try:
            with open(os.path.join(OUTPUT_DIR, 'best_transformer_metalearner_params.json'), 'r') as f:
                model_params = json.load(f)
            log.info(f"Loaded best parameters: {model_params}")
        except FileNotFoundError:
            log.warning("Tuned parameters not found. Using safe defaults.")
            model_params = {
                'embed_dim': 128, 'num_heads': 4, 'num_layers': 2,
                'dim_feedforward': 256, 'dropout': 0.3, 'use_cls_token': True,
                'lr': 1e-3, 'batch_size': 32
            }
            
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_meta_train, y_meta_train, test_size=0.10, stratify=y_meta_train, random_state=RANDOM_STATE
        )

        model, final_score, val_metrics, best_epoch, stopped_epoch = train_transformer_meta_learner(
            X_train_final, y_train_final, X_val_final, y_val_final, 
            n_classes, model_params, device, 
            patience=args.patience, tol=args.tol, max_epochs=args.max_epochs
        )
        
        model_path = os.path.join(OUTPUT_DIR, 'transformer_meta_learner_final.pth')
        torch.save(model.state_dict(), model_path)
        log.info(f"Final model weights saved to {model_path}!")

        log.info("--- Internal Validation Metrics (Best Epoch) ---")
        log.info(f"  -> F1 Weighted (Target): {final_score:.4f}")
        log.info(f"  -> Stopped Epoch:        {stopped_epoch} | Best Epoch: {best_epoch}")
        log.info(f"  -> Accuracy:             {val_metrics['accuracy']:.4f}")
        log.info(f"  -> Weighted P:           {val_metrics['precision_weighted']:.4f}")
        log.info(f"  -> Weighted R:           {val_metrics['recall_weighted']:.4f}")
        log.info(f"  -> Weighted S:           {val_metrics['specificity_weighted']:.4f}")

        # Final Test Set Evaluation
        if not X_meta_test.empty and not y_meta_test.empty:
            test_data, _ = df_to_modality_dict(X_meta_test)
            test_loader = create_dataloader(test_data, y_meta_test, batch_size=128, shuffle=False)
            
            _, test_metrics, test_preds = evaluate_model(model, test_loader, n_classes, device)
            
            # ---------------------------------------------------------
            # CONFUSION MATRIX GENERATION
            # ---------------------------------------------------------
            log.info("--- Generating Test Set Confusion Matrix Diagram ---")
            import matplotlib.pyplot as plt
            import seaborn as sns
            test_cm = confusion_matrix(y_meta_test, test_preds, labels=list(range(n_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title('Test Set Confusion Matrix (Transformer Meta-Learner)')
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
                wandb_metrics = {f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
                wandb_metrics["test/best_epoch"] = best_epoch
                wandb.log(wandb_metrics)

        if args.use_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()
