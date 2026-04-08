#!/usr/bin/env python
"""
Hyperparameter tuning script for contrastive pretraining using Optuna.

This script tunes the hyperparameters for the self-supervised contrastive encoder:
- embed_dim: Encoder dimensionality
- projection_dim: Contrastive projection head dimension
- temperature: Contrastive temperature parameter
- lr: Learning rate
- weight_decay: L2 penalty
- transformer_d_model / num_heads / num_layers (if using transformer)

Usage:
    python examples/tune_pretrain_contrastive.py --data_dir /path/to/data --n_trials 50
    
    # Tune transformer encoder
    python examples/tune_pretrain_contrastive.py --encoder_type transformer --n_trials 50
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import optuna
import tempfile
import signal
import sys
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from performance_extensions.contrastive_learning import (
    ContrastiveMultiOmicsEncoder,
    ContrastiveLearningLoss
)
from performance_extensions.training_utils import (
    MultiOmicsDataset,
    collate_augmented_multiomics,
    pretrain_contrastive,
    save_pretrained_encoders
)

from logging_utils import log

# Global flag for interruption handling
interrupted = False

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info(f"Random seed set to {seed}")

def handle_interruption(signum, frame):
    """Handle interruption signals (SIGINT, SIGTERM) gracefully."""
    global interrupted
    if not interrupted:
        interrupted = True
        log.warning(f"\nReceived interruption signal ({signum}). Finishing current trial...")
        log.warning("Press Ctrl+C again to force exit.")
    else:
        log.error("\nForced interruption. Exiting immediately.")
        sys.exit(1)

def is_db_writable(db_path):
    if not os.path.exists(db_path):
        parent_dir = os.path.dirname(db_path) or '.'
        return os.access(parent_dir, os.W_OK)
    return os.access(db_path, os.W_OK)

def ensure_writable_db(db_path):
    if is_db_writable(db_path):
        return db_path
    
    log.warning(f"Database at {db_path} is read-only. Copying to a writable location...")
    candidate_paths = [
        os.path.join(os.getcwd(), 'optuna_contrastive_working.db'),
        os.path.join(tempfile.gettempdir(), 'optuna_contrastive_working.db')
    ]
    for candidate in candidate_paths:
        try:
            parent_dir = os.path.dirname(candidate) or '.'
            if os.access(parent_dir, os.W_OK):
                if os.path.exists(db_path):
                    import shutil
                    shutil.copy2(db_path, candidate)
                log.info(f"Using writable database at: {candidate}")
                return candidate
        except Exception:
            continue
    log.warning("No writable location found, using in-memory database")
    return ":memory:"


def load_multiomics_data(data_dir, skip_modalities=None, impute_strategy='none'):
    data_dir = Path(data_dir)
    all_modalities = ['GeneExpr', 'miRNA', 'Meth', 'CNV', 'Prot', 'SNV']
    skip_modalities = set(skip_modalities or [])
    
    data = {}
    modality_dims = {}
    nan_stats = {}
    case_ids = None
    labels = None
    METADATA_COLS = {'class', 'case_id'}
    
    for modality in all_modalities:
        if modality in skip_modalities:
            continue
            
        file_path = data_dir / f"data_{modality}_.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            
            if 'case_id' in df.columns:
                df = df.sort_values('case_id')
                if case_ids is None:
                    case_ids = df['case_id'].values
            
            if labels is None and 'class' in df.columns:
                labels = df['class'].values
            
            feature_cols = [col for col in df.columns if col.lower() not in METADATA_COLS and col not in METADATA_COLS]
            features = df[feature_cols].values.astype(np.float32)
            
            nan_mask = np.isnan(features)
            nan_count = np.sum(nan_mask)
            nan_samples = np.sum(np.any(nan_mask, axis=1))
            
            if nan_count > 0:
                if impute_strategy == 'none':
                    pass
                elif impute_strategy == 'median':
                    col_medians = np.nanmedian(features, axis=0)
                    nan_indices = np.where(nan_mask)
                    features[nan_indices] = col_medians[nan_indices[1]]
                elif impute_strategy == 'mean':
                    col_means = np.nanmean(features, axis=0)
                    nan_indices = np.where(nan_mask)
                    features[nan_indices] = col_means[nan_indices[1]]
                elif impute_strategy == 'zero':
                    features = np.nan_to_num(features, nan=0.0)
                elif impute_strategy == 'drop':
                    valid_rows = ~np.any(nan_mask, axis=1)
                    features = features[valid_rows]
            
            data[modality] = features
            modality_dims[modality] = features.shape[1]
    
    return data, modality_dims, nan_stats, case_ids, labels


def objective(trial, args, data, modality_dims, labels, case_ids, device):
    global interrupted
    if interrupted:
        raise optuna.TrialPruned()
    
    log.info(f"--- Starting Trial {trial.number} ---")
    
    # HYPERPARAMETERS
    temperature = trial.suggest_categorical('temperature', [0.02, 0.03, 0.04, 0.05])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    
    projection_dim_choices = [32, 64, 128, 256]
    projection_dim = trial.suggest_categorical('projection_dim', projection_dim_choices)
    
    if args.encoder_type == 'mlp':
        embed_dim_choices = [64, 128, 256, 512]
        embed_dim = trial.suggest_categorical('embed_dim', embed_dim_choices)
        transformer_d_model = None
        transformer_num_heads = None
        transformer_num_layers = None
    else:
        transformer_d_model_choices = [64, 128, 256, 512]
        transformer_d_model = trial.suggest_categorical('transformer_d_model', transformer_d_model_choices)
        embed_dim = transformer_d_model  # Embed dim natively matches d_model for transformers here
        transformer_num_heads = trial.suggest_categorical('transformer_num_heads', [4, 8])
        transformer_num_layers = trial.suggest_int('transformer_num_layers', 1, 4)
        
        # Ensure d_model is divisible by num_heads
        while transformer_d_model % transformer_num_heads != 0:
            transformer_num_heads = transformer_num_heads // 2
            if transformer_num_heads < 2:
                transformer_num_heads = 2
                break
    
    if args.full_batch:
        batch_size = -1
    else:
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    log.info(f"Trial {trial.number} params: temp={temperature:.4f}, lr={lr:.6f}, "
             f"wd={weight_decay:.6f}, proj_dim={projection_dim}, embed_dim={embed_dim}")
    if args.encoder_type == 'transformer':
        log.info(f"  Transformer: d_model={transformer_d_model}, heads={transformer_num_heads}, layers={transformer_num_layers}")

    # SPLIT DATA (80% / 20%) FOR VALIDATION (No K-Fold for Contrastive pretraining to save mass time overhead)
    n_samples = list(data.values())[0].shape[0]
    indices = np.arange(n_samples)
    
    if labels is not None and len(np.unique(labels)) > 1:
        train_idx, val_idx = train_test_split(indices, test_size=args.val_size, stratify=labels, random_state=args.seed)
    else:
        train_idx, val_idx = train_test_split(indices, test_size=args.val_size, random_state=args.seed)
    
    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    
    from torch.utils.data import DataLoader
    train_dataset = MultiOmicsDataset(train_data, apply_augmentation=True, num_augmented_views=2)
    val_dataset = MultiOmicsDataset(val_data, apply_augmentation=True, num_augmented_views=2)
    
    actual_batch_size = len(train_dataset) if batch_size == -1 else batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True, 
        collate_fn=collate_augmented_multiomics
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        collate_fn=collate_augmented_multiomics
    )

    # Initialize Model
    model = ContrastiveMultiOmicsEncoder(
        modality_dims=modality_dims,
        embed_dim=embed_dim,
        projection_dim=projection_dim,
        encoder_type=args.encoder_type,
        transformer_d_model=transformer_d_model,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine scheduler just for this trial length
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    loss_fn = ContrastiveLearningLoss(
        temperature=temperature,
        use_cross_modal=args.use_cross_modal
    ).to(device)
    
    # We will hook into pretrain_contrastive internally, but since it doesn't support an Optuna pruning hook cleanly inside the function natively,
    # we'll just run it. Wait, pretrain contrastive is fast but takes epochs. Let's just run it! Early stopping is natively supported.
    
    # Empty run dir
    trial_ckpt_dir = Path(tempfile.mkdtemp())
    
    try:
        metrics = pretrain_contrastive(
            model=model,
            dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=args.num_epochs,
            checkpoint_dir=trial_ckpt_dir,
            checkpoint_interval=1000, # Avoid disk IO spam during tuning
            keep_last_n_checkpoints=1,
            warmup_epochs=max(5, int(args.num_epochs * 0.05)),
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
            patience=args.patience if args.patience > 0 else None,
            verbose=False,
            log_interval=1000 # Suppress verbose pretraining logs
        )
        best_val_loss = metrics.get('best_val_loss', metrics.get('best_loss', float('inf')))
    except Exception as e:
        log.error(f"Trial failed: {e}")
        return float('inf')
    finally:
        import shutil
        import gc
        if trial_ckpt_dir.exists():
            shutil.rmtree(trial_ckpt_dir, ignore_errors=True)
            
        # Clean up PyTorch memory so subsequent trials don't OOM
        try:
            del model
            del optimizer
            del loss_fn
            del train_loader
            del val_loader
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Report best val_loss (contrastive loss on validation set)
    trial.set_user_attr('best_val_loss', float(best_val_loss))
    trial.set_user_attr('epochs_run', len(metrics.get('epoch_losses', [])))
    
    log.info(f"--- Trial {trial.number} Finished: best_val_loss = {best_val_loss:.4f} ---")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Tuning Contrastive Pretraining with Optuna")
    
    data_args = parser.add_argument_group('data configuration')
    data_args.add_argument('--data_dir', type=str, default='final_processed_datasets')
    data_args.add_argument('--skip_modalities', nargs='+', type=str, default=None)
    data_args.add_argument('--impute_strategy', type=str, default=None)
    data_args.add_argument('--val_size', type=float, default=0.2)
    
    model_args = parser.add_argument_group('model architecture defaults')
    model_args.add_argument('--encoder_type', type=str, default='mlp', choices=['mlp', 'transformer'])
    model_args.add_argument('--use_cross_modal', action='store_true', help='Use Cross-Modal Constrastive Loss')
    
    tuning_args = parser.add_argument_group('tuning config')
    tuning_args.add_argument('--n_trials', type=int, default=50)
    tuning_args.add_argument('--num_epochs', type=int, default=100)
    tuning_args.add_argument('--patience', type=int, default=20)
    tuning_args.add_argument('--full_batch', action='store_true')
    tuning_args.add_argument('--max_grad_norm', type=float, default=1.0)
    tuning_args.add_argument('--study_name', type=str, default=None)
    
    system_args = parser.add_argument_group('system config')
    system_args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    system_args.add_argument('--seed', type=int, default=42)
    system_args.add_argument('--use_wandb', action='store_true')
    system_args.add_argument('--wandb_project', type=str, default='contrastive-tuning')
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, handle_interruption)
    signal.signal(signal.SIGTERM, handle_interruption)
    set_seed(args.seed)
    
    device = torch.device(args.device)
    log.info(f"Using device: {device} | Encoder: {args.encoder_type}")
    
    if args.impute_strategy is None:
        args.impute_strategy = 'none' if args.encoder_type == 'transformer' else 'median'
    
    data, modality_dims, nan_stats, case_ids, labels = load_multiomics_data(
        args.data_dir, skip_modalities=args.skip_modalities, impute_strategy=args.impute_strategy
    )
    
    if not data:
        log.error("Failed to load data.")
        return
        
    study_name = args.study_name or f'contrastive_{args.encoder_type}_tune'
    db_path = os.environ.get('OPTUNA_DB_PATH', './optuna_contrastive_studies.db')
    writable_db = ensure_writable_db(db_path)
    storage = f"sqlite:///{writable_db}" if writable_db != ":memory:" else None
    
    wandb_callback = None
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=study_name, config=vars(args))
            wandb_callback = optuna.integration.WeightsAndBiasesCallback(
                metric_name='best_val_loss', wandb_kwargs={'project': args.wandb_project}
            )
        except Exception as e:
            log.warning(f"Failed to initialize WandB: {e}")
    
    study = optuna.create_study(
        direction='minimize', # We want to MINIMIZE contrastive loss!
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    callbacks = [wandb_callback] if wandb_callback else []
    def interruption_callback(study, trial):
        if interrupted:
            study.stop()
    callbacks.append(interruption_callback)
    
    try:
        study.optimize(
            lambda t: objective(t, args, data, modality_dims, labels, case_ids, device),
            n_trials=args.n_trials,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        log.warning("Optimization interrupted by user")
        
    log.info("\n==================================")
    log.info("Hyperparameter Tuning Complete")
    log.info("==================================")
    
    if len(study.trials) > 0:
        best_trial = study.best_trial
        log.info(f"\nBest trial: {best_trial.number} (Loss: {best_trial.value:.4f})")
        log.info("Best parameters:")
        for k, v in best_trial.params.items():
            log.info(f"  {k}: {v}")
            
        results_dir = Path('contrastive_tuning_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save output
        out_data = {
            'study_name': study_name,
            'encoder_type': args.encoder_type,
            'best_trial': best_trial.number,
            'best_loss': best_trial.value,
            'best_params': best_trial.params,
            'n_trials': len(study.trials)
        }
        with open(results_dir / f"{study_name}_best_config.json", 'w') as f:
            json.dump(out_data, f, indent=2)
        log.info(f"Saved config to {results_dir}/{study_name}_best_config.json")
    else:
        log.warning("No trials completed.")

if __name__ == "__main__":
    main()
