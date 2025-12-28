#!/usr/bin/env python3
"""
inspect_params.py
A small helper to load and pretty-print Optuna best-params JSON files produced by tune_models.py

Usage examples:
    python inspect_params.py tuning_results/best_params_multiclass_qml_tuning_CNV_app1_pca_standard.json
    python inspect_params.py tuning_results/  # prints all best_params_*.json files in the dir
"""
import json
import os
import argparse
from glob import glob


def load_and_print(path):
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"\n--- {os.path.basename(path)} ---")
    for k, v in data.items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and display best hyperparameters from Optuna tuning results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Inspect single file
  python inspect_params.py tuning_results/best_params_multiclass_qml_tuning_CNV_app1_pca_standard.json
  
  # Inspect all files in directory
  python inspect_params.py tuning_results/
  
  # Inspect multiple files/directories
  python inspect_params.py tuning_results/ model1.json model2.json
  
Description:
  Pretty-prints best hyperparameters from tune_models.py output.
  Searches for best_params_*.json files when given a directory.
        """)
    parser.add_argument('paths', nargs='+', metavar='PATH',
                       help='JSON file(s) or directory(ies) containing best_params_*.json files')
    args = parser.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(sorted(glob(os.path.join(p, 'best_params_*.json'))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: {p} not found; skipping")

    if not files:
        print('No best_params_*.json files found.')
        return

    for f in files:
        try:
            load_and_print(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}")


if __name__ == '__main__':
    main()
