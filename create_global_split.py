import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import glob

def create_global_split(data_dir='final_processed_datasets', out_train='data/global_train', out_test='data/global_test', test_size=0.2, random_state=42):
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)
    
    # Find all parquet files
    files = glob.glob(f"{data_dir}/data_*.parquet")
    if not files:
        print(f"No parquet files found in {data_dir}")
        return
        
    print(f"Found {len(files)} files. Computing global split...")
    
    # Read the first file to get all case_ids and stratify by class
    df_base = pd.read_parquet(files[0])
    
    if 'case_id' not in df_base.columns or 'class' not in df_base.columns:
        print("Error: 'case_id' or 'class' column missing from base dataset.")
        return
        
    # Get unique cases
    cases = df_base[['case_id', 'class']].drop_duplicates()
    
    # Split case_ids
    train_cases, test_cases = train_test_split(
        cases, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=cases['class']
    )
    
    train_ids = set(train_cases['case_id'])
    test_ids = set(test_cases['case_id'])
    
    print(f"Global Split: {len(train_ids)} Train cases, {len(test_ids)} Test cases.")
    
    # Split every modality based on these case_ids
    for file in files:
        modality_name = os.path.basename(file)
        df = pd.read_parquet(file)
        
        df_train = df[df['case_id'].isin(train_ids)].sort_values('case_id').reset_index(drop=True)
        df_test = df[df['case_id'].isin(test_ids)].sort_values('case_id').reset_index(drop=True)
        
        train_path = os.path.join(out_train, modality_name)
        test_path = os.path.join(out_test, modality_name)
        
        df_train.to_parquet(train_path)
        df_test.to_parquet(test_path)
        
        print(f"[{modality_name}] Train: {len(df_train)}, Test: {len(df_test)}")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Create global train/test splits for all parquet datasets.")
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SOURCE_DIR', 'final_processed_datasets'), help='Input directory with parquet files')
    parser.add_argument('--out_train', type=str, default=os.environ.get('GLOBAL_TRAIN_DIR', 'data/global_train'), help='Output training directory')
    parser.add_argument('--out_test', type=str, default=os.environ.get('GLOBAL_TEST_DIR', 'data/global_test'), help='Output testing directory')
    parser.add_argument('--test_size', type=float, default=float(os.environ.get('TEST_SPLIT_SIZE', 0.2)), help='Fraction of data for testing')
    parser.add_argument('--random_state', type=int, default=int(os.environ.get('RANDOM_STATE', 42)), help='Random seed for stratification')
    
    args = parser.parse_args()
    
    create_global_split(
        data_dir=args.data_dir,
        out_train=args.out_train,
        out_test=args.out_test,
        test_size=args.test_size,
        random_state=args.random_state
    )
