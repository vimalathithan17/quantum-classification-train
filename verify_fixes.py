#!/usr/bin/env python3
"""
Quick verification that all bug fixes are in place.

This script performs static analysis to verify that:
1. X_val_scaled is used (not undefined X_val)
2. fold_idx is not used (should be fold+1 or _final)
3. MaskedTransformer is not double-wrapped
4. inference.py passes tuple to meta-learner
5. Unused imports are removed
"""
import os
import re
import sys

def check_file(filepath, patterns, description):
    """Check if patterns exist or don't exist in a file."""
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_passed = True
    for pattern, should_exist, msg in patterns:
        found = re.search(pattern, content)
        if should_exist and not found:
            print(f"  ✗ FAIL: {msg}")
            all_passed = False
        elif not should_exist and found:
            print(f"  ✗ FAIL: {msg}")
            all_passed = False
        else:
            print(f"  ✓ PASS: {msg}")
    
    return all_passed


def main():
    print("=" * 70)
    print("Bug Fix Verification")
    print("=" * 70)
    
    results = []
    
    # 1. Check X_val_scaled fix in qml_models.py
    print("\n1. X_val_scaled Fix (qml_models.py)")
    print("-" * 50)
    results.append(check_file(
        'qml_models.py',
        [
            (r'val_all_zero\s*=\s*np\.all\(X_val_scaled\s*==\s*0', True, 
             "Uses X_val_scaled (not X_val) in validation"),
            (r'val_all_zero\s*=\s*np\.all\(X_val\s*==\s*0', False, 
             "Does not use undefined X_val"),
        ],
        "X_val_scaled fix"
    ))
    
    # 2. Check fold_idx fix in cfe_relupload.py
    print("\n2. fold_idx Fix (cfe_relupload.py)")
    print("-" * 50)
    results.append(check_file(
        'cfe_relupload.py',
        [
            (r'fold\{fold\+1\}', True, 
             "Uses fold+1 for run name in CV loop"),
            (r'_final', True, 
             "Uses _final suffix for final model"),
            (r'fold_idx', False, 
             "Does not use undefined fold_idx"),
        ],
        "fold_idx fix"
    ))
    
    # 3. Check MaskedTransformer single wrapping in tune_models.py
    print("\n3. MaskedTransformer Single Wrap (tune_models.py)")
    print("-" * 50)
    results.append(check_file(
        'tune_models.py',
        [
            (r"steps_list\.append\(\('scaler',\s*MaskedTransformer\(scaler", True, 
             "Wraps scaler with MaskedTransformer in steps_list"),
            (r"scaler\s*=\s*MaskedTransformer\(scaler\)", False, 
             "Does not double-wrap scaler before steps_list"),
        ],
        "MaskedTransformer wrapping"
    ))
    
    # 4. Check inference.py tuple input
    print("\n4. Inference Tuple Input (inference.py)")
    print("-" * 50)
    results.append(check_file(
        'inference.py',
        [
            (r'meta_learner\.predict\(\(X_base,\s*X_mask\)\)', True, 
             "Passes tuple (X_base, X_mask) to meta-learner"),
            (r'X_mask\s*=\s*np\.column_stack', True, 
             "Builds X_mask from indicator columns"),
        ],
        "Tuple input fix"
    ))
    
    # 5. Check unused imports removed
    print("\n5. Unused Imports Removed")
    print("-" * 50)
    results.append(check_file(
        'cfe_standard.py',
        [
            (r'from utils\.masked_transformers import MaskedTransformer', False, 
             "MaskedTransformer import removed from cfe_standard.py"),
        ],
        "Unused imports"
    ))
    results.append(check_file(
        'cfe_relupload.py',
        [
            (r'from utils\.masked_transformers import MaskedTransformer', False, 
             "MaskedTransformer import removed from cfe_relupload.py"),
        ],
        "Unused imports"
    ))
    
    # 6. Check duplicate import os removed
    print("\n6. Duplicate Imports Removed (inference.py)")
    print("-" * 50)
    with open('inference.py', 'r') as f:
        content = f.read()
    import_os_count = len(re.findall(r'^import os$', content, re.MULTILINE))
    if import_os_count == 1:
        print("  ✓ PASS: Single 'import os' statement")
        results.append(True)
    else:
        print(f"  ✗ FAIL: Found {import_os_count} 'import os' statements (expected 1)")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Verification Results: {passed}/{total} checks passed")
    
    if all(results):
        print("✓ All bug fixes verified!")
        return 0
    else:
        print("✗ Some checks failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
