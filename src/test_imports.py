#!/usr/bin/env python3
"""
Test script to verify imports and module structure.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing module imports...")
    
    tests = []
    
    # Test dataset imports
    try:
        from dataset import (
            MLDatasetLoader,
            load_and_extract_data,
            prepare_train_test_split
        )
        tests.append(("dataset", True, None))
        print("✓ dataset module imports OK")
    except Exception as e:
        tests.append(("dataset", False, str(e)))
        print(f"✗ dataset module failed: {e}")
    
    # Test learning imports
    try:
        from learning import (
            ChoquetClassifier,
            ChoquetTnormClassifier,
            train_ensemble_main,
            train_aggregate_main
        )
        tests.append(("learning", True, None))
        print("✓ learning module imports OK")
    except Exception as e:
        tests.append(("learning", False, str(e)))
        print(f"✗ learning module failed: {e}")
    
    # Test fuzzy_measure imports
    try:
        from fuzzy_measure import (
            fuzzy_power,
            fuzzy_weight,
            fuzzy_power_tnorm,
            fuzzy_weight_tnorm
        )
        tests.append(("fuzzy_measure", True, None))
        print("✓ fuzzy_measure module imports OK")
    except Exception as e:
        tests.append(("fuzzy_measure", False, str(e)))
        print(f"✗ fuzzy_measure module failed: {e}")
    
    # Test evaluation imports
    try:
        from evaluation import (
            aggregate_multiple_results,
            compute_statistics,
            create_boxplots,
            create_comparison_plots
        )
        tests.append(("evaluation", True, None))
        print("✓ evaluation module imports OK")
    except Exception as e:
        tests.append(("evaluation", False, str(e)))
        print(f"✗ evaluation module failed: {e}")
    
    # Test optimize imports
    try:
        from learning.optimize import GD_minimize, objective, objective_tnorm
        tests.append(("learning.optimize", True, None))
        print("✓ learning.optimize module imports OK")
    except Exception as e:
        tests.append(("learning.optimize", False, str(e)))
        print(f"✗ learning.optimize module failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)
    print(f"Import Tests: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All imports successful!")
        return True
    else:
        print("\n✗ Some imports failed:")
        for module, success, error in tests:
            if not success:
                print(f"  - {module}: {error}")
        return False


def test_config():
    """Test that configuration file exists and can be loaded."""
    print("\n" + "="*60)
    print("Testing configuration...")
    
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['dataset', 'noise', 'training', 'models', 'choquet']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            print(f"✗ Missing config sections: {missing}")
            return False
        
        print(f"✓ Config file OK: {config_path}")
        print(f"  - Sections: {', '.join(config.keys())}")
        print(f"  - Models: {len(config['models'])}")
        print(f"  - Seeds: {config['training']['n_seeds']}")
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def test_structure():
    """Test that all expected files and directories exist."""
    print("\n" + "="*60)
    print("Testing project structure...")
    
    src_dir = Path(__file__).parent
    
    expected = {
        'files': [
            'main.py',
            'config/config.yaml',
            'dataset/__init__.py',
            'dataset/dataloader.py',
            'dataset/load_dataset.py',
            'learning/__init__.py',
            'learning/choquet_learnable.py',
            'learning/train_ensemble.py',
            'learning/train_aggregate.py',
            'learning/optimize/__init__.py',
            'learning/optimize/gradient_descent.py',
            'learning/optimize/objective_functions.py',
            'fuzzy_measure/__init__.py',
            'fuzzy_measure/classical.py',
            'fuzzy_measure/tnorm.py',
            'evaluation/__init__.py',
            'evaluation/aggregate_results.py',
            'evaluation/plot_results.py',
        ],
        'dirs': [
            'config',
            'dataset',
            'learning',
            'learning/optimize',
            'fuzzy_measure',
            'evaluation',
            'figures',
        ]
    }
    
    all_ok = True
    
    # Check directories
    for dir_path in expected['dirs']:
        full_path = src_dir / dir_path
        if full_path.exists():
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            all_ok = False
    
    # Check files
    for file_path in expected['files']:
        full_path = src_dir / file_path
        if full_path.exists():
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CHOQUET AGGREGATION - MODULE TEST")
    print("="*60)
    
    results = {
        'imports': test_imports(),
        'config': test_config(),
        'structure': test_structure()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.capitalize():20s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nYou can now run:")
        print("  python main.py --config config/config.yaml --n_jobs 2")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease fix the issues above before running the pipeline.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
