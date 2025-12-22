#!/usr/bin/env python3
"""
Main Training Pipeline

This script orchestrates the complete training pipeline:
1. Train ensemble classifiers on dataset
2. Train Choquet aggregation on ensemble outputs
"""

import argparse
import sys

from src.learning.train_ensemble import main as train_ensemble_main
from src.learning.train_aggregate import main as train_aggregate_main


def main():
    """
    Main function to run the complete training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Complete training pipeline: Ensemble + Choquet Aggregation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    parser.add_argument(
        '--skip_ensemble',
        action='store_true',
        help='Skip ensemble training (use existing results)'
    )
    parser.add_argument(
        '--skip_aggregate',
        action='store_true',
        help='Skip aggregation training'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Path to existing results directory (for aggregation only)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPLETE TRAINING PIPELINE")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Parallel jobs: {args.n_jobs}")
    print("="*80 + "\n")
    
    # Step 1: Train Ensemble Classifiers
    if not args.skip_ensemble:
        print("\n" + "="*80)
        print("STEP 1: TRAINING ENSEMBLE CLASSIFIERS")
        print("="*80 + "\n")
        
        try:
            train_ensemble_main(
                config_path=args.config,
                n_jobs=args.n_jobs
            )
            print("\n✓ Ensemble training completed successfully\n")
        except Exception as e:
            print(f"\n✗ Ensemble training failed: {e}\n")
            if not args.skip_aggregate:
                print("Stopping pipeline due to ensemble training failure.")
                sys.exit(1)
    else:
        print("\n⊗ Skipping ensemble training (using existing results)\n")
    
    # Step 2: Train Choquet Aggregation
    if not args.skip_aggregate:
        print("\n" + "="*80)
        print("STEP 2: TRAINING CHOQUET AGGREGATION")
        print("="*80 + "\n")
        
        try:
            train_aggregate_main(
                config_path=args.config,
                results_dir=args.results_dir,
                n_jobs=args.n_jobs
            )
            print("\n✓ Choquet aggregation completed successfully\n")
        except Exception as e:
            print(f"\n✗ Choquet aggregation failed: {e}\n")
            sys.exit(1)
    else:
        print("\n⊗ Skipping Choquet aggregation\n")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
