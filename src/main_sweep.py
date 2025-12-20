#!/usr/bin/env python3
"""
Pipeline Runner for Sweep Experiments

This script runs all the sweep experiments defined in the config directories:
- sweep_samples: Tests 4 class pairs with sweep on max_samples_per_class
- sweep_window: Tests 4 class pairs with sweep on window_size

Usage:
    python run_sweep_pipeline.py --mode samples --n_jobs 1
    python run_sweep_pipeline.py --mode window --n_jobs 1
    python run_sweep_pipeline.py --mode all --n_jobs 1
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime


def override_data_path_in_config(config_path: str, data_path: str) -> str:
    """
    Override data_path in config file temporarily.
    
    Parameters
    ----------
    config_path : str
        Path to original config file.
    data_path : str
        New data path to use.
    
    Returns
    -------
    str
        Path to temporary config file with overridden data_path.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data_path
    config['dataset']['data_path'] = data_path
    
    # Save to temporary file
    temp_config_path = Path(config_path).parent / f"temp_{Path(config_path).name}"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(temp_config_path)


def run_experiment(config_path: str, n_jobs: int = 1, run_aggregation: bool = True, data_path: str = None, verbose: bool = False):
    """
    Run a single experiment (training + optional aggregation).
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    n_jobs : int
        Number of parallel jobs.
    run_aggregation : bool
        Whether to run Choquet aggregation after training.
    data_path : str, optional
        Override data_path in config.
    verbose : bool
        Whether to print detailed output.
    """
    # Override data_path if provided
    if data_path:
        print(f"Overriding data_path with: {data_path}")
        config_path = override_data_path_in_config(config_path, data_path)
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {config_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Run ensemble training
    train_cmd = [
        sys.executable,
        "-m",
        "src.learning.train_ensemble",
        "--config", config_path,
        "--n_jobs", str(n_jobs)
    ]
    
    if not verbose:
        train_cmd.append("--quiet")
    
    print(f"Command: {' '.join(train_cmd)}\n")
    result = subprocess.run(train_cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n ERROR: Training failed for {config_path}")
        return False
    
    print(f"\n Training completed successfully for {config_path}")
    
    # Run Choquet aggregation
    if run_aggregation:
        print(f"\n{'-'*80}")
        print("Running Choquet aggregation...")
        print(f"{'-'*80}\n")
        
        agg_cmd = [
            sys.executable,
            "-m",
            "src.learning.train_aggregate",
            "--config", config_path,
            "--n_jobs", str(n_jobs)
        ]
        
        if not verbose:
            agg_cmd.append("--quiet")
        
        print(f"Command: {' '.join(agg_cmd)}\n")
        result = subprocess.run(agg_cmd, check=False)
        
        if result.returncode != 0:
            print(f"\n ERROR: Aggregation failed for {config_path}")
            return False
        
        print(f"\n Aggregation completed successfully for {config_path}")
        # Clean up temporary config if it was created
    if data_path and 'temp_' in config_path:
        Path(config_path).unlink(missing_ok=True)
        return True


def run_sweep_experiments(mode: str, n_jobs: int = 1, run_aggregation: bool = True, data_path: str = None, verbose: bool = False):
    """
    Run all sweep experiments for a given mode.
    
    Parameters
    ----------
    mode : str
        Either 'samples', 'window', or 'all'.
    n_jobs : int
        Number of parallel jobs.
    run_aggregation : bool
        Whether to run Choquet aggregation after each training.
    data_path : str, optional
        Override data_path in all configs.
    verbose : bool
        Whether to print detailed output.
    """
    base_dir = Path("src/config")
    
    # Define config directories based on mode
    if mode == "samples":
        config_dirs = [base_dir / "sweep_samples"]
    elif mode == "window":
        config_dirs = [base_dir / "sweep_window"]
    elif mode == "all":
        config_dirs = [base_dir / "sweep_samples", base_dir / "sweep_window"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'samples', 'window', or 'all'.")
    
    # Collect all config files
    config_files = []
    for config_dir in config_dirs:
        if not config_dir.exists():
            print(f"WARNING: Config directory not found: {config_dir}")
            continue
        
        configs = sorted(config_dir.glob("*.yaml"))
        if len(configs) == 0:
            print(f"WARNING: No config files found in {config_dir}")
            continue
        
        config_files.extend(configs)
    
    if len(config_files) == 0:
        print("ERROR: No config files found!")
        return
    
    print(f"\n{'='*80}")
    print(f"SWEEP PIPELINE - MODE: {mode.upper()}")
    print(f"Found {len(config_files)} configuration(s) to run")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Run aggregation: {run_aggregation}")
    print(f"{'='*80}\n")
    
    # Display all configs
    print("Configuration files:")
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {config_file.relative_to(base_dir)}")
    print()
    
    # Run experiments
    success_count = 0
    failed_configs = []
    
    start_time = datetime.now()
    
    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(config_files)}")
        print(f"{'='*80}")
        
        success = run_experiment(
            config_path=str(config_file),
            n_jobs=n_jobs,
            run_aggregation=run_aggregation,
            data_path=data_path,
            verbose=verbose
        )
        
        if success:
            success_count += 1
        else:
            failed_configs.append(config_file)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(config_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_configs)}")
    print(f"Duration: {duration}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_configs:
        print(f"\nFailed configurations:")
        for config_file in failed_configs:
            print(f"  - {config_file.relative_to(base_dir)}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run sweep experiments for Choquet aggregation project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sample sweep experiments only
  python run_sweep_pipeline.py --mode samples --n_jobs 1
  
  # Run window sweep experiments only
  python run_sweep_pipeline.py --mode window --n_jobs 1
  
  # Run all sweep experiments
  python run_sweep_pipeline.py --mode all --n_jobs 1
  
  # Run without Choquet aggregation (faster)
  python run_sweep_pipeline.py --mode samples --n_jobs 1 --skip-aggregation
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["samples", "window", "all"],
        help="Which sweep mode to run: 'samples' (max_samples_per_class), 'window' (window_size), or 'all'"
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for training (default: 1)"
    )
    
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip Choquet aggregation (only run ensemble training)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override data_path in all config files"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (prints all model scores)"
    )
    
    args = parser.parse_args()
    
    run_sweep_experiments(
        mode=args.mode,
        n_jobs=args.n_jobs,
        run_aggregation=not args.skip_aggregation,
        data_path=args.data_path,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
