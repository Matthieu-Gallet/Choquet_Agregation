#!/usr/bin/env python3
"""
Choquet Aggregation Training and Evaluation

This script trains Choquet aggregation models on ensemble classifier outputs
and compares them with logistic regression baseline.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed


from .choquet_learnable import ChoquetClassifier, ChoquetTnormClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """
    Parse experiment parameters from folder name.
    
    Examples: 'seed_00', 'seed_03_noise_0.1', 'label_noise_5.0_seed_02'
    
    Parameters
    ----------
    folder_name : str
        Name of the experiment folder.
    
    Returns
    -------
    dict
        Dictionary of parsed parameters.
    """
    params = {}
    parts = folder_name.split('_')
    
    i = 0
    while i < len(parts):
        if parts[i] == 'seed' and i + 1 < len(parts):
            params['seed'] = int(parts[i + 1])
            i += 2
        elif parts[i] in ['noise', 'label', 'data'] and i + 1 < len(parts):
            # Handle cases like 'label_noise_5.0' or 'noise_0.1'
            if parts[i] == 'label' and i + 2 < len(parts) and parts[i + 1] == 'noise':
                params['label_noise'] = float(parts[i + 2])
                i += 3
            elif parts[i] == 'data' and i + 2 < len(parts) and parts[i + 1] == 'noise':
                params['data_noise'] = float(parts[i + 2])
                i += 3
            else:
                params['noise'] = float(parts[i + 1])
                i += 2
        else:
            i += 1
    
    return params


def load_experiment_data(result_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and test data from experiment results.
    
    Parameters
    ----------
    result_dir : Path
        Path to experiment results directory.
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
    """
    train_data = np.load(result_dir / 'train.npy', allow_pickle=True).item()
    test_data = np.load(result_dir / 'test.npy', allow_pickle=True).item()
    
    # Extract predictions from base classifiers
    y_train = train_data['y_train']
    y_test = test_data['y_test']
    
    # Stack predictions from all models (excluding ground truth)
    train_preds = []
    test_preds = []
    
    for key in sorted(train_data.keys()):
        if key != 'y_train':
            train_preds.append(train_data[key])
            test_preds.append(test_data[key])
    
    X_train = np.array(train_preds).T  # Shape: (n_samples, n_classifiers)
    X_test = np.array(test_preds).T
    
    return X_train, y_train, X_test, y_test


def create_choquet_models(config: Dict, seed: int) -> Dict:
    """
    Create Choquet aggregation models based on configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing Choquet parameters.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary of model instances.
    """
    choquet_config = config.get('choquet', {})
    
    models = {
        'Choquet_Weight': ChoquetClassifier(
            methode='Weight',
            optimizer=choquet_config.get('optimizer_weight', 'L-BFGS-B'),
            process_data=choquet_config.get('process_data', True),
            jac=choquet_config.get('jac', True)
        ),
        'Choquet_Power': ChoquetClassifier(
            methode='Power',
            optimizer=choquet_config.get('optimizer_power', 'L-BFGS-B'),
            process_data=choquet_config.get('process_data', True),
            jac=choquet_config.get('jac', True)
        ),
        'Choquet_Weight_TN3': ChoquetTnormClassifier(
            methode='Weight',
            optimizer=choquet_config.get('optimizer_tnorm', 'GD'),
            tnorm_c=3,
            alpha=choquet_config.get('alpha', None),
            process_data=choquet_config.get('process_data', True),
            jac=choquet_config.get('jac', True),
            niter=choquet_config.get('niter', 1000),
            stpa=choquet_config.get('stpa_weight', 0.005),
            stpy=choquet_config.get('stpy_weight', 0.005)
        ),
        'Choquet_Weight_TN6': ChoquetTnormClassifier(
            methode='Weight',
            optimizer=choquet_config.get('optimizer_tnorm', 'GD'),
            tnorm_c=6,
            alpha=choquet_config.get('alpha', None),
            process_data=choquet_config.get('process_data', True),
            jac=choquet_config.get('jac', True),
            niter=choquet_config.get('niter', 1000),
            stpa=choquet_config.get('stpa_weight', 0.005),
            stpy=choquet_config.get('stpy_weight', 0.005)
        ),
        'Choquet_Power_TN3': ChoquetTnormClassifier(
            methode='Power',
            optimizer=choquet_config.get('optimizer_tnorm', 'GD'),
            tnorm_c=3,
            alpha=choquet_config.get('alpha', None),
            process_data=choquet_config.get('process_data', True),
            jac=choquet_config.get('jac', True),
            niter=choquet_config.get('niter', 1000),
            stpa=choquet_config.get('stpa_power', 0.001),
            stpy=choquet_config.get('stpy_power', 0.001)
        ),
        'Choquet_Power_TN6': ChoquetTnormClassifier(
            methode='Power',
            optimizer=choquet_config.get('optimizer_tnorm', 'GD'),
            tnorm_c=6,
            alpha=choquet_config.get('alpha', None),
            process_data=choquet_config.get('process_data', True),
            jac=choquet_config.get('jac', True),
            niter=choquet_config.get('niter', 1000),
            stpa=choquet_config.get('stpa_power', 0.001),
            stpy=choquet_config.get('stpy_power', 0.001)
        ),
        'LogisticRegression': LogisticRegression(
            penalty=choquet_config.get('lr_penalty', None),
            max_iter=choquet_config.get('lr_max_iter', 1000),
            random_state=seed
        )
    }
    
    return models


def train_and_evaluate_single_experiment(
    result_dir: Path,
    config: Dict,
    verbose: bool = True
) -> Dict:
    """
    Train and evaluate models on a single experiment.
    
    Parameters
    ----------
    result_dir : Path
        Path to experiment results directory.
    config : dict
        Configuration dictionary.
    verbose : bool, default=True
        Whether to print progress.
    
    Returns
    -------
    dict
        Dictionary containing metrics and experiment parameters.
    """
    # Parse experiment parameters from folder name
    folder_name = result_dir.name
    params = parse_folder_name(folder_name)
    
    # Check if parent directory is a sweep directory
    parent_name = result_dir.parent.name
    if parent_name.startswith('sweep_'):
        # Extract sweep parameter and value
        # Format: sweep_param_name_value (e.g., sweep_window_size_7, sweep_max_samples_per_class_100)
        parts = parent_name.split('_')
        if len(parts) >= 3:
            # Find the index of 'sweep'
            sweep_idx = parts.index('sweep')
            # Everything after 'sweep' until the last part is the param name
            # The last part is the value
            param_parts = parts[sweep_idx + 1:]
            if len(param_parts) >= 2:
                param_value = param_parts[-1]
                param_name = '_'.join(param_parts[:-1])
                try:
                    # Try to convert to appropriate type
                    if '.' in param_value:
                        params[param_name] = float(param_value)
                    else:
                        params[param_name] = int(param_value)
                except ValueError:
                    params[param_name] = param_value
    
    if verbose:
        print(f"\nProcessing: {folder_name}")
        print(f"  Parameters: {params}")
    
    # Load data
    try:
        X_train, y_train, X_test, y_test = load_experiment_data(result_dir)
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None
    
    if verbose:
        print(f"  Data shape: Train={X_train.shape}, Test={X_test.shape}")
    
    # Create models
    seed = params.get('seed', 0)
    models = create_choquet_models(config, seed)
    
    # Train and evaluate each model
    results = params.copy()
    
    for model_name, model in models.items():
        try:
            # Train
            model.fit(X_train, y_train)
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_train = model.predict_proba(X_train)[:, 1]
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            train_auc = roc_auc_score(y_train, y_proba_train)
            test_auc = roc_auc_score(y_test, y_proba_test)
            
            # Store results
            results[f'{model_name}_train_acc'] = train_acc
            results[f'{model_name}_test_acc'] = test_acc
            results[f'{model_name}_train_f1'] = train_f1
            results[f'{model_name}_test_f1'] = test_f1
            results[f'{model_name}_train_auc'] = train_auc
            results[f'{model_name}_test_auc'] = test_auc
            
            if verbose:
                print(f"  {model_name}: Test Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        
        except Exception as e:
            print(f"  ERROR with {model_name}: {e}")
            results[f'{model_name}_train_acc'] = np.nan
            results[f'{model_name}_test_acc'] = np.nan
            results[f'{model_name}_train_f1'] = np.nan
            results[f'{model_name}_test_f1'] = np.nan
            results[f'{model_name}_train_auc'] = np.nan
            results[f'{model_name}_test_auc'] = np.nan
    
    return results


def process_results_directory(
    results_base_dir: Path,
    config: Dict,
    output_csv: Path = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process all experiment results in a directory.
    
    Parameters
    ----------
    results_base_dir : Path
        Base directory containing experiment results.
    config : dict
        Configuration dictionary.
    output_csv : Path, optional
        Path to save results CSV. If None, auto-generated.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : bool, default=True
        Whether to print progress.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all results.
    """
    # Find all experiment directories (including those in sweep subdirectories)
    exp_dirs = []
    
    # First check for direct seed directories (no sweep)
    direct_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and (d / 'train.npy').exists()]
    exp_dirs.extend(direct_dirs)
    
    # Then check for sweep directories
    sweep_dirs = [d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith('sweep_')]
    for sweep_dir in sweep_dirs:
        seed_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and (d / 'train.npy').exists()]
        exp_dirs.extend(seed_dirs)
    
    if len(exp_dirs) == 0:
        print(f"No experiment directories found in {results_base_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Process experiments in parallel
    if n_jobs == 1 or verbose:
        # Sequential processing for better progress display
        results_list = []
        for exp_dir in tqdm(exp_dirs, desc="Processing experiments"):
            result = train_and_evaluate_single_experiment(exp_dir, config, verbose=False)
            if result is not None:
                results_list.append(result)
    else:
        # Parallel processing
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(train_and_evaluate_single_experiment)(exp_dir, config, verbose=False)
            for exp_dir in tqdm(exp_dirs, desc="Processing experiments")
        )
        results_list = [r for r in results_list if r is not None]
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Sort by parameters
    sort_cols = [col for col in ['seed', 'noise', 'label_noise', 'data_noise'] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Save to CSV
    if output_csv is None:
        output_csv = results_base_dir / 'choquet_aggregation_results.csv'
    
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(f"Shape: {df.shape}")
    
    return df


def main(config_path: str, results_dir: str = None, n_jobs: int = -1):
    """
    Main function to process ensemble classifier results with Choquet aggregation.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.
    results_dir : str, optional
        Path to results directory. If None, uses config hash directory.
    n_jobs : int, default=-1
        Number of parallel jobs.
    """
    print("="*80)
    print("CHOQUET AGGREGATION EVALUATION")
    print("="*80)
    
    # Load configuration
    config = load_config(config_path)
    print(f"Configuration loaded from: {config_path}\n")
    
    # Determine results directory
    if results_dir is None:
        # Use config hash to find results directory
        import hashlib
        config_str = yaml.dump(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        results_base_dir = Path(__file__).parent.parent / 'results' / config_hash
    else:
        results_base_dir = Path(results_dir)
    
    if not results_base_dir.exists():
        print(f"ERROR: Results directory not found: {results_base_dir}")
        return
    
    print(f"Processing results from: {results_base_dir}\n")
    
    # Process all experiments
    df = process_results_directory(
        results_base_dir=results_base_dir,
        config=config,
        n_jobs=n_jobs,
        verbose=True
    )
    
    # Compile ensemble scores if save_scores is enabled
    if config.get('training', {}).get('save_scores', True):
        ensemble_scores_list = []
        for exp_dir in [d for d in results_base_dir.iterdir() if d.is_dir() and (d / 'scores.csv').exists()]:
            scores_df = pd.read_csv(exp_dir / 'scores.csv')
            ensemble_scores_list.append(scores_df)
        
        if ensemble_scores_list:
            ensemble_df = pd.concat(ensemble_scores_list, ignore_index=True)
            ensemble_csv = results_base_dir / 'ensemble_scores.csv'
            ensemble_df.to_csv(ensemble_csv, index=False)
            print(f"\nEnsemble scores saved to: {ensemble_csv}")
            
            # Delete individual scores.csv files from subdirectories
            for exp_dir in [d for d in results_base_dir.iterdir() if d.is_dir() and (d / 'scores.csv').exists()]:
                (exp_dir / 'scores.csv').unlink()
                print(f"Deleted: {exp_dir / 'scores.csv'}")
            
            # Print ensemble summary
            print("\n" + "="*80)
            print("ENSEMBLE CLASSIFIERS SUMMARY (Test F1)")
            print("="*80)
            
            test_f1_cols = [col for col in ensemble_df.columns if col == 'test_f1']
            if 'test_f1' in ensemble_df.columns and 'model' in ensemble_df.columns:
                summary_stats = ensemble_df.groupby('model')['test_f1'].agg(['mean', 'std']).round(4)
                print("\nPer Classifier:")
                print(summary_stats)
                
                overall_mean = ensemble_df['test_f1'].mean()
                print(f"\nOverall Mean (all classifiers): {overall_mean:.4f}")
                
                # AUC summary
                summary_auc = ensemble_df.groupby('model')['test_auc'].agg(['mean', 'std']).round(4)
                print("\nPer Classifier AUC:")
                print(summary_auc)
                
                overall_auc = ensemble_df['test_auc'].mean()
                print(f"\nOverall Mean AUC (all classifiers): {overall_auc:.4f}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CHOQUET AGGREGATION SUMMARY - F1 SCORES")
    print("="*80)
    
    # Identify F1 metric columns
    f1_cols = [col for col in df.columns if 'test_f1' in col]
    
    if f1_cols:
        print("\nTest F1 Metrics Summary:")
        summary_stats_f1 = df[f1_cols].agg(['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)])
        summary_stats_f1.index = ['mean', 'std', '25%', '50%', '75%']
        print(summary_stats_f1.round(4))
    
    print("\n" + "="*80)
    print("CHOQUET AGGREGATION SUMMARY - AUC-ROC")
    print("="*80)
    
    # Identify AUC metric columns
    auc_cols = [col for col in df.columns if 'test_auc' in col]
    
    if auc_cols:
        print("\nTest AUC-ROC Metrics Summary:")
        summary_stats_auc = df[auc_cols].agg(['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)])
        summary_stats_auc.index = ['mean', 'std', '25%', '50%', '75%']
        print(summary_stats_auc.round(4))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Choquet aggregation models on ensemble classifier outputs"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Path to results directory (default: use config hash)'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        results_dir=args.results_dir,
        n_jobs=args.n_jobs
    )
