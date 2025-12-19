#!/usr/bin/env python3
"""
Ensemble Classifier Training Script

This script trains multiple classifiers on binary classification tasks
using configuration from YAML file, with support for parameter sweeps
and parallel execution across multiple seeds.
"""

import numpy as np
import yaml
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


from dataset.dataloader import load_and_extract_data, prepare_train_test_split
from dataset.load_dataset import MLDatasetLoader

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_config_hash(config: Dict) -> str:
    """
    Generate a hash from the config dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    str
        First 16 characters of the hash.
    """
    config_str = yaml.dump(config, sort_keys=True)
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:16]


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_from_config(model_config: Dict, random_state: int = None) -> Any:
    """
    Instantiate a model from configuration.
    
    Parameters
    ----------
    model_config : dict
        Model configuration with 'type' and 'params'.
    random_state : int, optional
        Random state to use for models that support it.
    
    Returns
    -------
    object
        Instantiated model.
    """
    model_type = model_config['type']
    params = model_config.get('params', {}).copy()
    
    # Add random_state if the model supports it and it's not already specified
    models_with_random_state = [
        'RandomForestClassifier', 'SVC', 'AdaBoostClassifier',
        'GradientBoostingClassifier', 'DecisionTreeClassifier', 'MLPClassifier'
    ]
    
    if random_state is not None and model_type in models_with_random_state:
        if 'random_state' not in params:
            params['random_state'] = random_state
    
    model_classes = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'SVC': SVC,
        'AdaBoostClassifier': AdaBoostClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'MLPClassifier': MLPClassifier,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](**params)


def train_and_evaluate_models(
    models_config: List[Dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = None,
    save_scores: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Train and evaluate all models.
    
    Parameters
    ----------
    models_config : list of dict
        List of model configurations.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    random_state : int, optional
        Random state to use for models.
    save_scores : bool, default=True
        Whether to save scores.
    
    Returns
    -------
    tuple
        (train_results, test_results, scores_dict)
    """
    # Flatten data
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    train_results = {'y_train': y_train}
    test_results = {'y_test': y_test}
    scores_dict = {} if save_scores else None
    
    for model_config in models_config:
        model_name = model_config['name']
        
        try:
            # Instantiate and train model
            model = get_model_from_config(model_config, random_state=random_state)
            model.fit(X_train_flat, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_flat)
            y_pred_test = model.predict(X_test_flat)
            
            # Probabilities
            y_proba_train = model.predict_proba(X_train_flat)[:, 1]
            y_proba_test = model.predict_proba(X_test_flat)[:, 1]
            
            # Store probabilities
            train_results[model_name] = y_proba_train
            test_results[model_name] = y_proba_test
            
            # Compute metrics
            if save_scores:
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)
                train_f1 = f1_score(y_train, y_pred_train, average='weighted')
                test_f1 = f1_score(y_test, y_pred_test, average='weighted')
                train_auc = roc_auc_score(y_train, y_proba_train)
                test_auc = roc_auc_score(y_test, y_proba_test)
                
                scores_dict[model_name] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'train_auc': train_auc,
                    'test_auc': test_auc
                }
                
                print(f"  [TRAIN] {model_name}: F1={test_f1:.4f}, AUC={test_auc:.4f}")
        
        except Exception as e:
            print(f"  ERROR with {model_name}: {e}")
            continue
    
    return train_results, test_results, scores_dict


def train_single_seed(
    seed: int,
    config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Dict,
    group_names: Dict,
    output_dir: Path,
    sweep_params: Dict = None
) -> None:
    """
    Train models for a single seed.
    
    Parameters
    ----------
    seed : int
        Random seed.
    config : dict
        Configuration dictionary.
    X : np.ndarray
        Feature array (preloaded).
    y : np.ndarray
        Labels array (preloaded).
    groups : np.ndarray
        Group identifiers (preloaded).
    class_names : dict
        Mapping from label to class name.
    group_names : dict
        Mapping from int to group name.
    output_dir : Path
        Output directory for results.
    sweep_params : dict, optional
        Parameters to override for sweep.
    """
    print(f"\n{'='*80}")
    print(f"SEED: {seed}")
    print(f"{'='*80}")
    
    # Prepare split parameters
    split_params = {
        'X': X,
        'y': y,
        'groups': groups,
        'class_names': class_names,
        'group_names': group_names,
        'seed': seed,
        'max_samples_per_class': config['dataset'].get('max_samples_per_class'),
        'n_splits': config['dataset']['n_splits'],
        'label_noise_percentage': config['noise']['label_noise_percentage'],
        'data_noise_std': config['noise']['data_noise_std'],
        'verbose': False
    }
    
    # Override with sweep parameters if provided
    if sweep_params:
        split_params.update(sweep_params)
    
    # Prepare train/test split
    try:
        data = prepare_train_test_split(**split_params)
    except Exception as e:
        print(f"ERROR preparing split for seed {seed}: {e}")
        return
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"  [TRAIN] Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    # Train and evaluate models
    train_results, test_results, scores_dict = train_and_evaluate_models(
        models_config=config['models'],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        save_scores=config['training'].get('save_scores', True)
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'train.npy', train_results)
    np.save(output_dir / 'test.npy', test_results)
    
    if scores_dict and config['training'].get('save_scores', True):
        import pandas as pd
        scores_df = pd.DataFrame(scores_dict).T
        scores_df.insert(0, 'model', scores_df.index)
        scores_df.insert(0, 'seed', seed)
        scores_df.to_csv(output_dir / 'scores.csv', index=False)
    
    print(f"Results saved to {output_dir}")


def run_single_sweep(
    config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Dict,
    group_names: Dict,
    base_output_dir: Path,
    param_name: str,
    param_value: Any,
    n_jobs: int = -1
) -> None:
    """
    Run training for a single sweep parameter value across all seeds.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    X : np.ndarray
        Feature array (preloaded).
    y : np.ndarray
        Labels array (preloaded).
    groups : np.ndarray
        Group identifiers (preloaded).
    class_names : dict
        Mapping from label to class name.
    group_names : dict
        Mapping from int to group name.
    base_output_dir : Path
        Base output directory.
    param_name : str
        Name of the parameter being swept.
    param_value : any
        Value of the parameter for this sweep.
    n_jobs : int, default=-1
        Number of parallel jobs.
    """
    print(f"\n{'='*80}")
    print(f"SWEEP: {param_name} = {param_value}")
    print(f"{'='*80}")
    
    # Create sweep directory
    sweep_dir = base_output_dir / f"sweep_{param_name}_{param_value}"
    
    # Prepare sweep parameters
    sweep_params = {param_name: param_value}
    
    # Run across all seeds in parallel
    n_seeds = config['training']['n_seeds']
    
    Parallel(n_jobs=n_jobs)(
        delayed(train_single_seed)(
            seed=seed,
            config=config,
            X=X,
            y=y,
            groups=groups,
            class_names=class_names,
            group_names=group_names,
            output_dir=sweep_dir / f"seed_{seed:02d}",
            sweep_params=sweep_params
        )
        for seed in tqdm(range(n_seeds), desc=f"Seeds ({param_name}={param_value})")
    )


def run_double_sweep(
    config: Dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Dict,
    group_names: Dict,
    base_output_dir: Path,
    n_jobs: int = -1
) -> None:
    """
    Run training for double parameter sweep.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    X : np.ndarray
        Feature array (preloaded).
    y : np.ndarray
        Labels array (preloaded).
    groups : np.ndarray
        Group identifiers (preloaded).
    class_names : dict
        Mapping from label to class name.
    group_names : dict
        Mapping from int to group name.
    base_output_dir : Path
        Base output directory.
    n_jobs : int, default=-1
        Number of parallel jobs.
    """
    param1_name = config['double_sweep']['param1_name']
    param1_values = config['double_sweep']['param1_values']
    param2_name = config['double_sweep']['param2_name']
    param2_values = config['double_sweep']['param2_values']
    
    print(f"\n{'#'*80}")
    print(f"# DOUBLE SWEEP")
    print(f"# {param1_name}: {param1_values}")
    print(f"# {param2_name}: {param2_values}")
    print(f"{'#'*80}")
    
    n_seeds = config['training']['n_seeds']
    
    # Loop over first parameter
    for param1_value in param1_values:
        # Loop over second parameter
        for param2_value in param2_values:
            print(f"\n{'='*80}")
            print(f"SWEEP: {param1_name}={param1_value}, {param2_name}={param2_value}")
            print(f"{'='*80}")
            
            # Create directory name
            sweep_dir = base_output_dir / f"{param1_name}_{param1_value}_{param2_name}_{param2_value}"
            
            # Prepare sweep parameters
            sweep_params = {
                param1_name: param1_value,
                param2_name: param2_value
            }
            
            # Run across all seeds in parallel
            Parallel(n_jobs=n_jobs)(
                delayed(train_single_seed)(
                    seed=seed,
                    config=config,
                    X=X,
                    y=y,
                    groups=groups,
                    class_names=class_names,
                    group_names=group_names,
                    output_dir=sweep_dir / f"seed_{seed:02d}",
                    sweep_params=sweep_params
                )
                for seed in tqdm(
                    range(n_seeds),
                    desc=f"Seeds ({param1_name}={param1_value}, {param2_name}={param2_value})"
                )
            )


def main(config_path: str, n_jobs: int = -1):
    """
    Main training function.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all cores).
    """
    print(f"Config: {config_path}\n")
    # Load configuration
    config = load_config(config_path)
    
    # Generate hash for output directory
    config_hash = get_config_hash(config)
    base_output_dir = Path(__file__).parent.parent.parent / 'src/results' / config_hash
    
    print(f"Config hash: {config_hash}")
    print(f"Output directory: {base_output_dir}\n")
    
    # Save config to output directory
    base_output_dir.mkdir(parents=True, exist_ok=True)
    with open(base_output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Load dataset and extract windows once
    print("Loading dataset and extracting windows...")
    loader = MLDatasetLoader(config['dataset']['data_path'])
    print(f"Dataset loaded: {loader.n_groups} groups")
    
    X, y, groups, class_names, group_names = load_and_extract_data(
        loader=loader,
        class_pair=tuple(config['dataset']['class_pair']),
        date=config['dataset']['date'],
        window_size=config['dataset']['window_size'],
        skip_optim_offset=config['dataset']['skip_optim_offset'],
        orbit=config['dataset'].get('orbit', 'DSC'),
        polarisation=config['dataset'].get('polarisation', ['HH', 'HV']),
        normalize=config['dataset'].get('normalize', False),
        remove_nodata=config['dataset'].get('remove_nodata', True),
        scale_type=config['dataset'].get('scale_type', 'amplitude'),
        max_mask_value=config['dataset'].get('max_mask_value', 1),
        max_mask_percentage=config['dataset'].get('max_mask_percentage', 0.0),
        min_valid_percentage=config['dataset'].get('min_valid_percentage', 100.0),
        verbose=True
    )
    
    print(f"Data extracted: {len(X)} samples\n")
    
    # Check for sweep configurations
    has_double_sweep = 'double_sweep' in config and config['double_sweep'] is not None
    has_single_sweep = 'sweep' in config and config['sweep'] is not None
    
    if has_double_sweep:
        # Run double sweep
        run_double_sweep(config, X, y, groups, class_names, group_names, base_output_dir, n_jobs=n_jobs)
    
    elif has_single_sweep:
        # Run single parameter sweep
        param_name = config['sweep']['param_name']
        param_values = config['sweep']['values']
        
        for param_value in param_values:
            run_single_sweep(
                config, X, y, groups, class_names, group_names, base_output_dir,
                param_name, param_value, n_jobs=n_jobs
            )
    
    else:
        # No sweep - run for all seeds with default parameters
        print("No sweep detected. Running with default parameters...\n")
        n_seeds = config['training']['n_seeds']
        
        Parallel(n_jobs=n_jobs)(
            delayed(train_single_seed)(
                seed=seed,
                config=config,
                X=X,
                y=y,
                groups=groups,
                class_names=class_names,
                group_names=group_names,
                output_dir=base_output_dir / f"seed_{seed:02d}",
                sweep_params=None
            )
            for seed in tqdm(range(n_seeds), desc="Seeds")
        )
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {base_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ensemble classifiers")
    parser.add_argument(
        '--config',
        type=str,
        default='../config/config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    
    args = parser.parse_args()
    
    main(config_path=args.config, n_jobs=args.n_jobs)
