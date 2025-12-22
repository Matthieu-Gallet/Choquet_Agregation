#!/usr/bin/env python3
"""
Custom DataLoader for Choquet Aggregation binary classification tasks.

This module provides functions to load and prepare data for binary classification
using specific class pairs, with train/test splitting respecting group structure.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


def add_label_noise(y: np.ndarray, noise_percentage: float, seed: int = 42) -> np.ndarray:
    """
    Add noise to labels by randomly flipping a percentage of them.

    Parameters
    ----------
    y : np.ndarray
        Original labels (0 or 1).
    noise_percentage : float
        Percentage of labels to flip (0-100).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy labels.
    """
    if noise_percentage <= 0:
        return y.copy()
    
    np.random.seed(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_flip = int(n_samples * noise_percentage / 100.0)
    
    # Randomly select indices to flip
    flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)
    
    # Flip the labels (0 -> 1, 1 -> 0)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    
    return y_noisy


def add_data_noise(X: np.ndarray, noise_std: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to the data.

    Parameters
    ----------
    X : np.ndarray
        Original data array.
    noise_std : float
        Standard deviation of the Gaussian noise to add.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy data.
    """
    if noise_std <= 0:
        return X.copy()
    
    np.random.seed(seed)
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = X + noise
    
    return X_noisy


def filter_max_samples_per_class(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_samples_per_class: Optional[int],
    seed: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter the dataset to have at most max_samples_per_class for each class.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Labels array.
    groups : np.ndarray
        Group identifiers.
    max_samples_per_class : int or None
        Maximum number of samples per class. If None, no filtering.
    seed : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print information.

    Returns
    -------
    tuple
        (X_filtered, y_filtered, groups_filtered)
    """
    if max_samples_per_class is None:
        return X, y, groups
    
    np.random.seed(seed)
    unique_classes = np.unique(y)
    selected_indices = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        n_class = len(class_indices)
        
        if n_class > max_samples_per_class:
            # Randomly sample max_samples_per_class indices
            sampled_indices = np.random.choice(
                class_indices, 
                size=max_samples_per_class, 
                replace=False
            )
            selected_indices.extend(sampled_indices)
            if verbose:
                print(f"  [FILTER] Class {class_label}: {n_class} -> {max_samples_per_class} samples")
        else:
            selected_indices.extend(class_indices)
            if verbose:
                print(f"  [FILTER] Class {class_label}: {n_class} samples (kept all)")
    
    selected_indices = np.array(selected_indices)
    
    return X[selected_indices], y[selected_indices], groups[selected_indices]


def split_train_test_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets using GroupKFold.
    
    Uses the first fold from StratifiedGroupKFold for train/test split.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Labels array.
    groups : np.ndarray
        Group identifiers for GroupKFold.
    n_splits : int, default=5
        Number of folds for GroupKFold.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print information.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, groups_train, groups_test)
    """
    # Use seed to determine both shuffle and fold selection
    # For consecutive seeds (0,1,2,3,4 with n_splits=5), they share the same shuffle
    # but iterate through different folds
    base_seed = random_state // n_splits  # Determines the shuffle
    fold_idx = random_state % n_splits    # Determines which fold to use
    
    # Shuffle data with base_seed
    np.random.seed(base_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    groups = groups[indices]
    
    # Create StratifiedGroupKFold splitter with shuffle using base_seed
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)
    
    # Generate all folds and select the one corresponding to fold_idx
    folds = list(gkf.split(X, y, groups))
    train_idx, test_idx = folds[fold_idx]
    
    if verbose:
        print(f"  [SPLIT] Selected fold {fold_idx} from {n_splits} splits (base_seed={base_seed})")
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    
    if verbose:
        unique_classes = np.unique(y)
        print(f"  [SPLIT] Using first fold from {n_splits} splits")
        print(f"  [SPLIT] Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        for class_id in unique_classes:
            train_count = np.sum(y_train == class_id)
            test_count = np.sum(y_test == class_id)
            train_ratio = train_count / len(y_train) * 100
            test_ratio = test_count / len(y_test) * 100
            print(f"  [SPLIT] Class {class_id}: Train={train_count} ({train_ratio:.1f}%), Test={test_count} ({test_ratio:.1f}%)")
        print(f"  [SPLIT] Train groups: {len(np.unique(groups_train))}, Test groups: {len(np.unique(groups_test))}")
    
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def load_and_extract_data(
    loader,
    class_pair: Tuple[str, str],
    date: str,
    window_size: int = 32,
    skip_optim_offset: bool = True,
    orbit: str = 'DSC',
    polarisation: List[str] = ['HH', 'HV'],
    normalize: bool = False,
    remove_nodata: bool = False,
    scale_type: str = 'log10',
    max_mask_value: int = 1,
    max_mask_percentage: float = 5.0,
    min_valid_percentage: float = 100.0,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    """
    Load data and extract windows for a binary classification task.

    Parameters
    ----------
    loader : MLDatasetLoader
        Instance of the dataset loader.
    class_pair : tuple of str
        Tuple of two class names (e.g., ('ABL', 'ACC')).
    date : str
        Acquisition date in 'YYYYMMDD' format.
    window_size : int, default=32
        Size of the extraction window.
    skip_optim_offset : bool, default=True
        Whether to skip window offset optimization.
    orbit : str, default='DSC'
        Orbit type ('ASC' or 'DSC').
    polarisation : list of str, default=['HH', 'HV']
        List of polarizations to use.
    normalize : bool, default=False
        Whether to normalize the data.
    remove_nodata : bool, default=False
        Whether to remove nodata values.
    scale_type : str, default='log10'
        Scaling type ('intensity', 'amplitude', or 'log10').
    max_mask_value : int, default=1
        Maximum accepted mask value.
    max_mask_percentage : float, default=5.0
        Maximum percentage of pixels with mask > max_mask_value.
    min_valid_percentage : float, default=100.0
        Minimum percentage of valid pixels.
    verbose : bool, default=True
        Whether to display progress information.

    Returns
    -------
    tuple
        (X, y, groups, class_names, group_names)
        - X: Feature array (N, window_size, window_size, n_pol)
        - y: Labels (0 or 1)
        - groups: Group identifiers (encoded as int)
        - class_names: Mapping from label to class name
        - group_names: Mapping from int to group name
    """
    class_0, class_1 = class_pair
    
    if verbose:
        print(f"  [DATA] Binary classification: {class_0} (0) vs {class_1} (1)")
        print(f"  [DATA] Parameters: date={date}, orbit={orbit}, window={window_size}x{window_size}, scale={scale_type}")
    
    X_all = []
    y_all = []
    groups_all = []
    masks_all = []
    
    # Get groups for each class
    groups_class_0 = loader.get_groups_by_class(class_0)
    groups_class_1 = loader.get_groups_by_class(class_1)
    
    all_groups = groups_class_0 + groups_class_1
    unique_groups = sorted(list(set(all_groups)))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    
    # Create group -> class mapping
    group_to_class = {}
    for group in groups_class_0:
        group_to_class[group] = (class_0, 0)
    for group in groups_class_1:
        group_to_class[group] = (class_1, 1)
    
    # Process all groups
    pbar = tqdm(unique_groups, desc="Groups", unit="grp", disable=not verbose)
    for group_name in pbar:
        class_name, class_label = group_to_class[group_name]
        
        if verbose:
            pbar.set_postfix_str(f"{group_name} ({class_name})")
        
        try:
            # Load data
            data = loader.load_data(
                group_name=group_name,
                orbit=orbit,
                polarisation=polarisation,
                start_date=date,
                end_date=date,
                normalize=normalize,
                remove_nodata=remove_nodata,
                scale_type=scale_type
            )
            
            if data['images'].shape[2] == 0:
                continue
            
            # Take the single acquisition
            if len(polarisation) == 1:
                images = data['images'][:, :, 0]  # (H, W)
                images = images[:, :, np.newaxis]  # (H, W, 1)
            else:
                images = data['images'][:, :, 0, :]  # (H, W, n_pol)
            
            masks = data['masks'][:, :, 0]  # (H, W)
            
            # Extract windows
            windows, window_masks, positions = loader.extract_windows(
                image=images,
                mask=masks,
                window_size=window_size,
                stride=window_size,
                max_mask_value=max_mask_value,
                max_mask_percentage=max_mask_percentage,
                min_valid_percentage=min_valid_percentage,
                skip_optim_offset=skip_optim_offset
            )
            
            if windows is None:
                continue
            
            n_windows = len(windows)
            
            # Store each window
            for idx in range(n_windows):
                X_all.append(windows[idx])
                y_all.append(class_label)
                groups_all.append(group_to_int[group_name])
                masks_all.append(window_masks[idx])
            
        except Exception as e:
            if verbose:
                print(f"  Error with {group_name}: {e}")
            continue
    
    if len(X_all) == 0:
        raise ValueError("No windows extracted. Check parameters.")
    
    # Convert to arrays
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    groups = np.array(groups_all, dtype=np.int32)
    
    class_names = {0: class_0, 1: class_1}
    group_names = {v: k for k, v in group_to_int.items()}
    
    if verbose:
        print(f"  [DATA] Extracted: {len(X)} windows, shape={X.shape}, groups={len(np.unique(groups))}")
    
    return X, y, groups, class_names, group_names


def prepare_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Dict,
    group_names: Dict,
    seed: int = 42,
    max_samples_per_class: Optional[int] = None,
    n_splits: int = 5,
    label_noise_percentage: float = 0.0,
    data_noise_std: float = 0.0,
    verbose: bool = True
) -> Dict:
    """
    Prepare train/test split from loaded data.

    This function applies optional filtering, noise augmentation,
    and splits the data using GroupKFold.

    Parameters
    ----------
    X : np.ndarray
        Feature array.
    y : np.ndarray
        Labels array.
    groups : np.ndarray
        Group identifiers.
    class_names : dict
        Mapping from label to class name.
    group_names : dict
        Mapping from int to group name.
    seed : int, default=42
        Random seed for reproducibility.
    max_samples_per_class : int or None, default=None
        Maximum number of samples per class. If None, use all samples.
    n_splits : int, default=5
        Number of folds for GroupKFold.
    label_noise_percentage : float, default=0.0
        Percentage of labels to flip (0-100). Applied only to training set.
    data_noise_std : float, default=0.0
        Standard deviation of Gaussian noise to add to data. Applied only to training set.
    verbose : bool, default=True
        Whether to display progress information.

    Returns
    -------
    dict
        Dictionary containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training labels
        - y_test: Test labels
        - groups_train: Training group identifiers
        - groups_test: Test group identifiers
        - class_names: Mapping from label to class name
        - group_names: Mapping from int to group name
    """
    if verbose:
        print(f"  [TRAIN] Preparing split (seed={seed}, max_samples={max_samples_per_class})")
    
    # Split train/test
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_train_test_groupkfold(
        X, y, groups,
        n_splits=n_splits,
        random_state=seed,
        verbose=verbose
    )
    
    # Apply max samples per class filter to training set only
    if max_samples_per_class is not None:
        if verbose:
            print(f"  [TRAIN] Filtering to max {max_samples_per_class} samples per class...")
        X_train, y_train, groups_train = filter_max_samples_per_class(
            X_train, y_train, groups_train, max_samples_per_class, seed, verbose=verbose
        )
    
    # Apply label noise to training set
    if label_noise_percentage > 0:
        if verbose:
            print(f"  [TRAIN] Adding label noise ({label_noise_percentage}%)...")
        y_train = add_label_noise(y_train, label_noise_percentage, seed)
    
    # Apply data noise to training set
    if data_noise_std > 0:
        if verbose:
            print(f"  [TRAIN] Adding data noise (std={data_noise_std})...")
        X_train = add_data_noise(X_train, data_noise_std, seed)
    
    # Final summary
    if verbose:
        print(f"  [DATA] Train: {len(X_train)} samples ({class_names[0]}:{np.sum(y_train == 0)}, {class_names[1]}:{np.sum(y_train == 1)})")
        print(f"  [DATA] Test: {len(X_test)} samples ({class_names[0]}:{np.sum(y_test == 0)}, {class_names[1]}:{np.sum(y_test == 1)})")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'groups_train': groups_train,
        'groups_test': groups_test,
        'class_names': class_names,
        'group_names': group_names
    }


def example_usage():
    """Example usage of the binary classification dataloader."""
    from load_dataset import MLDatasetLoader
    
    # Initialize loader
    loader = MLDatasetLoader('../DATASET/PAZTSX_CRYO_ML.hdf5')
    
    # Load and extract data once
    X, y, groups, class_names, group_names = load_and_extract_data(
        loader=loader,
        class_pair=('ABL', 'ACC'),
        date='20200804',
        window_size=64,
        skip_optim_offset=True,
        verbose=True
    )
    
    # Prepare train/test split
    data = prepare_train_test_split(
        X=X,
        y=y,
        groups=groups,
        class_names=class_names,
        group_names=group_names,
        seed=42,
        max_samples_per_class=100,
        n_splits=5,
        label_noise_percentage=5.0,
        data_noise_std=0.01,
        verbose=True
    )
    
    print("Data loaded successfully!")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")


if __name__ == "__main__":
    example_usage()
