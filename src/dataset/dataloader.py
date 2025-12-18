#!/usr/bin/env python3
"""
Custom DataLoader for Choquet Aggregation binary classification tasks.

This module provides functions to load and prepare data for binary classification
using specific class pairs, with train/test splitting respecting group structure.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import GroupKFold
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
    
    print(f"  [NOISE] Flipped {n_flip}/{n_samples} labels ({noise_percentage}%)")
    
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
    
    print(f"  [NOISE] Added Gaussian noise (std={noise_std}) to data")
    
    return X_noisy


def filter_max_samples_per_class(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_samples_per_class: Optional[int],
    seed: int = 42
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
            print(f"  [FILTER] Class {class_label}: {n_class} -> {max_samples_per_class} samples")
        else:
            selected_indices.extend(class_indices)
            print(f"  [FILTER] Class {class_label}: {n_class} samples (kept all)")
    
    selected_indices = np.array(selected_indices)
    
    return X[selected_indices], y[selected_indices], groups[selected_indices]


def split_train_test_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets using GroupKFold (first split only).

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
    shuffle : bool, default=True
        Whether to shuffle the groups before splitting.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Shuffle groups if requested
    if shuffle:
        np.random.seed(random_state)
        unique_groups = np.unique(groups)
        shuffled_groups = unique_groups.copy()
        np.random.shuffle(shuffled_groups)
        
        # Create mapping from old to new group indices
        group_mapping = {old: new for new, old in enumerate(shuffled_groups)}
        groups_shuffled = np.array([group_mapping[g] for g in groups])
    else:
        groups_shuffled = groups
    
    # Create GroupKFold splitter
    gkf = GroupKFold(n_splits=n_splits)
    
    # Get first split
    train_idx, test_idx = next(gkf.split(X, y, groups_shuffled))
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"\n  [SPLIT] Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  [SPLIT] Train groups: {len(np.unique(groups[train_idx]))}, Test groups: {len(np.unique(groups[test_idx]))}")
    
    return X_train, X_test, y_train, y_test


def extract_binary_class_data(
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
) -> Dict:
    """
    Extract windows for a binary classification task from two classes.

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
    dict
        Dictionary containing:
        - X: Feature array (N, window_size, window_size, n_pol)
        - y: Labels (0 or 1)
        - groups: Group identifiers (encoded as int)
        - masks: Mask arrays
        - class_names: Mapping from label to class name
        - group_names: Mapping from int to group name
    """
    class_0, class_1 = class_pair
    
    print(f"\n{'='*70}")
    print(f"Binary Classification: {class_0} (0) vs {class_1} (1)")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  - Date: {date}")
    print(f"  - Orbit: {orbit}")
    print(f"  - Polarization: {polarisation}")
    print(f"  - Window size: {window_size}x{window_size}")
    print(f"  - Scale type: {scale_type}")
    print(f"  - Max mask: {max_mask_value}, {max_mask_percentage}%")
    print(f"  - Min valid: {min_valid_percentage}%\n")
    
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
    pbar = tqdm(unique_groups, desc="Groups", unit="grp")
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
    masks = np.array(masks_all)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Extraction Results:")
        print(f"  - Total windows: {len(X)}")
        print(f"  - X.shape: {X.shape}")
        print(f"  - y distribution: {np.unique(y, return_counts=True)}")
        print(f"  - Unique groups: {len(np.unique(groups))}")
    
    return {
        'X': X,
        'y': y,
        'groups': groups,
        'masks': masks,
        'class_names': {0: class_0, 1: class_1},
        'group_names': {v: k for k, v in group_to_int.items()}
    }


def load_binary_classification_data(
    loader,
    class_pair: Tuple[str, str],
    date: str = '20200804',
    window_size: int = 32,
    skip_optim_offset: bool = True,
    seed: int = 42,
    max_samples_per_class: Optional[int] = None,
    n_splits: int = 5,
    label_noise_percentage: float = 0.0,
    data_noise_std: float = 0.0,
    verbose: bool = True
) -> Dict:
    """
    Load and prepare binary classification data with train/test split.

    This function extracts data for two classes, applies optional filtering,
    noise augmentation, and splits the data using GroupKFold.

    Parameters
    ----------
    loader : MLDatasetLoader
        Instance of the dataset loader.
    class_pair : tuple of str
        Tuple of two class names (e.g., ('ABL', 'ACC')).
        Supported pairs: ('ABL', 'ACC'), ('FOR', 'PLA'), ('ROC', 'ABL'), ('ROC', 'PLA')
    date : str, default='20200804'
        Acquisition date in 'YYYYMMDD' format.
    window_size : int, default=32
        Size of the extraction window.
    skip_optim_offset : bool, default=True
        Whether to skip window offset optimization.
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
        - X_train: Training features (N_train, window_size, window_size, n_pol)
        - X_test: Test features (N_test, window_size, window_size, n_pol)
        - y_train: Training labels
        - y_test: Test labels
        - groups_train: Training group identifiers
        - groups_test: Test group identifiers
        - class_names: Mapping from label to class name
        - group_names: Mapping from int to group name
        - metadata: Dictionary with all parameters
    """
    print(f"\n{'#'*70}")
    print(f"# BINARY CLASSIFICATION DATALOADER")
    print(f"{'#'*70}")
    print(f"Class pair: {class_pair[0]} vs {class_pair[1]}")
    print(f"Date: {date}")
    print(f"Seed: {seed}")
    print(f"Max samples per class: {max_samples_per_class}")
    
    # Extract data
    data = extract_binary_class_data(
        loader=loader,
        class_pair=class_pair,
        date=date,
        window_size=window_size,
        skip_optim_offset=skip_optim_offset,
        orbit='DSC',
        polarisation=['HH', 'HV'],
        normalize=False,
        remove_nodata=False,
        scale_type='log10',
        max_mask_value=1,
        max_mask_percentage=5.0,
        min_valid_percentage=100.0,
        verbose=verbose
    )
    
    X = data['X']
    y = data['y']
    groups = data['groups']
    
    # Apply max samples per class filter
    if max_samples_per_class is not None:
        print(f"\n{'='*70}")
        print("Filtering by max samples per class...")
        X, y, groups = filter_max_samples_per_class(
            X, y, groups, max_samples_per_class, seed
        )
        print(f"  After filtering: {len(X)} total samples")
    
    # Split train/test
    print(f"\n{'='*70}")
    print("Splitting train/test with GroupKFold...")
    X_train, X_test, y_train, y_test = split_train_test_groupkfold(
        X, y, groups,
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    
    # Get groups for train and test
    # Re-map to get the actual groups for train/test
    train_indices = []
    test_indices = []
    for i in range(len(X)):
        if any(np.array_equal(X[i], X_train[j]) for j in range(len(X_train))):
            train_indices.append(i)
        else:
            test_indices.append(i)
    
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]
    
    # Apply label noise to training set
    if label_noise_percentage > 0:
        print(f"\n{'='*70}")
        print("Applying label noise to training set...")
        y_train = add_label_noise(y_train, label_noise_percentage, seed)
    
    # Apply data noise to training set
    if data_noise_std > 0:
        print(f"\n{'='*70}")
        print("Applying data noise to training set...")
        X_train = add_data_noise(X_train, data_noise_std, seed)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL DATASET:")
    print(f"  Train: {len(X_train)} samples")
    print(f"    - Class 0 ({data['class_names'][0]}): {np.sum(y_train == 0)}")
    print(f"    - Class 1 ({data['class_names'][1]}): {np.sum(y_train == 1)}")
    print(f"  Test: {len(X_test)} samples")
    print(f"    - Class 0 ({data['class_names'][0]}): {np.sum(y_test == 0)}")
    print(f"    - Class 1 ({data['class_names'][1]}): {np.sum(y_test == 1)}")
    print(f"{'='*70}\n")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'groups_train': groups_train,
        'groups_test': groups_test,
        'class_names': data['class_names'],
        'group_names': data['group_names'],
        'metadata': {
            'class_pair': class_pair,
            'date': date,
            'window_size': window_size,
            'orbit': 'DSC',
            'polarisation': ['HH', 'HV'],
            'scale_type': 'log10',
            'seed': seed,
            'max_samples_per_class': max_samples_per_class,
            'n_splits': n_splits,
            'label_noise_percentage': label_noise_percentage,
            'data_noise_std': data_noise_std,
            'max_mask_value': 1,
            'max_mask_percentage': 5.0,
            'min_valid_percentage': 100.0
        }
    }


def example_usage():
    """Example usage of the binary classification dataloader."""
    from load_dataset import MLDatasetLoader
    
    # Initialize loader
    loader = MLDatasetLoader('../DATASET/PAZTSX_CRYO_ML.hdf5')
    
    # Define class pairs
    class_pairs = [
        ('ABL', 'ACC'),
        ('FOR', 'PLA'),
        ('ROC', 'ABL'),
        ('ROC', 'PLA')
    ]
    
    # Load data for the first pair
    data = load_binary_classification_data(
        loader=loader,
        class_pair=class_pairs[0],
        date='20200804',
        window_size=64,
        skip_optim_offset=True,
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
