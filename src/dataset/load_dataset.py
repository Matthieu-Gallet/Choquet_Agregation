#!/usr/bin/env python3
"""
Utilities for loading and manipulating HDF5 dataset optimized for ML.

Features:
- Fast extraction by class, temporal period
- Create temporal sequences for LSTM/Transformer
- Automatic data normalization
- Filter by metadata (angle, resolution, etc.)
- Extract sliding windows for learning
- Support filtering by mask values

Learning scenarios:
1. Temporal stacking classification (group k-fold)
2. Temporal prediction LSTM (time series)
3. Domain adaptation HH vs HV
4. Domain adaptation PAZ vs TerraSAR-X
"""

import h5py
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
from joblib import Parallel, delayed


class MLDatasetLoader:
    """Class to efficiently load the optimized HDF5 dataset with window extraction"""
    
    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: Path to HDF5 file
        """
        self.hdf5_path = hdf5_path
        self.file = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata in memory for fast access"""
        with h5py.File(self.hdf5_path, 'r') as f:
            meta = f['metadata']
            self.classes = json.loads(meta.attrs['classes'])
            self.n_groups = meta.attrs['n_total_groups']
            self.nodata = meta.attrs['nodata_value']
            
            # Charger les index
            self.class_index = {}
            for class_name in f['index/by_class'].keys():
                entries_json = f[f'index/by_class/{class_name}'].attrs['entries_json']
                self.class_index[class_name] = json.loads(entries_json)
            
            temp_ranges_json = f['index/temporal_ranges'].attrs['ranges_json']
            self.temporal_ranges = json.loads(temp_ranges_json)
    
    def __enter__(self):
        """Context manager entry"""
        self.file = h5py.File(self.hdf5_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.file:
            self.file.close()
    
    def get_group_info(self, group_name: str) -> Dict:
        """Get information for a group"""
        with h5py.File(self.hdf5_path, 'r') as f:
            if group_name not in f['data']:
                raise ValueError(f"Group {group_name} not found")
            
            group = f['data'][group_name]
            return {
                'class': group.attrs['class'],
                'latitude': group.attrs['latitude'],
                'longitude': group.attrs['longitude'],
                'elevation': group.attrs['elevation'],
                'orientation': group.attrs['orientation'],
                'slope': group.attrs['slope'],
                'orbits': list(group.keys())
            }
    
    def extract_windows(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        window_size: int,
        stride: Optional[int] = None,
        max_mask_value: int = 3,
        max_mask_percentage: float = 100.0,
        min_valid_percentage: float = 50.0,
        skip_optim_offset: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Extract windows from an image with mask filtering.
        Automatically optimize starting positions to maximize the number of valid windows.
        
        Args:
            image: Image (H, W) or (H, W, C) or (H, W, T) or (H, W, C, T)
            mask: Mask (H, W) or (H, W, T)
            window_size: Window size (square)
            stride: Stride step (if None, = window_size for non-overlapping)
            max_mask_value: Maximum accepted mask value (0, 1, 2, 3)
            max_mask_percentage: Max percentage of pixels with mask > max_mask_value
            min_valid_percentage: Min percentage of valid pixels (non nodata)
            skip_optim_offset: If True, skip offset optimization and use (0, 0)
        
        Returns:
            windows: Array of extracted windows
            window_masks: Array of corresponding masks
            positions: List of (y, x) positions for each window
        """
        if stride is None:
            stride = window_size
        
        # Handle dimensions
        if image.ndim == 2:  # (H, W)
            h, w = image.shape
            has_channels = False
            has_time = False
        elif image.ndim == 3:  # (H, W, C) ou (H, W, T)
            h, w, c = image.shape
            has_channels = True
            has_time = False
        elif image.ndim == 4:  # (H, W, C, T)
            h, w, c, t = image.shape
            has_channels = True
            has_time = True
        
        if mask.ndim == 2:
            mask_has_time = False
        elif mask.ndim == 3:
            mask_has_time = True
        
        # =====================================================================
        # OPTIMIZATION: Find the best starting offset (start_y, start_x)
        # to maximize the number of valid windows
        # =====================================================================
        if skip_optim_offset:
            # Skip optimization and use (0, 0) directly
            best_start_y = 0
            best_start_x = 0
            best_count = -1  # Unknown count when skipping optimization
        else:
            def count_valid_windows(start_y, start_x):
                """Count the number of valid windows for a given offset"""
                count = 0
                for y in range(start_y, h - window_size + 1, stride):
                    for x in range(start_x, w - window_size + 1, stride):
                        # Extract window for testing
                        if mask_has_time:
                            window_mask = mask[y:y+window_size, x:x+window_size, :]
                        else:
                            window_mask = mask[y:y+window_size, x:x+window_size]
                        
                        # Check mask criterion
                        if mask_has_time:
                            bad_pixels = np.any(window_mask > max_mask_value, axis=-1)
                        else:
                            bad_pixels = window_mask > max_mask_value
                        
                        bad_percentage = (np.sum(bad_pixels) / (window_size * window_size)) * 100.0
                        
                        # Check nodata
                        if image.ndim == 2:
                            window = image[y:y+window_size, x:x+window_size]
                        elif image.ndim == 3:
                            window = image[y:y+window_size, x:x+window_size, :]
                        elif image.ndim == 4:
                            window = image[y:y+window_size, x:x+window_size, :, :]
                        
                        # Check nodata and NaN (handle both invalid value types)
                        if has_time:
                            # Check if pixels are nodata OR NaN
                            is_invalid = (window == self.nodata) | np.isnan(window)
                            valid_pixels = np.all(~is_invalid, axis=-1)
                            if has_channels:
                                valid_pixels = np.all(valid_pixels, axis=-1)
                            valid_percentage = (np.sum(valid_pixels) / (window_size * window_size)) * 100.0
                        else:
                            if has_channels:
                                is_invalid = (window == self.nodata) | np.isnan(window)
                                valid_pixels = np.all(~is_invalid, axis=-1)
                            else:
                                is_invalid = (window == self.nodata) | np.isnan(window)
                                valid_pixels = ~is_invalid
                            valid_percentage = (np.sum(valid_pixels) / (window_size * window_size)) * 100.0
                        
                        # Count if criteria are met
                        if bad_percentage <= max_mask_percentage and valid_percentage >= min_valid_percentage:
                            count += 1
                
                return count
            
            # Test all possible offsets (0 to stride-1 for each dimension)
            # OPTIMIZATION: Parallelization with joblib
            best_count = 0
            best_start_y = 0
            best_start_x = 0
            
            # Limit search to stride (or window_size if non-overlapping)
            max_offset = min(stride, window_size)
            
            # Create list of all offsets to test
            offsets_to_test = []
            for start_y in range(max_offset):
                for start_x in range(max_offset):
                    if start_y + window_size <= h and start_x + window_size <= w:
                        offsets_to_test.append((start_y, start_x))
            
            # Calculate in parallel the number of valid windows for each offset
            counts = Parallel(n_jobs=-1)(
                delayed(count_valid_windows)(start_y, start_x)
                for start_y, start_x in tqdm(offsets_to_test, desc="Optimizing offset", leave=False)
            )
            
            # Find the best offset
            if len(counts) > 0:
                best_idx = np.argmax(counts)
                best_start_y, best_start_x = offsets_to_test[best_idx]
                best_count = counts[best_idx]
        
        # =====================================================================
        # EXTRACTION with the best offset found
        # =====================================================================
        windows = []
        window_masks = []
        positions = []
        
        # Sweep the image with optimal offset
        for y in range(best_start_y, h - window_size + 1, stride):
            for x in range(best_start_x, w - window_size + 1, stride):
                # Extract window
                if image.ndim == 2:
                    window = image[y:y+window_size, x:x+window_size]
                elif image.ndim == 3:
                    window = image[y:y+window_size, x:x+window_size, :]
                elif image.ndim == 4:
                    window = image[y:y+window_size, x:x+window_size, :, :]
                
                if mask_has_time:
                    window_mask = mask[y:y+window_size, x:x+window_size, :]
                else:
                    window_mask = mask[y:y+window_size, x:x+window_size]
                
                # Check mask criterion
                # Pixels with mask > max_mask_value
                if mask_has_time:
                    bad_pixels = np.any(window_mask > max_mask_value, axis=-1)
                else:
                    bad_pixels = window_mask > max_mask_value
                
                bad_percentage = (np.sum(bad_pixels) / (window_size * window_size)) * 100.0
                
                # Check nodata and NaN (handle both invalid value types)
                if has_time:
                    # Check if pixels are nodata OR NaN
                    is_invalid = (window == self.nodata) | np.isnan(window)
                    valid_pixels = np.all(~is_invalid, axis=-1)
                    if has_channels:
                        valid_pixels = np.all(valid_pixels, axis=-1)
                    valid_percentage = (np.sum(valid_pixels) / (window_size * window_size)) * 100.0
                else:
                    if has_channels:
                        is_invalid = (window == self.nodata) | np.isnan(window)
                        valid_pixels = np.all(~is_invalid, axis=-1)
                    else:
                        is_invalid = (window == self.nodata) | np.isnan(window)
                        valid_pixels = ~is_invalid
                    valid_percentage = (np.sum(valid_pixels) / (window_size * window_size)) * 100.0
                
                # Accept window if criteria are met (use strict inequalities for determinism)
                if bad_percentage <= max_mask_percentage and valid_percentage >= min_valid_percentage:
                    # Ensure float32 type and keep NaN values as-is
                    window = window.astype(np.float32)
                    
                    windows.append(window)
                    window_masks.append(window_mask)
                    positions.append((y, x))
        
        if len(windows) == 0:
            return None, None, []
        
        return np.array(windows), np.array(window_masks), positions
    
    def load_data(
        self,
        group_name: str,
        orbit: str = 'DSC',
        polarisation: Union[str, List[str]] = 'HH',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        normalize: bool = False,
        remove_nodata: bool = True,
        scale_type: str = 'intensity'
    ) -> Dict:
        """
        Load data for a specific group.
        
        Args:
            group_name: Group name (e.g. 'ABL001')
            orbit: 'ASC' or 'DSC'
            polarisation: 'HH', 'HV' or ['HH', 'HV'] for dual-pol
            start_date: Start date (format 'YYYYMMDD')
            end_date: End date (format 'YYYYMMDD')
            normalize: If True, normalize with pre-calculated stats
            remove_nodata: If True, replace nodata with NaN
            scale_type: 'intensity' (default), 'amplitude' (data**0.5), or 'log10' (log10 scale)
        
        Returns:
            Dict containing: images, masks, timestamps, metadata
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            # Support dual-pol
            if isinstance(polarisation, list):
                # Dual-pol: load HH and HV with aligned timestamps
                data_list = []
                for pol in polarisation:
                    path = f'data/{group_name}/{orbit}/{pol}'
                    if path not in f:
                        raise ValueError(f"Path {path} not found in dataset")
                    data_list.append(f[path])
                
                # Trouver les timestamps communs
                timestamps_hh = data_list[0]['timestamps'][:]
                timestamps_hv = data_list[1]['timestamps'][:]
                
                # Intersection of timestamps
                common_ts = np.intersect1d(timestamps_hh, timestamps_hv)
                
                if len(common_ts) == 0:
                    raise ValueError(f"No common timestamps between HH and HV for {group_name}")
                
                # Filter by dates if specified
                if start_date or end_date:
                    mask_ts = np.ones(len(common_ts), dtype=bool)
                    if start_date:
                        mask_ts &= common_ts >= start_date.encode('utf-8')
                    if end_date:
                        mask_ts &= common_ts <= end_date.encode('utf-8')
                    common_ts = common_ts[mask_ts]
                
                if len(common_ts) == 0:
                    raise ValueError(f"No data in specified date range")
                
                # Load data for common timestamps
                images_list = []
                masks_list = []
                angles_list = []
                
                # First, determine minimum dimensions
                min_h, min_w = None, None
                for pol, data_pol in zip(polarisation, data_list):
                    ts_pol = data_pol['timestamps'][:]
                    indices = [np.where(ts_pol == ts)[0][0] for ts in common_ts]
                    img_pol = data_pol['images'][:, :, indices]
                    h, w, t = img_pol.shape
                    if min_h is None:
                        min_h, min_w = h, w
                    else:
                        min_h = min(min_h, h)
                        min_w = min(min_w, w)
                
                # Load and crop
                for pol, data_pol in zip(polarisation, data_list):
                    # Find indices
                    ts_pol = data_pol['timestamps'][:]
                    indices = [np.where(ts_pol == ts)[0][0] for ts in common_ts]
                    
                    # Load and crop to min_h, min_w
                    img_pol = data_pol['images'][:min_h, :min_w, indices]
                    mask_pol = data_pol['masks'][:min_h, :min_w, indices]
                    
                    images_list.append(img_pol)
                    masks_list.append(mask_pol)
                    
                    if pol == polarisation[0]:  # Only once
                        angles_list = data_pol['angles_incidence'][:][indices]
                
                # Stack in last dimension: (H, W, T, 2)
                images = np.stack(images_list, axis=-1)
                
                # Mask: take max between HH and HV
                masks = np.maximum(masks_list[0], masks_list[1])
                
                timestamps = common_ts
                angles = angles_list
                
                metadata = {
                    'polarisation': polarisation,
                    'dual_pol': True
                }
                
            else:
                # Single pol
                path = f'data/{group_name}/{orbit}/{polarisation}'
                
                if path not in f:
                    raise ValueError(f"Path {path} not found in dataset")
                
                pol_data = f[path]
                
                # Load data
                images = pol_data['images'][:]
                masks = pol_data['masks'][:]
                timestamps = pol_data['timestamps'][:]
                angles = pol_data['angles_incidence'][:]
                
                # Filter by dates
                if start_date or end_date:
                    mask_ts = np.ones(len(timestamps), dtype=bool)
                    if start_date:
                        mask_ts &= timestamps >= start_date.encode('utf-8')
                    if end_date:
                        mask_ts &= timestamps <= end_date.encode('utf-8')
                    
                    if not np.any(mask_ts):
                        raise ValueError(f"No data in specified date range")
                    
                    images = images[:, :, mask_ts]
                    masks = masks[:, :, mask_ts]
                    timestamps = timestamps[mask_ts]
                    angles = angles[mask_ts]
                
                metadata = {
                    'mean': pol_data.attrs['stat_mean'],
                    'std': pol_data.attrs['stat_std'],
                    'min': pol_data.attrs['stat_min'],
                    'max': pol_data.attrs['stat_max'],
                    'n_samples': pol_data.attrs['n_timestamps'],
                    'polarisation': polarisation,
                    'dual_pol': False
                }
            
            # Replace nodata BEFORE applying transformations
            if remove_nodata:
                images = np.where(images == self.nodata, np.nan, images)
            
            # Apply scale transformation
            if scale_type == 'amplitude':
                # Use sqrt and handle NaN properly
                images_transformed = np.where(images >= 0, np.sqrt(images), np.nan)
                images = images_transformed.astype(np.float32)
            elif scale_type == 'log10':
                # Handle both NaN and non-positive values
                images = np.where(images > 0, np.log10(images), np.nan)
            # If 'intensity', keep as is (default)
            
            if normalize and not isinstance(polarisation, list):
                mean = metadata['mean']
                std = metadata['std']
                if std > 0:
                    images = (images - mean) / std
            
            return {
                'images': images,
                'masks': masks,
                'timestamps': [t.decode('utf-8') for t in timestamps],
                'angles_incidence': angles,
                'metadata': metadata,
                'group': group_name,
                'orbit': orbit
            }
    
    def get_groups_by_class(self, class_name: str) -> List[str]:
        """Return the list of groups for a given class"""
        if class_name not in self.class_index:
            return []
        return [entry['group'] for entry in self.class_index[class_name]]
    
    def get_all_groups_with_classes(self) -> Dict[str, str]:
        """Return a dictionary {group_name: class_name}"""
        group_to_class = {}
        for class_name in self.classes:
            for group in self.get_groups_by_class(class_name):
                group_to_class[group] = class_name
        return group_to_class
    
    def get_statistics_summary(self) -> Dict:
        """Return a summary of dataset statistics"""
        stats = {
            'by_class': {},
            'global': {
                'n_groups': self.n_groups,
                'n_classes': len(self.classes),
            }
        }
        
        # Stats by class
        for class_name in self.classes:
            groups = self.get_groups_by_class(class_name)
            stats['by_class'][class_name] = {
                'n_groups': len(groups),
                'groups': groups
            }
    
        
        return stats
