#!/usr/bin/env python3
"""
Result aggregation utilities for Choquet experiments.

This module provides functions to aggregate results from multiple experiments
and compute statistics across different parameter configurations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import glob


def aggregate_multiple_results(
    results_base_dir: Path,
    pattern: str = "*/choquet_aggregation_results.csv"
) -> pd.DataFrame:
    """
    Aggregate results from multiple experiment directories.
    
    Parameters
    ----------
    results_base_dir : Path
        Base directory containing multiple result directories.
    pattern : str, default="*/choquet_aggregation_results.csv"
        Glob pattern to find result CSV files.
    
    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with all results.
    """
    csv_files = list(results_base_dir.glob(pattern))
    
    if len(csv_files) == 0:
        print(f"No CSV files found matching pattern: {pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} result files")
    
    # Load and concatenate all DataFrames
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Add directory name as experiment identifier
        df['experiment_dir'] = csv_file.parent.name
        dfs.append(df)
    
    aggregated_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Aggregated shape: {aggregated_df.shape}")
    
    return aggregated_df


def compute_statistics(
    df: pd.DataFrame,
    group_by: Union[str, List[str]] = None,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compute statistics (mean, std, median, quantiles) for metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    group_by : str or list of str, optional
        Column(s) to group by (e.g., 'noise', ['noise', 'label_noise']).
    metrics : list of str, optional
        List of metric columns to compute statistics for.
        If None, automatically detects columns containing '_acc' or '_f1'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with statistics for each group and metric.
    """
    # Identify metric columns if not specified
    if metrics is None:
        metrics = [col for col in df.columns if any(x in col for x in ['_acc', '_f1'])]
    
    if not metrics:
        print("No metrics found in DataFrame")
        return pd.DataFrame()
    
    # Define aggregation functions
    agg_functions = {
        'mean': 'mean',
        'std': 'std',
        'median': 'median',
        'q25': lambda x: x.quantile(0.25),
        'q75': lambda x: x.quantile(0.75),
        'min': 'min',
        'max': 'max',
        'count': 'count'
    }
    
    if group_by is None:
        # Compute statistics for all data
        stats = pd.DataFrame()
        for metric in metrics:
            for stat_name, stat_func in agg_functions.items():
                if callable(stat_func):
                    value = stat_func(df[metric])
                else:
                    value = df[metric].agg(stat_func)
                stats.loc[metric, stat_name] = value
        return stats
    
    else:
        # Group by specified columns
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Create aggregation dictionary for each metric
        agg_dict = {}
        for metric in metrics:
            agg_dict[metric] = list(agg_functions.values())
        
        # Compute grouped statistics
        stats = df.groupby(group_by).agg(agg_dict)
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        
        return stats


def create_comparison_table(
    df: pd.DataFrame,
    baseline_model: str = 'LogisticRegression',
    metric: str = 'test_f1'
) -> pd.DataFrame:
    """
    Create a comparison table showing relative performance to baseline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    baseline_model : str, default='LogisticRegression'
        Name of the baseline model.
    metric : str, default='test_f1'
        Metric to compare (e.g., 'test_f1', 'test_acc').
    
    Returns
    -------
    pd.DataFrame
        Comparison table with absolute and relative performance.
    """
    # Get all model columns for the specified metric
    model_cols = [col for col in df.columns if metric in col and col.startswith(('Choquet', 'Logistic'))]
    
    baseline_col = f'{baseline_model}_{metric}'
    if baseline_col not in model_cols:
        print(f"Baseline column {baseline_col} not found")
        return pd.DataFrame()
    
    comparison = pd.DataFrame()
    
    for col in model_cols:
        model_name = col.replace(f'_{metric}', '')
        
        # Absolute performance
        comparison[f'{model_name}_mean'] = [df[col].mean()]
        comparison[f'{model_name}_std'] = [df[col].std()]
        
        # Relative to baseline
        if col != baseline_col:
            diff = df[col] - df[baseline_col]
            comparison[f'{model_name}_vs_{baseline_model}_mean'] = [diff.mean()]
            comparison[f'{model_name}_vs_{baseline_model}_pct'] = [100 * diff.mean() / df[baseline_col].mean()]
    
    return comparison.T


def find_best_configurations(
    df: pd.DataFrame,
    metric: str = 'test_f1',
    top_k: int = 10
) -> pd.DataFrame:
    """
    Find the best performing configurations across all experiments.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    metric : str, default='test_f1'
        Metric to use for ranking.
    top_k : int, default=10
        Number of top configurations to return.
    
    Returns
    -------
    pd.DataFrame
        Top-k configurations sorted by metric.
    """
    # Get all model columns for the specified metric
    model_cols = [col for col in df.columns if metric in col and col.startswith(('Choquet', 'Logistic'))]
    
    if not model_cols:
        print(f"No metric columns found for {metric}")
        return pd.DataFrame()
    
    # Compute best score for each experiment
    df_copy = df.copy()
    df_copy['best_score'] = df_copy[model_cols].max(axis=1)
    df_copy['best_model'] = df_copy[model_cols].idxmax(axis=1)
    df_copy['best_model'] = df_copy['best_model'].str.replace(f'_{metric}', '')
    
    # Sort and return top-k
    param_cols = [col for col in df.columns if col not in model_cols]
    result_cols = param_cols + ['best_model', 'best_score'] + model_cols
    
    return df_copy[result_cols].nlargest(top_k, 'best_score')


def compute_win_rates(
    df: pd.DataFrame,
    metric: str = 'test_f1',
    baseline_model: str = 'LogisticRegression'
) -> pd.Series:
    """
    Compute win rates (percentage of experiments where model beats baseline).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    metric : str, default='test_f1'
        Metric to compare.
    baseline_model : str, default='LogisticRegression'
        Name of the baseline model.
    
    Returns
    -------
    pd.Series
        Win rates for each model.
    """
    model_cols = [col for col in df.columns if metric in col and col.startswith(('Choquet', 'Logistic'))]
    
    baseline_col = f'{baseline_model}_{metric}'
    if baseline_col not in model_cols:
        print(f"Baseline column {baseline_col} not found")
        return pd.Series()
    
    win_rates = {}
    for col in model_cols:
        model_name = col.replace(f'_{metric}', '')
        if col != baseline_col:
            wins = (df[col] > df[baseline_col]).sum()
            win_rates[model_name] = 100 * wins / len(df)
    
    return pd.Series(win_rates).sort_values(ascending=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate and analyze Choquet experiment results")
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Base directory containing experiment results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='aggregated_results.csv',
        help='Output CSV file name'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Aggregate results
    print("Aggregating results...")
    df = aggregate_multiple_results(results_dir)
    
    if not df.empty:
        # Save aggregated results
        output_path = results_dir / args.output
        df.to_csv(output_path, index=False)
        print(f"Aggregated results saved to: {output_path}")
        
        # Compute and display statistics
        print("\n" + "="*80)
        print("STATISTICS BY NOISE LEVEL")
        print("="*80)
        
        if 'noise' in df.columns:
            stats = compute_statistics(df, group_by='noise')
            print(stats)
        
        # Display win rates
        print("\n" + "="*80)
        print("WIN RATES (vs LogisticRegression)")
        print("="*80)
        win_rates = compute_win_rates(df, metric='test_f1')
        print(win_rates)
        
        # Find best configurations
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS")
        print("="*80)
        best = find_best_configurations(df, metric='test_f1', top_k=10)
        print(best)
