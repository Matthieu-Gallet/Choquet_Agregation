#!/usr/bin/env python3
"""
Visualization utilities for Choquet aggregation results.

This module provides functions to create boxplots and comparison plots
for analyzing Choquet aggregation performance across different configurations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def create_boxplots(
    df: pd.DataFrame,
    metric: str = 'test_f1',
    group_by: str = None,
    output_dir: Path = None,
    figsize: Tuple[int, int] = (15, 6),
    show_points: bool = True
) -> plt.Figure:
    """
    Create boxplots comparing different models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    metric : str, default='test_f1'
        Metric to plot (e.g., 'test_f1', 'test_acc').
    group_by : str, optional
        Column to group by (e.g., 'noise', 'seed').
    output_dir : Path, optional
        Directory to save plots. If None, plots are not saved.
    figsize : tuple, default=(15, 6)
        Figure size (width, height).
    show_points : bool, default=True
        Whether to show individual data points.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    # Get all model columns for the specified metric
    model_cols = [col for col in df.columns if metric in col and col.startswith(('Choquet', 'Logistic'))]
    
    if not model_cols:
        print(f"No metric columns found for {metric}")
        return None
    
    # Prepare data for plotting
    plot_data = []
    for col in model_cols:
        model_name = col.replace(f'_{metric}', '')
        for idx, value in df[col].items():
            row = {'Model': model_name, 'Value': value}
            if group_by and group_by in df.columns:
                row[group_by] = df.loc[idx, group_by]
            plot_data.append(row)
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    if group_by and group_by in df.columns:
        # Create subplots for each group
        groups = sorted(plot_df[group_by].unique())
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(figsize[0], figsize[1]), sharey=True)
        
        if n_groups == 1:
            axes = [axes]
        
        for ax, group in zip(axes, groups):
            group_data = plot_df[plot_df[group_by] == group]
            
            # Create boxplot
            sns.boxplot(
                data=group_data,
                x='Model',
                y='Value',
                ax=ax,
                palette='Set2'
            )
            
            # Overlay points if requested
            if show_points:
                sns.stripplot(
                    data=group_data,
                    x='Model',
                    y='Value',
                    ax=ax,
                    color='black',
                    alpha=0.3,
                    size=3
                )
            
            ax.set_title(f'{group_by} = {group}')
            ax.set_xlabel('')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    else:
        # Single plot without grouping
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        sns.boxplot(
            data=plot_df,
            x='Model',
            y='Value',
            ax=ax,
            palette='Set2'
        )
        
        # Overlay points if requested
        if show_points:
            sns.stripplot(
                data=plot_df,
                x='Model',
                y='Value',
                ax=ax,
                color='black',
                alpha=0.3,
                size=3
            )
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        group_suffix = f'_{group_by}' if group_by else ''
        filename = f'boxplot_{metric}{group_suffix}.png'
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / filename}")
    
    return fig


def create_comparison_plots(
    df: pd.DataFrame,
    baseline_model: str = 'LogisticRegression',
    metric: str = 'test_f1',
    group_by: str = None,
    output_dir: Path = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Create comparison plots showing performance relative to baseline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    baseline_model : str, default='LogisticRegression'
        Name of the baseline model.
    metric : str, default='test_f1'
        Metric to compare.
    group_by : str, optional
        Column to group by (e.g., 'noise').
    output_dir : Path, optional
        Directory to save plots.
    figsize : tuple, default=(12, 5)
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    model_cols = [col for col in df.columns if metric in col and col.startswith(('Choquet', 'Logistic'))]
    
    baseline_col = f'{baseline_model}_{metric}'
    if baseline_col not in model_cols:
        print(f"Baseline column {baseline_col} not found")
        return None
    
    # Calculate differences from baseline
    diff_data = []
    for col in model_cols:
        if col != baseline_col:
            model_name = col.replace(f'_{metric}', '')
            diff = df[col] - df[baseline_col]
            
            for idx, value in diff.items():
                row = {'Model': model_name, 'Difference': value}
                if group_by and group_by in df.columns:
                    row[group_by] = df.loc[idx, group_by]
                diff_data.append(row)
    
    diff_df = pd.DataFrame(diff_data)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if group_by and group_by in df.columns:
        # Plot 1: Difference by group
        sns.boxplot(
            data=diff_df,
            x=group_by,
            y='Difference',
            hue='Model',
            ax=ax1,
            palette='Set2'
        )
        ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_title(f'Performance vs {baseline_model}')
        ax1.set_ylabel(f'Δ {metric.replace("_", " ").title()}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win rate by group
        win_data = []
        for group in sorted(diff_df[group_by].unique()):
            group_data = diff_df[diff_df[group_by] == group]
            for model in group_data['Model'].unique():
                model_data = group_data[group_data['Model'] == model]
                wins = (model_data['Difference'] > 0).sum()
                win_rate = 100 * wins / len(model_data)
                win_data.append({
                    group_by: group,
                    'Model': model,
                    'Win Rate (%)': win_rate
                })
        
        win_df = pd.DataFrame(win_data)
        
        # Pivot for grouped bar chart
        win_pivot = win_df.pivot(index=group_by, columns='Model', values='Win Rate (%)')
        win_pivot.plot(kind='bar', ax=ax2, rot=0)
        ax2.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_title(f'Win Rate vs {baseline_model}')
        ax2.set_ylabel('Win Rate (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    else:
        # Plot 1: Overall difference distribution
        sns.violinplot(
            data=diff_df,
            x='Model',
            y='Difference',
            ax=ax1,
            palette='Set2'
        )
        ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_title(f'Performance vs {baseline_model}')
        ax1.set_ylabel(f'Δ {metric.replace("_", " ").title()}')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win rates
        win_rates = []
        for model in diff_df['Model'].unique():
            model_data = diff_df[diff_df['Model'] == model]
            wins = (model_data['Difference'] > 0).sum()
            win_rate = 100 * wins / len(model_data)
            win_rates.append({'Model': model, 'Win Rate (%)': win_rate})
        
        win_df = pd.DataFrame(win_rates).sort_values('Win Rate (%)', ascending=False)
        
        sns.barplot(
            data=win_df,
            x='Model',
            y='Win Rate (%)',
            ax=ax2,
            palette='Set2'
        )
        ax2.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_title(f'Win Rate vs {baseline_model}')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        group_suffix = f'_{group_by}' if group_by else ''
        filename = f'comparison_{metric}{group_suffix}.png'
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / filename}")
    
    return fig


def create_performance_heatmap(
    df: pd.DataFrame,
    metric: str = 'test_f1',
    group_by: List[str] = None,
    output_dir: Path = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create heatmap showing performance across different configurations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with results.
    metric : str, default='test_f1'
        Metric to visualize.
    group_by : list of str, optional
        Two columns to group by for heatmap axes.
    output_dir : Path, optional
        Directory to save plots.
    figsize : tuple, default=(12, 8)
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    model_cols = [col for col in df.columns if metric in col and col.startswith(('Choquet', 'Logistic'))]
    
    if not model_cols:
        print(f"No metric columns found for {metric}")
        return None
    
    # Determine number of subplots
    n_models = len(model_cols)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, col in enumerate(model_cols):
        model_name = col.replace(f'_{metric}', '')
        
        if group_by and len(group_by) == 2 and all(g in df.columns for g in group_by):
            # Create pivot table for heatmap
            pivot = df.pivot_table(
                values=col,
                index=group_by[0],
                columns=group_by[1],
                aggfunc='mean'
            )
            
            # Plot heatmap
            sns.heatmap(
                pivot,
                ax=axes[idx],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': metric.replace('_', ' ').title()}
            )
            axes[idx].set_title(model_name)
        else:
            # Simple bar chart if grouping not possible
            mean_values = df[col].mean()
            axes[idx].bar([model_name], [mean_values])
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(model_name)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'heatmap_{metric}.png'
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / filename}")
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations for Choquet results")
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to results CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='test_f1',
        help='Metric to visualize'
    )
    parser.add_argument(
        '--group_by',
        type=str,
        default=None,
        help='Column to group by'
    )
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    
    print(f"Creating visualizations for {args.metric}...")
    
    # Create boxplots
    create_boxplots(
        df,
        metric=args.metric,
        group_by=args.group_by,
        output_dir=output_dir
    )
    
    # Create comparison plots
    create_comparison_plots(
        df,
        metric=args.metric,
        group_by=args.group_by,
        output_dir=output_dir
    )
    
    # Create heatmap if grouping variables available
    if args.group_by:
        group_cols = args.group_by.split(',')
        if len(group_cols) == 2:
            create_performance_heatmap(
                df,
                metric=args.metric,
                group_by=group_cols,
                output_dir=output_dir
            )
    
    print(f"Plots saved to: {output_dir}")
    plt.show()
