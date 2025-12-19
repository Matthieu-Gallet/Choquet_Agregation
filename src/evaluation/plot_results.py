#!/usr/bin/env python3
"""
Visualization utilities for Choquet aggregation results.

This module provides functions to create boxplots and comparison plots
for analyzing Choquet aggregation performance across different configurations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure matplotlib for high-quality rendering (fallback without LaTeX)
try:
    # Try to use LaTeX if available
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Charter", "Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.figsize": (10, 6),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })
except:
    # Fallback to matplotlib's built-in fonts if LaTeX is not available
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.figsize": (10, 6),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def _create_matplotlib_boxplot(ax, data, x_col, y_col, palette=None, show_points=False):
    """
    Create a matplotlib boxplot equivalent to seaborn's boxplot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column name for x-axis (categorical)
    y_col : str
        Column name for y-axis (numerical)
    palette : str or list, optional
        Color palette (ignored for now, using default colors)
    show_points : bool, default=False
        Whether to overlay individual points
    """
    # Get unique categories
    categories = sorted(data[x_col].unique())
    
    # Prepare data for boxplot
    box_data = []
    positions = []
    
    for i, category in enumerate(categories):
        cat_data = data[data[x_col] == category][y_col].values
        if len(cat_data) > 0:
            box_data.append(cat_data)
            positions.append(i)
    
    if not box_data:
        return
    
    # Create boxplot
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
    
    # Style the boxplot
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
    
    for i, box in enumerate(bp['boxes']):
        color = colors[i % len(colors)]
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    
    # Overlay points if requested
    if show_points:
        for i, category in enumerate(categories):
            cat_data = data[data[x_col] == category]
            x_jitter = np.random.normal(i, 0.1, size=len(cat_data))
            ax.scatter(x_jitter, cat_data[y_col], alpha=0.3, color='black', s=10, zorder=3)
    
def _create_matplotlib_violinplot(ax, data, x_col, y_col, palette=None):
    """
    Create a matplotlib violinplot equivalent to seaborn's violinplot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column name for x-axis (categorical)
    y_col : str
        Column name for y-axis (numerical)
    palette : str or list, optional
        Color palette (ignored for now, using default colors)
    """
    # Get unique categories
    categories = sorted(data[x_col].unique())
    
    # Prepare data for violinplot
    violin_data = []
    positions = []
    
    for i, category in enumerate(categories):
        cat_data = data[data[x_col] == category][y_col].values
        if len(cat_data) > 0:
            violin_data.append(cat_data)
            positions.append(i)
    
    if not violin_data:
        return
    
    # Create violinplot
    vp = ax.violinplot(violin_data, positions=positions, showmeans=False, showmedians=True)
    
    # Style the violinplot
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
    
    for i, body in enumerate(vp['bodies']):
        color = colors[i % len(colors)]
        body.set_facecolor(color)
        body.set_alpha(0.7)
    
    # Style median lines
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(2)
    
    # Set x-ticks
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)


def _create_matplotlib_barplot(ax, data, x_col, y_col, palette=None):
    """
    Create a matplotlib barplot equivalent to seaborn's barplot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column name for x-axis (categorical)
    y_col : str
        Column name for y-axis (numerical)
    palette : str or list, optional
        Color palette (ignored for now, using default colors)
    """
    # Get unique categories
    categories = data[x_col].unique()
    
    # Prepare data
    values = []
    positions = []
    
    for i, category in enumerate(categories):
        cat_data = data[data[x_col] == category][y_col].values
        if len(cat_data) > 0:
            values.append(np.mean(cat_data))  # Use mean like seaborn barplot
            positions.append(i)
    
    if not values:
        return
    
    # Create barplot
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
    bars = ax.bar(positions, values, color=[colors[i % len(colors)] for i in range(len(values))], alpha=0.7)
    
    # Set x-ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)


def _create_matplotlib_heatmap(ax, data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws=None):
    """
    Create a matplotlib heatmap equivalent to seaborn's heatmap.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    data : pd.DataFrame
        Data to plot (pivot table)
    annot : bool, default=True
        Whether to annotate cells with values
    fmt : str, default='.3f'
        String format for annotations
    cmap : str, default='YlOrRd'
        Colormap name
    cbar_kws : dict, optional
        Colorbar keyword arguments
    """
    # Create heatmap using imshow
    im = ax.imshow(data.values, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if cbar_kws and 'label' in cbar_kws:
        cbar.set_label(cbar_kws['label'])
    
    # Set ticks and labels
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add annotations
    if annot:
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                value = data.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:{fmt}}',
                                 ha='center', va='center', color='black',
                                 fontsize=8, weight='bold')
    
    return im


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
            
            # Create boxplot using matplotlib
            _create_matplotlib_boxplot(
                ax=ax,
                data=group_data,
                x_col='Model',
                y_col='Value',
                show_points=show_points
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
        
        # Create boxplot using matplotlib
        _create_matplotlib_boxplot(
            ax=ax,
            data=plot_df,
            x_col='Model',
            y_col='Value',
            show_points=show_points
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
        # For now, create separate boxplots for each model within each group
        models = sorted(diff_df['Model'].unique())
        groups = sorted(diff_df[group_by].unique())
        
        # Calculate positions for grouped boxplots
        n_models = len(models)
        n_groups = len(groups)
        width = 0.8 / n_models  # Width per model
        
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
        
        for i, model in enumerate(models):
            model_data = diff_df[diff_df['Model'] == model]
            positions = []
            box_data = []
            
            for j, group in enumerate(groups):
                group_model_data = model_data[model_data[group_by] == group]['Difference'].values
                if len(group_model_data) > 0:
                    positions.append(j - 0.4 + (i + 0.5) * width)
                    box_data.append(group_model_data)
            
            if box_data:
                bp = ax1.boxplot(box_data, positions=positions, widths=width*0.8, patch_artist=True)
                color = colors[i % len(colors)]
                for box in bp['boxes']:
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
        
        ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_title(f'Performance vs {baseline_model}')
        ax1.set_ylabel(f'Δ {metric.replace("_", " ").title()}')
        ax1.set_xticks(range(n_groups))
        ax1.set_xticklabels(groups)
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i % len(colors)], alpha=0.7) 
                          for i in range(n_models)]
        ax1.legend(legend_elements, models, bbox_to_anchor=(1.05, 1), loc='upper left')
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
        _create_matplotlib_violinplot(
            ax=ax1,
            data=diff_df,
            x_col='Model',
            y_col='Difference'
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
        
        _create_matplotlib_barplot(
            ax=ax2,
            data=win_df,
            x_col='Model',
            y_col='Win Rate (%)'
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
            _create_matplotlib_heatmap(
                ax=axes[idx],
                data=pivot,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
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
