#!/usr/bin/env python3
"""
Sweep Analysis and Visualization

This script analyzes sweep experiment results (window_size and max_samples_per_class)
and generates:
1. Boxplots showing F1 score vs sweep parameter for all aggregation methods
2. LaTeX tables comparing methods at specific parameter values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Configure matplotlib
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 16,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# Color scheme
COLORS = {
    'Choquet_Power': '#ff7f0e',           # Orange
    'Choquet_Power_Tnorm': '#ff9e4a',     # Light orange
    'Choquet_Power_Tnorm2': '#ffb366',    # Lighter orange
    'Choquet_Weight': '#1f77b4',          # Blue
    'Choquet_Weight_Tnorm': '#5499c7',    # Light blue
    'Choquet_Weight_Tnorm2': '#85b8d9',   # Lighter blue
    'Choquet_Tnorm': '#9467bd',           # Purple
    'LogisticRegression': '#000000',      # Black
}


def load_results_and_config(results_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Load results CSV and configuration from a results directory.
    
    Parameters
    ----------
    results_dir : Path
        Path to results directory.
    
    Returns
    -------
    tuple
        (DataFrame of results, configuration dict)
    """
    csv_path = results_dir / 'choquet_aggregation_results.csv'
    config_path = results_dir / 'config.yaml'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    df = pd.read_csv(csv_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return df, config


def get_aggregation_columns(df: pd.DataFrame, metric: str = 'test_f1') -> Dict[str, str]:
    """
    Get all aggregation method columns from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    metric : str
        Metric to extract (default: 'test_f1').
    
    Returns
    -------
    dict
        Mapping from method name to column name.
    """
    cols = {}
    
    # Find all columns matching the metric
    for col in df.columns:
        if metric in col and (col.startswith('Choquet') or col.startswith('Logistic')):
            # Extract method name
            method_name = col.replace(f'_{metric}', '')
            cols[method_name] = col
    
    return cols


def create_sweep_boxplot(
    df: pd.DataFrame,
    sweep_param: str,
    metric: str = 'test_f1',
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = None
) -> plt.Figure:
    """
    Create boxplot showing metric vs sweep parameter for all methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with sweep parameter column.
    sweep_param : str
        Name of the sweep parameter column.
    metric : str
        Metric to plot (default: 'test_f1').
    output_path : Path, optional
        Path to save figure.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        Created figure.
    """
    # Get aggregation columns
    method_cols = get_aggregation_columns(df, metric)
    
    if not method_cols:
        raise ValueError(f"No method columns found for metric {metric}")
    
    # Check if sweep parameter exists
    if sweep_param not in df.columns:
        raise ValueError(f"Sweep parameter '{sweep_param}' not found in DataFrame")
    
    # Get unique parameter values
    param_values = sorted(df[sweep_param].dropna().unique())
    
    if len(param_values) == 0:
        raise ValueError(f"No valid values found for {sweep_param}")
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort methods for consistent ordering
    methods = sorted(method_cols.keys())
    
    # Calculate box positions
    n_methods = len(methods)
    box_width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        col = method_cols[method]
        color = COLORS.get(method, '#808080')  # Gray for unknown methods
        
        # Collect data for each parameter value
        box_data = []
        positions = []
        
        for j, param_val in enumerate(param_values):
            param_data = df[df[sweep_param] == param_val][col].dropna()
            if len(param_data) > 0:
                box_data.append(param_data.values)
                positions.append(j + (i - n_methods/2 + 0.5) * box_width)
        
        # Create boxplot for this method
        if box_data:
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color=color, linewidth=1.0),
                capprops=dict(color=color, linewidth=1.0),
                boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color, linewidth=1.0),
            )
    
    # Set labels and title
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels(param_values)
    ax.set_xlabel(sweep_param.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'F1 Score vs {sweep_param.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS.get(m, '#808080'), linewidth=8, 
                   label=m.replace('_', ' '))
        for m in methods
    ]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_latex_table(
    df: pd.DataFrame,
    param_name: str,
    param_value: float,
    metric: str = 'test_f1',
    output_path: Path = None,
    caption: str = None
) -> str:
    """
    Create LaTeX table comparing methods at a specific parameter value.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    param_name : str
        Name of the parameter column.
    param_value : float
        Value of the parameter to filter.
    metric : str
        Metric to report (default: 'test_f1').
    output_path : Path, optional
        Path to save LaTeX file.
    caption : str, optional
        Table caption.
    
    Returns
    -------
    str
        LaTeX table code.
    """
    # Filter data for specific parameter value
    filtered_df = df[df[param_name] == param_value]
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for {param_name}={param_value}")
    
    # Get aggregation columns
    method_cols = get_aggregation_columns(filtered_df, metric)
    
    # Calculate statistics
    results = []
    for method, col in sorted(method_cols.items()):
        values = filtered_df[col].dropna()
        if len(values) > 0:
            mean = values.mean()
            std = values.std()
            results.append({
                'Method': method.replace('_', ' '),
                'Mean': mean,
                'Std': std
            })
    
    # Create LaTeX table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"    \centering")
    if caption:
        latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{tab:{param_name}_{param_value}}}")
    latex.append(r"    \begin{tabular}{lc}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Method} & \textbf{F1 Score} \\")
    latex.append(r"        \midrule")
    
    # Find best result
    best_mean = max(r['Mean'] for r in results)
    
    for r in results:
        mean_str = f"{r['Mean']:.4f}"
        std_str = f"{r['Std']:.4f}"
        
        # Bold if best
        if abs(r['Mean'] - best_mean) < 1e-6:
            latex.append(f"        \\textbf{{{r['Method']}}} & \\textbf{{{mean_str} $\\pm$ {std_str}}} \\\\")
        else:
            latex.append(f"        {r['Method']} & {mean_str} $\\pm$ {std_str} \\\\")
    
    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")
    
    latex_code = '\n'.join(latex)
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_code)
        print(f"Saved LaTeX table: {output_path}")
    
    return latex_code


def analyze_results_directory(
    results_dir: Path,
    figures_base_dir: Path,
    window_size_value: Optional[int] = None,
    max_samples_value: Optional[int] = None
) -> None:
    """
    Analyze a single results directory and generate figures and tables.
    
    Parameters
    ----------
    results_dir : Path
        Path to results directory.
    figures_base_dir : Path
        Base directory for saving figures.
    window_size_value : int, optional
        Specific window_size for LaTeX table.
    max_samples_value : int, optional
        Specific max_samples_per_class for LaTeX table.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {results_dir.name}")
    print(f"{'='*80}")
    
    # Load results and config
    try:
        df, config = load_results_and_config(results_dir)
    except Exception as e:
        print(f"ERROR loading results: {e}")
        return
    
    # Extract class pair
    class_pair = config['dataset']['class_pair']
    class_name = '_'.join(class_pair)
    
    print(f"Class pair: {class_pair}")
    print(f"Results shape: {df.shape}")
    
    # Create output directory
    output_dir = figures_base_dir / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for sweep parameters
    has_window_sweep = 'window_size' in df.columns and df['window_size'].notna().any()
    has_samples_sweep = 'max_samples_per_class' in df.columns and df['max_samples_per_class'].notna().any()
    
    # Create boxplots for sweeps
    if has_window_sweep:
        print("\nCreating window_size sweep plot...")
        try:
            create_sweep_boxplot(
                df=df,
                sweep_param='window_size',
                metric='test_f1',
                output_path=output_dir / 'sweep_window_size.png',
                title=f'{class_pair[0]} vs {class_pair[1]} - Window Size Sweep'
            )
        except Exception as e:
            print(f"ERROR creating window_size plot: {e}")
    
    if has_samples_sweep:
        print("\nCreating max_samples_per_class sweep plot...")
        try:
            create_sweep_boxplot(
                df=df,
                sweep_param='max_samples_per_class',
                metric='test_f1',
                output_path=output_dir / 'sweep_max_samples.png',
                title=f'{class_pair[0]} vs {class_pair[1]} - Max Samples Sweep'
            )
        except Exception as e:
            print(f"ERROR creating max_samples plot: {e}")
    
    # Create LaTeX tables for specific values
    if window_size_value is not None and has_window_sweep:
        print(f"\nCreating LaTeX table for window_size={window_size_value}...")
        try:
            create_latex_table(
                df=df,
                param_name='window_size',
                param_value=window_size_value,
                metric='test_f1',
                output_path=output_dir / f'table_window_size_{window_size_value}.tex',
                caption=f'Comparison of aggregation methods for {class_pair[0]} vs {class_pair[1]} with window size = {window_size_value}'
            )
        except Exception as e:
            print(f"ERROR creating window_size table: {e}")
    
    if max_samples_value is not None and has_samples_sweep:
        print(f"\nCreating LaTeX table for max_samples_per_class={max_samples_value}...")
        try:
            create_latex_table(
                df=df,
                param_name='max_samples_per_class',
                param_value=max_samples_value,
                metric='test_f1',
                output_path=output_dir / f'table_max_samples_{max_samples_value}.tex',
                caption=f'Comparison of aggregation methods for {class_pair[0]} vs {class_pair[1]} with max samples per class = {max_samples_value}'
            )
        except Exception as e:
            print(f"ERROR creating max_samples table: {e}")
    
    print(f"\nResults saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze sweep experiment results and generate plots and LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all result directories
  python analyze_sweeps.py --results_dir src/results --figures_dir figures
  
  # Generate tables for specific parameter values
  python analyze_sweeps.py --results_dir src/results --figures_dir figures \\
      --window_size 7 --max_samples 46
  
  # Analyze specific result directory
  python analyze_sweeps.py --results_dir src/results/2dcf866bd48f6278 \\
      --figures_dir figures --window_size 7
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to results directory (or parent directory containing multiple result folders)'
    )
    
    parser.add_argument(
        '--figures_dir',
        type=str,
        default='src/figures',
        help='Output directory for figures (default: src/figures)'
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        default=None,
        help='Specific window_size value for LaTeX table generation'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Specific max_samples_per_class value for LaTeX table generation'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    figures_path = Path(args.figures_dir)
    
    if not results_path.exists():
        print(f"ERROR: Results directory not found: {results_path}")
        return
    
    # Check if this is a single results directory or parent directory
    if (results_path / 'choquet_aggregation_results.csv').exists():
        # Single results directory
        analyze_results_directory(
            results_dir=results_path,
            figures_base_dir=figures_path,
            window_size_value=args.window_size,
            max_samples_value=args.max_samples
        )
    else:
        # Parent directory - analyze all subdirectories with results
        result_dirs = [
            d for d in results_path.iterdir()
            if d.is_dir() and (d / 'choquet_aggregation_results.csv').exists()
        ]
        
        if not result_dirs:
            print(f"No result directories found in {results_path}")
            return
        
        print(f"Found {len(result_dirs)} result directories to analyze")
        
        for result_dir in sorted(result_dirs):
            analyze_results_directory(
                results_dir=result_dir,
                figures_base_dir=figures_path,
                window_size_value=args.window_size,
                max_samples_value=args.max_samples
            )
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Figures saved to: {figures_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
