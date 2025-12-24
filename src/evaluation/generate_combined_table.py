#!/usr/bin/env python3
"""
Script to extract F1 scores from individual tables and generate combined LaTeX table.

This script reads the table_max_samples_2500.tex files from ROC_PLA, FOR_PLA, and ABL_ACC
directories and generates the combined table with the target format.
"""

import re
from pathlib import Path
import sys

def extract_f1_scores(tex_file_path):
    """
    Extract F1 scores from a LaTeX table file.

    Parameters
    ----------
    tex_file_path : Path
        Path to the .tex file

    Returns
    -------
    dict
        Dictionary mapping method names to (mean, std) tuples
    """
    with open(tex_file_path, 'r') as f:
        content = f.read()

    # Split by lines and find the table rows
    lines = content.split('\n')
    scores = {}

    for line in lines:
        line = line.strip()
        # Look for lines with & and \pm
        if '&' in line and '\\pm' in line and '\\\\' in line:
            # Remove all \textbf{...} from the entire line first
            line = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)

            # Extract method name and scores
            parts = line.split('&')
            if len(parts) >= 2:
                method_part = parts[0].strip()
                score_part = parts[1].strip()

                # Clean method name
                method_name = method_part.strip()
                if '\\pm' in score_part:
                    score_parts = score_part.split('\\pm')
                    if len(score_parts) >= 2:
                        mean_str = score_parts[0].strip().replace('$', '')
                        std_str = score_parts[1].replace('\\\\', '').replace('$', '').strip()

                        try:
                            mean = float(mean_str)
                            std = float(std_str)
                            scores[method_name] = (mean, std)
                        except ValueError as e:
                            print(f"Warning: Error parsing {method_name}: {e}")

    return scores

def format_score(mean, std):
    """Format score as mean ± std with appropriate precision."""
    # Convert to percentages and format
    mean_pct = mean * 100
    std_pct = std * 100

    # Use 1 decimal place and return formatted string
    return f"{mean_pct:.1f} ± {std_pct:.1f}"

def main():
    """Main function to generate the combined table."""

    # Base directory for figures
    figures_dir = Path("src/figures")

    # Map directories to column names
    dir_to_col = {
        "ABL_ACC": "AA",
        "FOR_PLA": "FL",
        "ROC_PLA": "RL"
    }

    # Extract scores from all tables
    all_scores = {}
    for dirname, col_name in dir_to_col.items():
        tex_file = figures_dir / dirname / "table_max_samples_2500.tex"
        if tex_file.exists():
            scores = extract_f1_scores(tex_file)
            all_scores[col_name] = scores
            print(f"Extracted {len(scores)} scores from {dirname}")
        else:
            print(f"Warning: {tex_file} not found")
            all_scores[col_name] = {}

    # Method mappings
    method_mapping = {
        'LogisticRegression': 'LR',
        'Choquet Weight': 'ChqW',
        'Choquet Power': 'ChqP',
        'Choquet Weight TN3': 'ChqW_T3',
        'Choquet Weight TN6': 'ChqW_T6',
        'Choquet Power TN3': 'ChqP_T3',
        'Choquet Power TN6': 'ChqP_T6'
    }

    # Collect all scores in the right format
    choquet_scores = {}
    tnorm_scores = {}

    for col in ['AA', 'FL', 'RL']:
        if col not in all_scores:
            continue

        scores = all_scores[col]

        # Choquet integral methods
        for method, short_name in [('LogisticRegression', 'LR'),
                                   ('Choquet Weight', 'ChqW'),
                                   ('Choquet Power', 'ChqP')]:
            if method in scores:
                mean, std = scores[method]
                if short_name not in choquet_scores:
                    choquet_scores[short_name] = {}
                choquet_scores[short_name][col] = (mean, std)

        # T-norm methods
        for method, short_name in [('Choquet Weight TN3', 'ChqW_T3'),
                                   ('Choquet Weight TN6', 'ChqW_T6'),
                                   ('Choquet Power TN3', 'ChqP_T3'),
                                   ('Choquet Power TN6', 'ChqP_T6')]:
            if method in scores:
                mean, std = scores[method]
                if short_name not in tnorm_scores:
                    tnorm_scores[short_name] = {}
                tnorm_scores[short_name][col] = (mean, std)

    # Format scores for choquet table
    choquet_values = []
    for method in ['LR', 'ChqW', 'ChqP']:
        for col in ['AA', 'FL', 'RL']:
            if method in choquet_scores and col in choquet_scores[method]:
                mean, std = choquet_scores[method][col]
                choquet_values.append(format_score(mean, std))
            else:
                choquet_values.append("N/A")

    # Format scores for tnorm table
    tnorm_values = []
    for method in ['ChqW_T3', 'ChqW_T6', 'ChqP_T3', 'ChqP_T6']:
        for col in ['AA', 'FL', 'RL']:
            if method in tnorm_scores and col in tnorm_scores[method]:
                mean, std = tnorm_scores[method][col]
                tnorm_values.append(format_score(mean, std))
            else:
                tnorm_values.append("N/A")

    # Combine all values
    all_values = choquet_values + tnorm_values

    # Build the table by replacing placeholders manually
    template = r"""\begin{table}[t!]
\centering
\caption{Weighted F1-score for the fuzzy aggregation methods. Left: Choquet integral with weighted and power fuzzy measures, and logistic regression. Right: Pre-aggregation with weighted and power fuzzy measures for the two T-norms.}
\label{tab: score_combined}
\begin{minipage}[t]{0.48\textwidth}
\centering
\subcaption{Choquet integral}
\label{tab: score_choquet}
\resizebox{\textwidth}{!}{%%
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}cccc@{}}
\toprule
\textbf{classification} & \textbf{AA}                       & \textbf{FL}                             & \textbf{RL}                       \\
\midrule
\rowcolor{gray!25}
LR            & $SCORE0$          & $SCORE1$          & $SCORE2$          \\
ChqW          & $SCORE3$          & $SCORE4$          & $SCORE5$          \\
\rowcolor{gray!25}
ChqP          & $\mathbf{SCORE6}$ & $\mathbf{SCORE7}$ & $\mathbf{SCORE8}$ \\
\bottomrule
\end{tabular}%%
}}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\subcaption{Pre-aggregation with T-norms}
\label{tab: score_tnorm}
\resizebox{\textwidth}{!}{%%
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{@{}ccccc@{}}
\toprule
\textbf{classification} &       & \textbf{AA}                       & \textbf{FL}                             & \textbf{RL}                       \\
\midrule
\rowcolor{gray!25}
{ChqW}        & $T^3$ & $\mathbf{SCORE9}$ & $\mathbf{SCORE10}$ & $\mathbf{SCORE11}$ \\
              & $T^6$ & $SCORE12$          & $SCORE13$          & $SCORE14$          \\
\rowcolor{gray!25}
{ChqP}        & $T^3$ & $SCORE15$          & $SCORE16$          & $SCORE17$          \\
              & $T^6$ & $SCORE18$          & $SCORE19$          & $SCORE20$          \\
\bottomrule
\end{tabular}%%
}}
\end{minipage}
\vspace{-1em}
\end{table}"""

    # Replace placeholders with actual values (reverse order to avoid SCORE1 replacing part of SCORE10)
    result = template
    for i in range(len(all_values) - 1, -1, -1):
        result = result.replace(f"SCORE{i}", all_values[i])

    # Save to file
    output_file = figures_dir / "combined_table.tex"
    with open(output_file, 'w') as f:
        f.write(result)

    print(f"✓ Combined table successfully generated: {output_file}")
    print(f"  - Choquet methods: {len(choquet_scores)} (LR, ChqW, ChqP)")
    print(f"  - T-norm methods: {len(tnorm_scores)} (ChqW_T3/T6, ChqP_T3/T6)")

if __name__ == "__main__":
    main()