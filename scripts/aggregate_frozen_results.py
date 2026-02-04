#!/usr/bin/env python
"""
Aggregate frozen CLIP benchmark results across seeds.

Computes:
- Mean and std for R@1, R@5, R@10, MdR, MnR
- Bootstrap confidence intervals
- Holm-Bonferroni correction for pairwise comparisons
- Publication-ready tables (LaTeX and markdown)

Usage:
    python scripts/aggregate_frozen_results.py \
        --results_dir results \
        --output_dir reports \
        --methods meanP seqTransf tightTransf \
        --seeds 0 1 2

"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate frozen CLIP benchmark results')

    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing result subdirectories')
    parser.add_argument('--output_dir', type=str, default='reports',
                        help='Output directory for reports')
    parser.add_argument('--methods', nargs='+',
                        default=['meanP', 'seqTransf', 'tightTransf'],
                        help='Methods to include')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Seeds to include')
    parser.add_argument('--settings', nargs='+', default=['hard', 'origin'],
                        help='Evaluation settings (hard, origin)')
    parser.add_argument('--bootstrap_n', type=int, default=1000,
                        help='Number of bootstrap samples for CI')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level')

    return parser.parse_args()


def load_metrics(results_dir, method, seed, setting):
    """Load metrics from a single experiment run."""
    dir_name = f"frozen_{method}_seed{seed}_{setting}"
    result_path = Path(results_dir) / dir_name

    # Try to find metrics file
    metrics_files = list(result_path.glob('metrics_epoch*.json'))

    if not metrics_files:
        # Try to parse from log file
        log_path = result_path / 'log.txt'
        if log_path.exists():
            return parse_log_file(log_path)
        return None

    # Get the latest metrics file
    latest_metrics = sorted(metrics_files)[-1]

    with open(latest_metrics) as f:
        data = json.load(f)

    return data.get('text_to_video', data)


def parse_log_file(log_path):
    """Parse metrics from log file if JSON not available."""
    metrics = {}

    with open(log_path) as f:
        content = f.read()

    # Parse Text-to-Video metrics
    import re

    # Look for pattern like: R@1: 45.2 - R@5: 72.3 - R@10: 82.1 - Median R: 2.0 - Mean R: 8.5
    pattern = r'R@1:\s*([\d.]+).*R@5:\s*([\d.]+).*R@10:\s*([\d.]+).*Median R:\s*([\d.]+).*Mean R:\s*([\d.]+)'

    matches = re.findall(pattern, content)
    if matches:
        # Take the last match (final evaluation)
        r1, r5, r10, mr, meanr = matches[-1]
        metrics = {
            'R1': float(r1),
            'R5': float(r5),
            'R10': float(r10),
            'MR': float(mr),
            'MeanR': float(meanr)
        }

    return metrics if metrics else None


def compute_statistics(values):
    """Compute mean and std for a list of values."""
    values = np.array(values)
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1) if len(values) > 1 else 0,
        'min': np.min(values),
        'max': np.max(values),
        'n': len(values)
    }


def bootstrap_ci(values, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)

    if n < 2:
        return values[0], values[0]

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper


def paired_bootstrap_test(values1, values2, n_bootstrap=1000):
    """Paired bootstrap test for difference in means."""
    values1 = np.array(values1)
    values2 = np.array(values2)

    observed_diff = np.mean(values1) - np.mean(values2)

    # Bootstrap under null hypothesis (no difference)
    combined = np.concatenate([values1, values2])
    n = len(values1)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample
        sample = np.random.choice(combined, size=2*n, replace=True)
        diff = np.mean(sample[:n]) - np.mean(sample[n:])
        bootstrap_diffs.append(diff)

    # Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value


def holm_bonferroni_correction(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction to multiple p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]

    corrected = np.zeros(n, dtype=bool)
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_pvals)):
        threshold = alpha / (n - i)
        if p <= threshold:
            corrected[idx] = True
        else:
            break

    return corrected


def generate_latex_table(results, setting, metrics=['R1', 'R5', 'R10', 'MR', 'MeanR']):
    """Generate LaTeX table for results."""
    lines = []

    # Table header
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Results on RTime-' + setting.capitalize() + r' (mean $\pm$ std across 3 seeds)}')
    lines.append(r'\begin{tabular}{l' + 'c' * len(metrics) + r'}')
    lines.append(r'\toprule')

    # Metric headers
    header = 'Method & ' + ' & '.join([f'R@{m[1:]}' if m.startswith('R') else m for m in metrics])
    lines.append(header + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for method in results:
        row = [method]
        for metric in metrics:
            if metric in results[method]:
                mean = results[method][metric]['mean']
                std = results[method][metric]['std']
                row.append(f'{mean:.1f} $\\pm$ {std:.1f}')
            else:
                row.append('-')
        lines.append(' & '.join(row) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_markdown_table(results, setting, metrics=['R1', 'R5', 'R10', 'MR', 'MeanR']):
    """Generate Markdown table for results."""
    lines = []

    # Title
    lines.append(f'## Results on RTime-{setting.capitalize()}')
    lines.append('')

    # Table header
    header = '| Method | ' + ' | '.join([f'R@{m[1:]}' if m.startswith('R') else m for m in metrics]) + ' |'
    lines.append(header)

    separator = '|' + '|'.join(['---'] * (len(metrics) + 1)) + '|'
    lines.append(separator)

    # Data rows
    for method in results:
        row = [method]
        for metric in metrics:
            if metric in results[method]:
                mean = results[method][metric]['mean']
                std = results[method][metric]['std']
                row.append(f'{mean:.1f} +/- {std:.1f}')
            else:
                row.append('-')
        lines.append('| ' + ' | '.join(row) + ' |')

    lines.append('')

    return '\n'.join(lines)


def main():
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all results
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print("Loading results...")
    for method in args.methods:
        for seed in args.seeds:
            for setting in args.settings:
                metrics = load_metrics(args.results_dir, method, seed, setting)
                if metrics:
                    for metric_name, value in metrics.items():
                        all_results[setting][method][metric_name].append(value)
                    print(f"  Loaded: {method}/seed{seed}/{setting}")
                else:
                    print(f"  Missing: {method}/seed{seed}/{setting}")

    # Compute statistics
    print("\nComputing statistics...")
    aggregated = defaultdict(dict)

    for setting in args.settings:
        print(f"\n=== RTime-{setting.capitalize()} ===")

        for method in args.methods:
            if method not in all_results[setting]:
                print(f"  {method}: No results found")
                continue

            aggregated[setting][method] = {}
            print(f"\n  {method}:")

            for metric in ['R1', 'R5', 'R10', 'MR', 'MeanR']:
                if metric not in all_results[setting][method]:
                    continue

                values = all_results[setting][method][metric]
                stats_result = compute_statistics(values)
                ci_low, ci_high = bootstrap_ci(values, args.bootstrap_n, args.alpha)

                aggregated[setting][method][metric] = {
                    'mean': stats_result['mean'],
                    'std': stats_result['std'],
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'values': values
                }

                print(f"    {metric}: {stats_result['mean']:.2f} +/- {stats_result['std']:.2f} "
                      f"(95% CI: [{ci_low:.2f}, {ci_high:.2f}])")

    # Pairwise comparisons with Holm-Bonferroni correction
    print("\n=== Pairwise Comparisons (R@1) ===")

    for setting in args.settings:
        print(f"\n{setting.capitalize()}:")

        methods_with_data = [m for m in args.methods if m in aggregated[setting]]
        comparisons = []
        comparison_labels = []

        for i, m1 in enumerate(methods_with_data):
            for m2 in methods_with_data[i+1:]:
                if 'R1' in aggregated[setting][m1] and 'R1' in aggregated[setting][m2]:
                    v1 = aggregated[setting][m1]['R1']['values']
                    v2 = aggregated[setting][m2]['R1']['values']

                    diff, p_val = paired_bootstrap_test(v1, v2, args.bootstrap_n)
                    comparisons.append((m1, m2, diff, p_val))
                    comparison_labels.append(f"{m1} vs {m2}")

        if comparisons:
            p_values = [c[3] for c in comparisons]
            significant = holm_bonferroni_correction(p_values, args.alpha)

            for i, (m1, m2, diff, p_val) in enumerate(comparisons):
                sig_marker = '*' if significant[i] else ''
                print(f"  {m1} vs {m2}: diff = {diff:+.2f}, p = {p_val:.4f}{sig_marker}")

    # Generate tables
    print("\nGenerating tables...")

    for setting in args.settings:
        if setting not in aggregated:
            continue

        # LaTeX table
        latex_table = generate_latex_table(aggregated[setting], setting)
        latex_path = Path(args.output_dir) / f'table_{setting}.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"  Saved: {latex_path}")

        # Markdown table
        md_table = generate_markdown_table(aggregated[setting], setting)
        md_path = Path(args.output_dir) / f'table_{setting}.md'
        with open(md_path, 'w') as f:
            f.write(md_table)
        print(f"  Saved: {md_path}")

    # Save raw aggregated results as JSON
    results_json = {}
    for setting in aggregated:
        results_json[setting] = {}
        for method in aggregated[setting]:
            results_json[setting][method] = {}
            for metric in aggregated[setting][method]:
                data = aggregated[setting][method][metric]
                results_json[setting][method][metric] = {
                    'mean': data['mean'],
                    'std': data['std'],
                    'ci_low': data['ci_low'],
                    'ci_high': data['ci_high'],
                    'values': data['values']
                }

    json_path = Path(args.output_dir) / 'aggregated_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # Print summary markdown
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for setting in args.settings:
        if setting in aggregated:
            print(generate_markdown_table(aggregated[setting], setting))


if __name__ == '__main__':
    main()
