#!/usr/bin/env python3
"""
aggregate_results.py - Aggregate and summarize RTime benchmark results

Usage:
    python scripts/aggregate_results.py [--results_dir RESULTS_DIR] [--output_dir OUTPUT_DIR]

This script:
    1. Parses evaluation logs for metrics (R@1, R@5, R@10, MdR, MnR)
    2. Computes mean +/- std across seeds for each method
    3. Generates results/summary.md with markdown tables
    4. Exports results/summary_detailed.csv for further analysis
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_metrics_from_log(log_path: str) -> Optional[Dict[str, float]]:
    """
    Parse evaluation metrics from a log file.

    Expected format (from CLIP4Clip eval_epoch):
        Text-to-Video:
            >>>  R@1: 45.2 - R@5: 72.1 - R@10: 82.3 - Median R: 2.0 - Mean R: 15.4
        Video-to-Text:
            >>>  V2T$R@1: 44.8 - V2T$R@5: 71.5 - ...

    Returns:
        Dictionary with metrics or None if parsing fails
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {log_path}: {e}")
        return None

    metrics = {}

    # Parse Text-to-Video metrics
    t2v_pattern = r'R@1:\s*([\d.]+)\s*-\s*R@5:\s*([\d.]+)\s*-\s*R@10:\s*([\d.]+)\s*-\s*Median R:\s*([\d.]+)\s*-\s*Mean R:\s*([\d.]+)'
    t2v_match = re.search(t2v_pattern, content)
    if t2v_match:
        metrics['T2V_R@1'] = float(t2v_match.group(1))
        metrics['T2V_R@5'] = float(t2v_match.group(2))
        metrics['T2V_R@10'] = float(t2v_match.group(3))
        metrics['T2V_MdR'] = float(t2v_match.group(4))
        metrics['T2V_MnR'] = float(t2v_match.group(5))

    # Parse Video-to-Text metrics
    v2t_pattern = r'V2T\$R@1:\s*([\d.]+)\s*-\s*V2T\$R@5:\s*([\d.]+)\s*-\s*V2T\$R@10:\s*([\d.]+)\s*-\s*V2T\$Median R:\s*([\d.]+)\s*-\s*V2T\$Mean R:\s*([\d.]+)'
    v2t_match = re.search(v2t_pattern, content)
    if v2t_match:
        metrics['V2T_R@1'] = float(v2t_match.group(1))
        metrics['V2T_R@5'] = float(v2t_match.group(2))
        metrics['V2T_R@10'] = float(v2t_match.group(3))
        metrics['V2T_MdR'] = float(v2t_match.group(4))
        metrics['V2T_MnR'] = float(v2t_match.group(5))

    if not metrics:
        print(f"  Warning: Could not parse metrics from {log_path}")
        return None

    return metrics


def compute_statistics(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    return mean, std


def format_metric(mean: float, std: float, show_std: bool = True) -> str:
    """Format metric as mean +/- std."""
    if show_std and std > 0:
        return f"{mean:.1f} +/- {std:.1f}"
    return f"{mean:.1f}"


def collect_results(results_dir: Path) -> Dict:
    """
    Collect all results from evaluation log files.

    Returns:
        Nested dict: results[setting][method][seed] = metrics_dict
    """
    results = defaultdict(lambda: defaultdict(dict))

    # Find all evaluation result files
    # Expected filename pattern: eval_{method}_seed{seed}_{setting}.txt
    pattern = re.compile(r'eval_(.+)_seed(\d+)_(origin|hard)\.txt')

    for file_path in results_dir.glob('eval_*.txt'):
        match = pattern.match(file_path.name)
        if not match:
            continue

        method = match.group(1)
        seed = int(match.group(2))
        setting = match.group(3)

        metrics = parse_metrics_from_log(file_path)
        if metrics:
            results[setting][method][seed] = metrics
            print(f"  Loaded: {file_path.name}")

    return results


def generate_summary_table(results: Dict, setting: str) -> List[str]:
    """Generate markdown table for a specific setting."""
    lines = []

    if setting not in results or not results[setting]:
        lines.append(f"*No results found for {setting} setting*")
        return lines

    methods = sorted(results[setting].keys())

    # Table header
    lines.append("| Method | R@1 | R@5 | R@10 | MdR | MnR | Seeds |")
    lines.append("|--------|-----|-----|------|-----|-----|-------|")

    for method in methods:
        seed_data = results[setting][method]
        seeds = sorted(seed_data.keys())

        # Collect metrics across seeds
        metrics_by_name = defaultdict(list)
        for seed in seeds:
            for metric_name, value in seed_data[seed].items():
                if metric_name.startswith('T2V_'):
                    metrics_by_name[metric_name].append(value)

        # Compute statistics
        r1_mean, r1_std = compute_statistics(metrics_by_name.get('T2V_R@1', []))
        r5_mean, r5_std = compute_statistics(metrics_by_name.get('T2V_R@5', []))
        r10_mean, r10_std = compute_statistics(metrics_by_name.get('T2V_R@10', []))
        mdr_mean, mdr_std = compute_statistics(metrics_by_name.get('T2V_MdR', []))
        mnr_mean, mnr_std = compute_statistics(metrics_by_name.get('T2V_MnR', []))

        # Format row
        show_std = len(seeds) > 1
        lines.append(
            f"| {method} | "
            f"{format_metric(r1_mean, r1_std, show_std)} | "
            f"{format_metric(r5_mean, r5_std, show_std)} | "
            f"{format_metric(r10_mean, r10_std, show_std)} | "
            f"{format_metric(mdr_mean, mdr_std, show_std)} | "
            f"{format_metric(mnr_mean, mnr_std, show_std)} | "
            f"{len(seeds)} |"
        )

    return lines


def generate_summary_md(results: Dict, output_path: Path):
    """Generate summary.md with all results."""
    lines = [
        "# RTime Benchmark Results",
        "",
        "Results from CLIP4Clip training on the RTime dataset.",
        "",
        "## Text-to-Video Retrieval",
        "",
    ]

    # Origin setting results
    lines.append("### RTime-Origin (Original videos only)")
    lines.append("")
    lines.extend(generate_summary_table(results, 'origin'))
    lines.append("")

    # Hard setting results
    lines.append("### RTime-Hard (Original + Reversed videos)")
    lines.append("")
    lines.extend(generate_summary_table(results, 'hard'))
    lines.append("")

    # Methodology notes
    lines.extend([
        "## Methodology",
        "",
        "- **Methods**: meanP (mean pooling), seqTransf (sequential transformer), tightTransf (tight transformer)",
        "- **Metrics**: R@K (Recall at K), MdR (Median Rank), MnR (Mean Rank)",
        "- **Seeds**: Results averaged across multiple random seeds (mean +/- std shown)",
        "- **Settings**:",
        "  - Origin: Text-to-video retrieval with original videos only",
        "  - Hard: Text-to-video retrieval with both original and reversed videos",
        "",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated: {output_path}")


def generate_detailed_csv(results: Dict, output_path: Path):
    """Generate detailed CSV with all individual results."""
    rows = []
    header = [
        'setting', 'method', 'seed',
        'T2V_R@1', 'T2V_R@5', 'T2V_R@10', 'T2V_MdR', 'T2V_MnR',
        'V2T_R@1', 'V2T_R@5', 'V2T_R@10', 'V2T_MdR', 'V2T_MnR'
    ]

    for setting in sorted(results.keys()):
        for method in sorted(results[setting].keys()):
            for seed in sorted(results[setting][method].keys()):
                metrics = results[setting][method][seed]
                row = [setting, method, str(seed)]
                for metric_name in header[3:]:
                    row.append(str(metrics.get(metric_name, '')))
                rows.append(row)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')

    print(f"Generated: {output_path}")


def generate_aggregated_csv(results: Dict, output_path: Path):
    """Generate aggregated CSV with mean/std across seeds."""
    header = [
        'setting', 'method', 'num_seeds',
        'T2V_R@1_mean', 'T2V_R@1_std',
        'T2V_R@5_mean', 'T2V_R@5_std',
        'T2V_R@10_mean', 'T2V_R@10_std',
        'T2V_MdR_mean', 'T2V_MdR_std',
        'T2V_MnR_mean', 'T2V_MnR_std',
    ]

    rows = []
    for setting in sorted(results.keys()):
        for method in sorted(results[setting].keys()):
            seed_data = results[setting][method]
            seeds = list(seed_data.keys())

            # Collect metrics
            metrics_by_name = defaultdict(list)
            for seed in seeds:
                for metric_name, value in seed_data[seed].items():
                    if metric_name.startswith('T2V_'):
                        metrics_by_name[metric_name].append(value)

            row = [setting, method, str(len(seeds))]
            for metric_name in ['T2V_R@1', 'T2V_R@5', 'T2V_R@10', 'T2V_MdR', 'T2V_MnR']:
                mean, std = compute_statistics(metrics_by_name.get(metric_name, []))
                row.extend([f"{mean:.2f}", f"{std:.2f}"])

            rows.append(row)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate RTime benchmark results'
    )
    parser.add_argument(
        '--results_dir', type=str, default='results',
        help='Directory containing evaluation result files (default: results)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results',
        help='Output directory for summary files (default: results)'
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    results_dir = project_dir / args.results_dir
    output_dir = project_dir / args.output_dir

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting results...")
    results = collect_results(results_dir)

    if not results:
        print("ERROR: No evaluation results found.")
        print(f"Expected files in {results_dir}/ with pattern: eval_{{method}}_seed{{seed}}_{{setting}}.txt")
        print("\nRun 'bash scripts/evaluate.sh' first to generate evaluation results.")
        sys.exit(1)

    # Count total results
    total_results = sum(
        len(seeds)
        for setting in results.values()
        for seeds in setting.values()
    )
    print(f"\nFound {total_results} evaluation results")

    # Generate outputs
    print("\nGenerating summary files...")
    generate_summary_md(results, output_dir / 'summary.md')
    generate_detailed_csv(results, output_dir / 'summary_detailed.csv')
    generate_aggregated_csv(results, output_dir / 'summary_aggregated.csv')

    print("\nDone!")
    print(f"\nView results:")
    print(f"  cat {output_dir / 'summary.md'}")


if __name__ == '__main__':
    main()
