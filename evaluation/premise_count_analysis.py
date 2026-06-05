import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pathlib import Path


def group_cases_by_premise_count(cases, predictions_dict):
    # Define bins
    bins = [
        ('0', 0, 0),
        ('1', 1, 1),
        ('2', 2, 2),
        ('3-5', 3, 5),
        ('6-10', 6, 10),
        ('10+', 11, 999)
    ]

    results = {
        'bins': [b[0] for b in bins],
        'bin_ranges': [(b[1], b[2]) for b in bins],
        'counts': [],
        'approaches': {}
    }

    # Initialize approach results
    for approach_name in predictions_dict.keys():
        results['approaches'][approach_name] = {
            'macro_f1': [],
            'micro_f1': []
        }

    # Group cases and calculate metrics
    for bin_label, min_count, max_count in bins:
        # Find cases in this bin
        bin_indices = []
        for i, case in enumerate(cases):
            n_premises = len(case.get('premises', []))
            if min_count <= n_premises <= max_count:
                bin_indices.append(i)

        results['counts'].append(len(bin_indices))

        if len(bin_indices) == 0:
            # No cases in this bin
            for approach_name in predictions_dict.keys():
                results['approaches'][approach_name]['macro_f1'].append(0.0)
                results['approaches'][approach_name]['micro_f1'].append(0.0)
            continue

        # Get labels for this bin
        y_true_bin = np.array([cases[i]['labels_binary'] for i in bin_indices])

        # Calculate F1 for each approach
        for approach_name, y_pred_all in predictions_dict.items():
            y_pred_bin = y_pred_all[bin_indices]

            # Add "No Violation" column (LexGLUE protocol)
            y_true_ext = np.hstack([y_true_bin, (y_true_bin.sum(axis=1) == 0).reshape(-1, 1)])
            y_pred_ext = np.hstack([y_pred_bin, (y_pred_bin.sum(axis=1) == 0).reshape(-1, 1)])

            macro_f1 = f1_score(y_true_ext, y_pred_ext, average='macro', zero_division=0)
            micro_f1 = f1_score(y_true_ext, y_pred_ext, average='micro', zero_division=0)

            results['approaches'][approach_name]['macro_f1'].append(macro_f1)
            results['approaches'][approach_name]['micro_f1'].append(micro_f1)

    return results


def plot_premise_count_analysis(results, title="Performance vs Premise Count",
                                output_path=None, colors=None):
    if colors is None:
        colors = {
            'Baseline': '#1f77b4',
            'Premise': '#ff7f0e',
            'Hybrid': '#2ca02c',
            'Full-text': '#1f77b4',
            'Premises': '#ff7f0e'
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_pos = np.arange(len(results['bins']))

    # Plot Macro F1
    for approach_name, metrics in results['approaches'].items():
        color = colors.get(approach_name, None)
        ax1.plot(x_pos, metrics['macro_f1'], marker='o', label=approach_name,
                linewidth=2, markersize=6, color=color)

    ax1.set_xlabel('Number of Premises Extracted', fontsize=11)
    ax1.set_ylabel('Macro F1', fontsize=11)
    ax1.set_title('Macro F1 vs Premise Count', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results['bins'])
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Add case counts as text
    for i, count in enumerate(results['counts']):
        ax1.text(i, -0.08, f'n={count}', ha='center', va='top',
                fontsize=8, color='gray', transform=ax1.get_xaxis_transform())

    # Plot Micro F1
    for approach_name, metrics in results['approaches'].items():
        color = colors.get(approach_name, None)
        ax2.plot(x_pos, metrics['micro_f1'], marker='o', label=approach_name,
                linewidth=2, markersize=6, color=color)

    ax2.set_xlabel('Number of Premises Extracted', fontsize=11)
    ax2.set_ylabel('Micro F1', fontsize=11)
    ax2.set_title('Micro F1 vs Premise Count', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results['bins'])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Add case counts
    for i, count in enumerate(results['counts']):
        ax2.text(i, -0.08, f'n={count}', ha='center', va='top',
                fontsize=8, color='gray', transform=ax2.get_xaxis_transform())

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()


def print_premise_count_table(results):
    """Print a table of F1 scores by premise count."""
    approach_names = list(results['approaches'].keys())

    # Header
    header = f"{'Premises':<10} {'N Cases':<10}"
    for name in approach_names:
        header += f" {name+' (macro)':<18} {name+' (micro)':<18}"
    print("="*len(header))
    print(header)
    print("="*len(header))

    # Rows
    for i, bin_label in enumerate(results['bins']):
        count = results['counts'][i]
        row = f"{bin_label:<10} {count:<10}"

        for name in approach_names:
            macro = results['approaches'][name]['macro_f1'][i]
            micro = results['approaches'][name]['micro_f1'][i]
            row += f" {macro:>18.4f} {micro:>18.4f}"

        print(row)

    print("="*len(header))
