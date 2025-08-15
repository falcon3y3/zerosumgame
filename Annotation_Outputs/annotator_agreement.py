# Full script: Annotator agreement with green visualization, side-by-side matrices
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# Bootstrap confidence interval
# -----------------------------
def bootstrap_ci(stat_func, labels_a, labels_b, n_boot=1000, ci=95, random_state=None):
    rng = np.random.default_rng(random_state)
    stats = []
    n = len(labels_a)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, n)
        stat = stat_func(labels_a.iloc[sample_idx], labels_b.iloc[sample_idx])
        stats.append(stat)
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return lower, upper

# -----------------------------
# Clean labels to Y/N
# -----------------------------
def clean_labels(series):
    series = series.astype(str).str.strip().str.upper()
    valid_labels = {'Y', 'N'}
    if not series.isin(valid_labels).all():
        bad_values = series[~series.isin(valid_labels)].unique()
        raise ValueError(f"Invalid labels found: {bad_values}. Only 'Y' or 'N' allowed.")
    return series

# -----------------------------
# Compute binary agreement metrics
# -----------------------------
def analyze_binary_annotations(df, col_a='label_a', col_b='label_b', n_boot=1000):
    df = df.dropna(subset=[col_a, col_b])
    
    # Clean labels and convert Y/N to 1/0
    labels_a_str = clean_labels(df[col_a])
    labels_b_str = clean_labels(df[col_b])
    map_labels = {'Y': 1, 'N': 0}
    labels_a = labels_a_str.map(map_labels)
    labels_b = labels_b_str.map(map_labels)

    # Define stat functions
    def raw_agree(a, b): return (a == b).mean()
    def kappa_func(a, b): return cohen_kappa_score(a, b)
    def bias_func(a, b): return a.mean() - b.mean()

    # Point estimates
    raw_agreement = raw_agree(labels_a, labels_b)
    kappa = kappa_func(labels_a, labels_b)
    bias_index = bias_func(labels_a, labels_b)

    # Bootstrap CIs
    raw_ci = bootstrap_ci(raw_agree, labels_a, labels_b, n_boot)
    kappa_ci = bootstrap_ci(kappa_func, labels_a, labels_b, n_boot)
    bias_ci = bootstrap_ci(bias_func, labels_a, labels_b, n_boot)

    # Confusion matrices
    cm_counts = confusion_matrix(labels_a_str, labels_b_str, labels=['Y', 'N'])
    cm_counts_df = pd.DataFrame(cm_counts, index=['A:Y', 'A:N'], columns=['B:Y', 'B:N'])
    cm_percent = cm_counts / cm_counts.sum().sum() * 100
    cm_percent_df = pd.DataFrame(np.round(cm_percent, 1), index=['A:Y', 'A:N'], columns=['B:Y', 'B:N'])

    results = {
        'raw_percent_agreement': (raw_agreement, raw_ci),
        'cohens_kappa': (kappa, kappa_ci),
        'label_distribution_bias': (bias_index, bias_ci),
        'confusion_matrix_counts': cm_counts_df,
        'confusion_matrix_percent': cm_percent_df
    }
    return results

# -----------------------------
# Save results as green JPEG image, side-by-side matrices
# -----------------------------
def save_stats_side_by_side_green(stats, filename="annotator_stats_side_by_side_green.jpeg"):
    cm_counts = stats['confusion_matrix_counts']
    cm_percent = stats['confusion_matrix_percent']

    # Text metrics height
    fig, ax_text = plt.subplots(figsize=(10, 2))
    ax_text.axis('off')
    y = 1.0
    line_height = 0.08
    for metric, value in stats.items():
        if "confusion_matrix" not in metric:
            estimate, ci = value
            ax_text.text(0, y, f"{metric.replace('_', ' ').title()}: {estimate:.3f} "
                                f"(95% CI: {ci[0]:.3f}, {ci[1]:.3f})",
                         fontsize=10, va='top')
            y -= line_height
    plt.tight_layout()
    plt.savefig("temp_text.png", dpi=300)
    plt.close()

    # Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    matrices = [(cm_counts, "Counts"), (cm_percent, "Percent")]
    for ax, (df_cm, title) in zip(axes, matrices):
        im = ax.imshow(df_cm.values, cmap=plt.cm.Greens)
        ax.set_xticks(range(len(df_cm.columns)))
        ax.set_xticklabels(df_cm.columns)
        ax.set_yticks(range(len(df_cm.index)))
        ax.set_yticklabels(df_cm.index)
        ax.set_title(f"Annotator Agreement {title}", fontsize=12)
        for i in range(df_cm.shape[0]):
            for j in range(df_cm.shape[1]):
                value = df_cm.iloc[i, j]
                if title == "Percent":
                    text = f"{value:.1f}%"
                else:
                    text = f"{value}"
                color = 'white' if (title=="Percent" and value > 50) or (title=="Counts" and value > df_cm.values.max()/2) else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
        ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Green stats image with side-by-side matrices saved as {filename}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = pd.read_excel("zerosum_annotated_combined.xlsx")  # adjust path
    stats = analyze_binary_annotations(df, col_a='annotator1', col_b='annotator2', n_boot=5000)

    # Print stats
    print("Binary task agreement stats with 95% CI:\n")
    for metric, value in stats.items():
        if "confusion_matrix" in metric:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(value)
        else:
            estimate, ci = value
            print(f"{metric}: {estimate:.3f} (95% CI: {ci[0]:.3f}, {ci[1]:.3f})")

    save_stats_side_by_side_green(stats)

