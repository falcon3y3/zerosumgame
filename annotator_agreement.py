#code for computing annotator agreement on ground truth dataset
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def bootstrap_ci(stat_func, labels_a, labels_b, n_boot=1000, ci=95, random_state=None):
    """
    Bootstrap confidence interval for a given statistic function.
    """
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

def clean_labels(series):
    """
    Normalize label strings to uppercase 'Y' or 'N' and validate.
    """
    series = series.astype(str).str.strip().str.upper()
    valid_labels = {'Y', 'N'}
    if not series.isin(valid_labels).all():
        bad_values = series[~series.isin(valid_labels)].unique()
        raise ValueError(f"Invalid labels found: {bad_values}. Only 'Y' or 'N' allowed.")
    return series

def analyze_binary_annotations(df, col_a='label_a', col_b='label_b', n_boot=1000):
    """
    Compute raw percent agreement, Cohen's kappa, label distribution bias,
    and confusion matrices for two annotators on a binary classification task.
    """
    df = df.dropna(subset=[col_a, col_b])
    
    # Clean and convert Y/N to binary
    labels_a_str = clean_labels(df[col_a])
    labels_b_str = clean_labels(df[col_b])
    map_labels = {'Y': 1, 'N': 0}
    labels_a = labels_a_str.map(map_labels)
    labels_b = labels_b_str.map(map_labels)

    # Stat functions
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

    # Confusion matrix counts
    cm_counts = confusion_matrix(labels_a_str, labels_b_str, labels=['Y', 'N'])
    cm_counts_df = pd.DataFrame(cm_counts, index=['A:Y', 'A:N'], columns=['B:Y', 'B:N'])

    # Confusion matrix percentages
    cm_percent = cm_counts / cm_counts.sum().sum() * 100
    cm_percent_df = pd.DataFrame(cm_percent, index=['A:Y', 'A:N'], columns=['B:Y', 'B:N'])

    results = {
        'raw_percent_agreement': (raw_agreement, raw_ci),
        'cohens_kappa': (kappa, kappa_ci),
        'label_distribution_bias': (bias_index, bias_ci),
        'confusion_matrix_counts': cm_counts_df,
        'confusion_matrix_percent': cm_percent_df
    }
    return results

if __name__ == "__main__":
    # Load Excel file
    df = pd.read_excel("zerosum_annotated_combined.xlsx")  # change path as needed
    
    # Change these to match your Excel column names
    stats = analyze_binary_annotations(df, col_a='annotator1', col_b='annotator2', n_boot=5000)
    
    print("Binary task agreement stats with 95% CI:\n")
    for metric, value in stats.items():
        if "confusion_matrix" in metric:
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(value)
        else:
            estimate, ci = value
            print(f"{metric}: {estimate:.3f} (95% CI: {ci[0]:.3f}, {ci[1]:.3f})")


