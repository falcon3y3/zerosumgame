#metricvisuals_fewshot.py
#  ===== METRICS =====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

predictions_df = pd.read_csv("results/fewshot_predictions.csv")
print(predictions_df.head())
valid_preds = predictions_df.dropna(subset=["predicted_zero_sum"])
y_true = valid_preds["ground_truth"]
y_pred = valid_preds["predicted_zero_sum"]

# Compute standard metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix as heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,       # Show counts
    fmt="d",          # Integer format
    cmap="Blues",     # Light blue palette
    cbar=True
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# Optional: save figure
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
