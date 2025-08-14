#this script removes any items of non-agreement between annotators
import pandas as pd

# === SETTINGS ===
input_file = "zerosum_annotated_combined.xlsx"       # Path to your Excel file
output_file = "groundtruth_filtered_file.xlsx"  # Where the cleaned file will be saved
col_a = "annotator1"  
col_b = "annotator2"  

# === LOAD EXCEL ===
df = pd.read_excel("zerosum_annotated_combined.xlsx")

# === CLEAN & FILTER ===
df_filtered = df[
    df[col_a].astype(str).str.strip().str.lower() ==
    df[col_b].astype(str).str.strip().str.lower()
]

# === SAVE TO NEW EXCEL FILE ===
df_filtered.to_excel(output_file, index=False)

print(f"Filtering complete. Saved {len(df_filtered)} rows to '{output_file}'.")
