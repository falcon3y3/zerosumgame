#little script to give us how many examples and non-examples in the finalized ground truth set
import pandas as pd

# Load Excel file
df = pd.read_excel("groundtruth_filtered_file.xlsx")  

# Column to check
col_name = "annotator1"  # change to the exact column name

# Clean the column (remove spaces, make uppercase)
col_clean = df[col_name].astype(str).str.strip().str.upper()

# Count 'Y'
count_y = (col_clean == "Y").sum()

# Count 'N'
count_n = (col_clean == "N").sum()

print(f"Number of rows in '{col_name}' containing 'Y': {count_y}")
print(f"Number of rows in '{col_name}' containing 'N': {count_n}")


