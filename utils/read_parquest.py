import pandas as pd

# Read the parquet file
df = pd.read_parquet('/Users/helium/ncsa/pdg/water_timeseries_argo_workflow/output/testoutput8.parquet')

# Find date/datetime columns
date_columns = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns

# Check unique dates in each date column
for col in date_columns:
    print(f"\nColumn: {col}")
    print(f"Date range: {df[col].min()} to {df[col].max()}")
    print(f"Unique dates: {df[col].nunique()}")
    print(f"Sample dates: {df[col].dropna().head(10).tolist()}")