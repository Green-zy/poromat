import pandas as pd

# Load the dense and porous data
dense_df = pd.read_csv("data/raw/dense.csv")
porous_df = pd.read_csv("data/raw/porous.csv")

# Concatenate both dataframes
combined_df = pd.concat([dense_df, porous_df], ignore_index=True)

# Save the combined data
combined_df.to_csv("data/full_data.csv", index=False)
print("Combined data saved to full_data.csv")
