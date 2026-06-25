import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the raw training dataset
train_data_path = "../data/ethics_raw/train.csv"

# Load the dataset
df = pd.read_csv(train_data_path)

# Display column names for inspection
print(df.columns)

# Keep only the label and input columns
df = df[['label', 'input']]

# Rename columns to match the training pipeline
df.columns = ['label', 'text']

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# Save the processed datasets
train_df.to_csv("../data/ethics_dataset_train.csv", index=False)
val_df.to_csv("../data/ethics_dataset_val.csv", index=False)

print("Dataset preprocessing completed successfully.")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
