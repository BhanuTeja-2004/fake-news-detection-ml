import pandas as pd

# Load both datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "Fake"
true["label"] = "Real"

# Combine both datasets
data = pd.concat([fake, true])

# Shuffle data (important)
data = data.sample(frac=1).reset_index(drop=True)

# Save combined dataset
data.to_csv("Fake_News_Dataset.csv", index=False)

print("Dataset Combined Successfully!")