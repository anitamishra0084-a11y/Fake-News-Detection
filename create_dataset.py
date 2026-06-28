import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Keep only required columns
fake = fake[["text", "label"]]
true = true[["text", "label"]]

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Save new dataset
data.to_csv("fake_news.csv", index=False)

print("fake_news.csv created successfully!")