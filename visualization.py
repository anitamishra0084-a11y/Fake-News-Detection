import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("fake_news.csv")

# Display first 5 rows
print(data.head())

# Count labels
print(data["label"].value_counts())

# --------------------
# Bar Chart
# --------------------
data["label"].value_counts().plot(kind="bar")

plt.title("Fake vs Real News")

plt.xlabel("Label")

plt.ylabel("Count")

plt.show()

# --------------------
# Pie Chart
# --------------------
plt.figure(figsize=(6,6))

data["label"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Distribution of Fake and Real News")

plt.ylabel("")

plt.show()