import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter

# Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:1000]")
texts = [sample["instruction"] for sample in dataset]
labels = [sample["category"] for sample in dataset]

# Train classifier
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Predict category for a new instruction
new_instruction = "The Indian national anthem is Jana Gana Mana"
X_new = vectorizer.transform([new_instruction])
predicted_category = model.predict(X_new)[0]
print(f"Predicted category: {predicted_category}")

# Print context for the predicted category
for sample in dataset:
    if sample["category"] == predicted_category and sample.get("response"):
        print(f"Response: {sample['response']}")
        break
else:
    print("No response available for this category.")