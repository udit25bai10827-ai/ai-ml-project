import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [50, 55, 60, 65, 70, 80, 85, 90],
    "previous_marks": [30, 35, 40, 45, 50, 60, 70, 80],
    "result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

# Features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("student_model.pkl", "wb"))

print("Model trained and saved!")