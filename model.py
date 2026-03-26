import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample dataset
data = {
    "study_hours": [
        1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,1,2,3,4,5,
        6,7,8,1,2,3,4,5,6,7
    ],
    
    "attendance": [
        50,55,60,65,70,75,80,85,60,65,70,75,80,85,90,55,60,65,70,75,
        80,85,90,50,55,60,65,70,75,80
    ],
    
    "previous_marks": [
        20,25,30,35,40,50,60,70,15,20,25,35,45,55,65,10,15,20,30,40,
        50,60,70,5,10,20,30,40,50,60
    ],
    
    "result": [
        "Fail","Fail","Fail","Fail","Pass","Pass","Pass","Pass",
        "Fail","Fail","Fail","Pass","Pass","Pass","Pass",
        "Fail","Fail","Fail","Fail","Pass",
        "Pass","Pass","Pass",
        "Fail","Fail","Fail","Fail","Pass","Pass","Pass"
    ]
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
