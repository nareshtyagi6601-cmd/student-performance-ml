import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "student_data.csv"

df = pd.read_csv(DATA_FILE)
X = df[["study_hours","attendance","previous_marks","assignment_score","practice_hours"]]
y = df["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

def predict_marks():
    try:
        values = [
            float(study_entry.get()),
            float(att_entry.get()),
            float(prev_entry.get()),
            float(assign_entry.get()),
            float(practice_entry.get())
        ]
        if not (0 <= values[1] <= 100 and 0 <= values[2] <= 100 and 0 <= values[3] <= 100):
            raise ValueError

        result = model.predict(pd.DataFrame([values], columns=X.columns))[0]
        result = max(0, min(100, result))

        if result >= 75:
            performance = "Excellent"
        elif result >= 60:
            performance = "Good"
        elif result >= 40:
            performance = "Average"
        else:
            performance = "Needs Improvement"

        output.config(
            text=f"Predicted Marks: {result:.1f}/100\nPerformance: {performance}"
        )
    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers. Attendance, previous marks and assignment score must be 0-100."
        )

root = tk.Tk()
root.title("Student Performance Prediction System")
root.geometry("520x500")
root.resizable(False, False)

tk.Label(
    root, text="Student Performance Prediction System",
    font=("Arial", 18, "bold")
).pack(pady=20)

frame = tk.Frame(root)
frame.pack(pady=5)

labels = [
    ("Study Hours / Day", "study"),
    ("Attendance (%)", "attendance"),
    ("Previous Marks (%)", "previous"),
    ("Assignment Score (%)", "assignment"),
    ("Practice Hours / Day", "practice")
]

entries = {}
for i, (label, key) in enumerate(labels):
    tk.Label(frame, text=label, font=("Arial", 11)).grid(
        row=i, column=0, sticky="w", padx=10, pady=8
    )
    e = tk.Entry(frame, width=25)
    e.grid(row=i, column=1, padx=10, pady=8)
    entries[key] = e

study_entry = entries["study"]
att_entry = entries["attendance"]
prev_entry = entries["previous"]
assign_entry = entries["assignment"]
practice_entry = entries["practice"]

tk.Button(
    root, text="Predict Performance",
    command=predict_marks, font=("Arial", 12, "bold"),
    padx=15, pady=8
).pack(pady=20)

output = tk.Label(root, text="Enter student details and click Predict.",
                  font=("Arial", 13), justify="center")
output.pack(pady=10)

tk.Label(
    root,
    text=f"Model: Decision Tree Regressor | MAE: {mae:.2f} | R²: {r2:.2f}",
    font=("Arial", 9)
).pack(side="bottom", pady=15)

root.mainloop()
