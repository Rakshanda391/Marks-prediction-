"""
Student Marks Prediction Using Machine Learning
------------------------------------------------
Predicts a student's marks from study hours using Linear Regression.
NOTE: The dataset (student_data.csv) is ILLUSTRATIVE sample data only.

Run:  python student_marks_prediction.py
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1. Load the data ----------
# Look for the CSV next to this script (works in VS Code / PyCharm).
# In Google Colab, upload student_data.csv and it is found in the current folder.
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:               # __file__ does not exist in notebooks/Colab
    base_dir = os.getcwd()

data = pd.read_csv(os.path.join(base_dir, "student_data.csv"))
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset size:", data.shape[0], "rows")
print("Missing values:", int(data.isnull().sum().sum()))

# ---------- 2. Separate input (X) and output (y) ----------
X = data[["Hours"]]   # input: study hours (2D, as scikit-learn expects)
y = data["Marks"]     # output: actual marks

# ---------- 3. Split into training and testing sets (80% / 20%) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", len(X_train), "| Testing samples:", len(X_test))

# ---------- 4. Train the Linear Regression model ----------
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained!")
print(f"Slope (m)     : {model.coef_[0]:.2f}")
print(f"Intercept (c) : {model.intercept_:.2f}")
print(f"Equation      : Marks = {model.coef_[0]:.2f} x Hours + {model.intercept_:.2f}")

# ---------- 5. Test the model ----------
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Hours": X_test["Hours"].values,
    "Actual Marks": y_test.values,
    "Predicted Marks": y_pred.round(2),
})
print("\nActual vs Predicted marks (test data):")
print(comparison.to_string(index=False))

# ---------- 6. Evaluate performance ----------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error  (MSE) : {mse:.2f}")
print(f"Root MSE (RMSE)           : {rmse:.2f}")
print(f"R2 Score                  : {r2:.3f}")

# ---------- 7. Visualization ----------
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="blue", label="Training data (actual)")
plt.scatter(X_test, y_test, color="green", label="Test data (actual)")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression line (predicted)")
plt.title("Study Hours vs Marks (Illustrative Data)")
plt.xlabel("Study Hours per Day")
plt.ylabel("Marks")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(base_dir, "marks_graph.png"), dpi=150, bbox_inches="tight")
print("\nGraph saved as marks_graph.png")
if matplotlib.get_backend().lower() != "agg":
    plt.show()

# ---------- 8. Predict marks for new input ----------
print("\n--- Predict Your Marks ---")
print("Type study hours (e.g. 5.5) and press Enter. Type 'q' to quit.")
while True:
    text = input("Enter study hours: ").strip()
    if text.lower() == "q":
        print("Goodbye!")
        break
    try:
        hours = float(text)
        if hours < 0 or hours > 24:
            print("Please enter hours between 0 and 24.")
            continue
        new_data = pd.DataFrame({"Hours": [hours]})
        predicted = min(100, max(0, model.predict(new_data)[0]))  # keep within 0-100
        print(f"Predicted marks for {hours} hours: {predicted:.1f} / 100 "
              "(an estimate, not actual marks)")
    except ValueError:
        print("Invalid input. Enter a number or 'q'.")
