# Student Marks Prediction Using Machine Learning
*A beginner-level mini project (Python, Pandas, Scikit-learn, Matplotlib)*

---
## 1. Introduction
Machine Learning lets computers learn patterns from data. This project uses **Linear Regression**, one of the simplest ML algorithms, to predict a student's marks from the number of hours they study. It is a small end-to-end example of the ML workflow: collect data, train, test, predict.

## 2. Problem Statement
Students and teachers often want a quick estimate of the marks a student may score for a given amount of study time. The problem is to build a model that learns the relationship between **study hours (input)** and **marks (output)** from sample data and estimates marks for new inputs.

## 3. Objectives
- Create and load a sample dataset using Pandas.
- Understand the relationship between study hours and marks.
- Split data into training and testing sets.
- Train a Linear Regression model.
- Evaluate it using MAE, MSE, RMSE and R² score.
- Visualize data and the regression line.
- Predict marks for user-entered study hours.

## 4. Software Requirements
| Item | Details |
|---|---|
| Language | Python 3.8 or higher |
| Libraries | pandas, scikit-learn, matplotlib |
| Editor | VS Code, PyCharm, or Google Colab |
| OS | Windows / Linux / macOS |

Installation:
```
pip install pandas scikit-learn matplotlib
```

## 5. Dataset
**Important: this is illustrative sample data created for learning. It is not real student data.**
File: `student_data.csv` — 30 rows, 2 columns.

| Hours (study hours/day) | Marks (out of 100) |
|---|---|
| 1.0 | 19 |
| 1.5 | 22 |
| 2.0 | 33 |
| 3.0 | 41 |
| 5.5 | 63 |
| 7.0 | 78 |
| 9.0 | 99 |
| ... | ... (full file in the CSV) |

## 6. Algorithm: Linear Regression
Linear Regression finds the **best straight line** through the data points. Once the line is known, we can read off the marks for any study hours.

**Formula:**  `y = m·x + c`
- `y` = predicted marks
- `x` = study hours
- `m` = slope (extra marks gained per extra hour)
- `c` = intercept (marks when hours = 0)

The model chooses `m` and `c` so that the total squared distance between actual points and the line is as small as possible (Least Squares Method).

## 7. Complete Python Code
See `student_marks_prediction.py` (included in the zip; run-tested). Main parts:
1. Load CSV with Pandas
2. Separate X (Hours) and y (Marks)
3. `train_test_split` (80% train / 20% test)
4. `LinearRegression().fit()`
5. `predict()` on test data and compute metrics
6. Plot with Matplotlib and save `marks_graph.png`
7. Loop that asks the user for hours and predicts marks

## 8. Data Visualization
The program draws a scatter plot of study hours vs marks (blue = training data, green = test data) and a **red regression line** showing the model's predictions. It is saved as `marks_graph.png`.

## 9. Model Training and Testing
- Data split: 24 training rows, 6 testing rows (`random_state=42` so results repeat).
- Training: `model.fit(X_train, y_train)`.
- Learned equation: **Marks = 9.46 × Hours + 10.69**

| Metric | Meaning | Result |
|---|---|---|
| MAE | Average absolute error in marks | 2.27 |
| MSE | Average squared error | 8.85 |
| RMSE | Square root of MSE (in marks) | 2.97 |
| R² | How well the line fits (1 = perfect) | 0.969 |

## 10. Prediction
The user types study hours; the program applies the learned equation and prints the **predicted** marks (clamped between 0 and 100). Type `q` to quit.

## 11. Sample Output
**Actual vs Predicted (test data):**

| Hours | Actual Marks | Predicted Marks |
|---|---|---|
| 4.2 | 56 | 50.44 |
| 8.5 | 89 | 91.14 |
| 6.8 | 74 | 75.05 |
| 9.5 | 100 | 100.60 |
| 5.0 | 54 | 58.01 |
| 5.5 | 63 | 62.74 |

**User predictions:**
```
Enter study hours: 5.5
Predicted marks for 5.5 hours: 62.7 / 100 (an estimate, not actual marks)
Enter study hours: 9
Predicted marks for 9.0 hours: 95.9 / 100 (an estimate, not actual marks)
Enter study hours: abc
Invalid input. Enter a number or 'q'.
```
**Graph description:** Points rise from the bottom-left (1 hour, about 19 marks) to the top-right (9.5 hours, about 100 marks). The red line passes close to nearly all points, showing a strong positive relationship.

> Note: Results are from illustrative data. Real data would give different numbers.

## 12. Code Explanation
| Section | What it does |
|---|---|
| Imports | Bring in pandas (data), sklearn (model, split, metrics), matplotlib (graph). |
| `pd.read_csv` | Reads the dataset into a table (DataFrame). |
| `X = data[["Hours"]]` | Double brackets make X 2-dimensional, as scikit-learn requires. |
| `y = data["Marks"]` | The value we want to predict. |
| `train_test_split` | Keeps 20% of data hidden to test on unseen examples. |
| `model.fit` | The learning step: finds slope and intercept. |
| `model.predict` | Uses the line to estimate marks. |
| Metrics | Compare predicted vs actual marks to measure accuracy. |
| `plt.scatter / plt.plot` | Draw the points and regression line. |
| `while True` loop | Repeatedly takes user input, validates it, and predicts. |

## 13. Advantages and Limitations
**Advantages**
- Simple, fast and easy to understand.
- Results are interpretable (slope shows marks gained per hour).
- Needs little data and computing power.
- Good for learning the ML workflow.

**Limitations**
- Uses only one factor (study hours); marks depend on many things.
- Sample data is small and illustrative, so it may not match reality.
- Assumes a straight-line relationship; real learning shows diminishing returns.
- Predictions outside the data range (e.g. 20 hours) are unreliable.
- Gives an estimate, not a guarantee.

## 14. Applications
- Quick study-planning aid for students.
- Teachers identifying students who may need support.
- Educational dashboards and e-learning platforms.
- Basis for similar predictions (sales vs advertising, salary vs experience, price vs area).

## 15. Future Scope
- Collect real data from many students (with permission).
- Add features: attendance, previous marks, sleep, assignments.
- Use Multiple Linear Regression, then Polynomial Regression or Random Forest.
- Build a web app (Flask / Streamlit) or GUI.
- Store data in a database and add cross-validation.

## 16. Conclusion
The project shows how Linear Regression can estimate marks from study hours. It covers data loading, splitting, training, testing, evaluation, visualization and prediction. Learning outcomes: using Pandas and Scikit-learn, understanding train/test splits, reading error metrics, and knowing that a model's prediction is an estimate that depends on data quality.

## 17. Viva Questions and Answers
1. **What is Machine Learning?** Teaching computers to learn patterns from data and make predictions without explicit rules.
2. **What is Linear Regression?** A supervised algorithm that fits a straight line to model the relationship between input and a numeric output.
3. **Write the formula.** y = mx + c, where m is slope and c is intercept.
4. **What are the independent and dependent variables here?** Independent: study hours. Dependent: marks.
5. **Why split data into train and test sets?** To check the model on unseen data and avoid a falsely good score.
6. **What does `random_state=42` do?** Makes the split the same every run so results are reproducible.
7. **What is R² score?** Fraction of variation in marks explained by the model; closer to 1 means better fit.
8. **What are MAE and RMSE?** MAE is the average absolute error; RMSE is the square root of the average squared error and penalizes large errors more.
9. **Why is X written with double brackets?** scikit-learn expects a 2D input (rows × features).
10. **What are the limitations of the model?** Uses one feature, small illustrative data, assumes a straight line, unreliable outside the data range.
