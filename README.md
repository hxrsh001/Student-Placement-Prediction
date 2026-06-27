# 🎓 Student Placement Prediction

A Machine Learning project that predicts whether a student is likely to be **Placed** or **Not Placed** based on academic and performance-related features. This project demonstrates a complete Machine Learning workflow, including data preprocessing, feature scaling, model training, prediction, data visualization, and model serialization.

---

## 🚀 Features

- Load dataset using Pandas
- Label Encoding for categorical data
- Feature Scaling using StandardScaler
- Split dataset into Training and Testing sets
- Train a Linear Regression model
- Evaluate model performance using R² Score and Mean Absolute Error (MAE)
- Predict placement status for new student data
- Save trained model using Joblib
- Visualize data using Seaborn and Matplotlib

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Joblib

---

## 📂 Dataset Features

The dataset is taken from the kaggle and contains the following attributes:

- Study Hours
- Attendance
- Sleep Hours
- Internet Usage
- Assignments Completed
- Previous Score
- Placement Status (Target)

---

## 📊 Data Visualization

The project includes the following visualizations:

- Correlation Heatmap
- Study Hours vs Exam Score Scatter Plot

These visualizations help in understanding the relationships between different features and the target variable.

---

## 📁 Project Structure

```
Student-Placement-Prediction/
│
├── student_perf_prediction.py
├── student_dataset_10000_rows.csv
├── student_prediction_output.png
├── visual_presentaion.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/hxrsh001/Student-Placement-Prediction.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Run the project

```bash
python student_perf_prediction.py
```

---

## 📈 Sample Prediction

Input Features:

- Study Hours: 7.5
- Attendance: 92%
- Sleep Hours: 7
- Internet Usage: 3 hours
- Assignments Completed: 18
- Previous Score: 84

Output:

```
Prediction: Student is likely to be Placed.
```

---

## 🔮 Future Improvements

- Replace Linear Regression with Logistic Regression for better classification performance.
- Improve model accuracy using advanced Machine Learning algorithms such as Decision Tree, Random Forest, and Support Vector Machine (SVM).
- Perform Hyperparameter Tuning to optimize model performance.
- Add more feature engineering and data preprocessing techniques.
- Build an interactive web application using Flask or Streamlit.
- Deploy the model online using Render or Hugging Face Spaces.
- Add more data visualizations and an interactive dashboard.
- Implement cross-validation for better model evaluation.
- Experiment with larger and more diverse datasets.

---

## 👨‍💻 Author

**Harsh Mishra**

B.Tech Computer Science & Artificial Intelligence

---

### ⭐ If you found this project useful, consider giving it a star!

## Output

The project predicts whether a student is likely to be **Placed** or **Not Placed** based on the given input features.
