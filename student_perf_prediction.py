import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


url = "https://raw.githubusercontent.com/hxrsh001/student-dataset/main/student_dataset_10000_rows.csv"
df = pd.read_csv(url)

print("Original Dataset:")
print(df, "\n")


# Label Encoding-
le = LabelEncoder()
df['placement_status'] = le.fit_transform(df['placement_status']) #(0 - not placed, 1 - placed)

# Feature Scaling-
scaler = StandardScaler()
df[['study_hours',
    'attendance',
    'sleep_hours',
    'internet_usage',
    'assignments_completed',
    'previous_score']] = scaler.fit_transform(
        df[['study_hours',
            'attendance',
            'sleep_hours',
            'internet_usage',
            'assignments_completed',
            'previous_score']]
)

print("Scaled Dataset: ")
print(df, "\n")


# Split Data-
x = df[['study_hours',
    'attendance',
    'sleep_hours',
    'internet_usage',
    'assignments_completed',
    'previous_score']]
y = df[['placement_status']]


# Train Test Split-
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

# Model Train-
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate Model-
y_pred = model.predict(x_test)

print("\nModel Performance")
print("----------------------")
print("R² Score :", round(r2_score(y_test, y_pred), 3))
print("MAE      :", round(mean_absolute_error(y_test, y_pred), 2))


# Prediction Data-

new_data = pd.DataFrame({
    "study_hours": [7.5],
    "attendance": [92],
    "sleep_hours": [7],
    "internet_usage": [3],
    "assignments_completed": [18],
    "previous_score": [84]
})

# Scale new data-
new_data_scaled = scaler.transform(new_data)

new_data_scaled = pd.DataFrame(
    new_data_scaled,
    columns=['study_hours',
    'attendance',
    'sleep_hours',
    'internet_usage',
    'assignments_completed',
    'previous_score']
)

prediction = model.predict(new_data_scaled)

if prediction[0][0] >= 0.5:
    print("\nPrediction : Student is likely to be Placed.")
else:
    print("\nPrediction : Student is NOT likely to be Placed.")

# Save Model-

joblib.dump(model, "student_model.joblib")
joblib.dump(scaler, "scaler1.joblib")
joblib.dump(le, "label_encoder.joblib")

# Visual Representation-
plt.figure(figsize=(12,5))

# Heatmap
plt.subplot(1,2,1)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

# Scatter Plot
plt.subplot(1,2,2)
sns.scatterplot(
    data=df,
    x="study_hours",
    y="exam_score",
    hue="placement_status"
)
plt.title("Study Hours vs Exam Score")

plt.tight_layout()
plt.show()