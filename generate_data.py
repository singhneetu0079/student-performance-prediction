
import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_score = np.random.uniform(30, 95, n)
sleep_hours = np.random.uniform(4, 10, n)
practice_tests = np.random.randint(0, 15, n)

# Generate final scores with some random variation
final_score = (
    3 * study_hours
    + 0.2 * attendance
    + 0.3 * previous_score
    + 1.5 * sleep_hours
    + 0.8 * practice_tests
    + np.random.normal(0, 5, n)
)

final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "practice_tests": practice_tests,
    "final_score": final_score
})

df.to_csv("data/student_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Dataset shape:", df.shape)
