import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Settings
st.set_page_config(
    page_title="AI Wellness & Health Monitoring",
    page_icon="🏥",
    layout="wide"
)

# Load Dataset
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\helath\wellness.csv.txt")

# Features and Target
X = df[[
    "SleepHours",
    "StudyHours",
    "ScreenTime",
    "WaterIntake",
    "ExerciseMinutes"
]]

y = df["BurnoutRisk"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# Title
st.title("🏥 AI Wellness & Health Monitoring System")

st.write(
    "Predict burnout risk and monitor overall health."
)

# Inputs
st.header("Student Wellness Information")

sleep = st.slider("Sleep Hours", 0, 12, 7)
study = st.slider("Study Hours", 0, 15, 5)
screen = st.slider("Screen Time (Hours)", 0, 15, 4)
water = st.slider("Water Intake (Liters)", 0, 6, 3)
exercise = st.slider("Exercise (Minutes)", 0, 120, 30)

st.header("Health Monitoring")

heart_rate = st.number_input(
    "Heart Rate (BPM)",
    min_value=40,
    max_value=200,
    value=75
)

steps = st.number_input(
    "Daily Steps",
    min_value=0,
    max_value=50000,
    value=8000
)

weight = st.number_input(
    "Weight (kg)",
    min_value=20,
    max_value=200,
    value=70
)

height = st.number_input(
    "Height (cm)",
    min_value=100,
    max_value=250,
    value=170
)

# Prediction
if st.button("Analyze Wellness & Health"):

    input_data = pd.DataFrame(
        [[sleep, study, screen, water, exercise]],
        columns=[
            "SleepHours",
            "StudyHours",
            "ScreenTime",
            "WaterIntake",
            "ExerciseMinutes"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max() * 100

    st.subheader("Burnout Prediction")

    if prediction == "High":
        st.error(
            f"🔥 High Burnout Risk ({probability:.1f}%)"
        )

    elif prediction == "Medium":
        st.warning(
            f"⚠️ Medium Burnout Risk ({probability:.1f}%)"
        )

    else:
        st.success(
            f"✅ Low Burnout Risk ({probability:.1f}%)"
        )

    # BMI
    bmi = weight / ((height / 100) ** 2)

    # Health Score
    health_score = (
        (sleep * 8)
        + (water * 8)
        + (exercise * 0.3)
        + (steps / 300)
    )

    if heart_rate > 100:
        health_score -= 15

    health_score = max(0, min(100, int(health_score)))

    st.subheader("Health Report")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Health Score",
        f"{health_score}/100"
    )

    col2.metric(
        "BMI",
        f"{bmi:.2f}"
    )

    col3.metric(
        "Heart Rate",
        f"{heart_rate} BPM"
    )

    # Heart Rate Analysis
    if heart_rate > 100:
        st.error("⚠️ High Heart Rate")

    elif heart_rate < 60:
        st.warning("⚠️ Low Heart Rate")

    else:
        st.success("✅ Heart Rate Normal")

    # Steps Analysis
    if steps < 3000:
        st.warning("🚶 Very Low Activity")

    elif steps < 7000:
        st.info("🚶 Moderate Activity")

    else:
        st.success("🏃 Active Lifestyle")

    # BMI Analysis
    if bmi < 18.5:
        st.warning("Underweight")

    elif bmi < 25:
        st.success("Normal Weight")

    elif bmi < 30:
        st.warning("Overweight")

    else:
        st.error("Obesity Risk")

    st.subheader("AI Recommendations")

    recommendations = []

    if sleep < 7:
        recommendations.append(
            "Increase sleep to 7-8 hours."
        )

    if water < 3:
        recommendations.append(
            "Drink more water."
        )

    if exercise < 30:
        recommendations.append(
            "Exercise daily."
        )

    if steps < 7000:
        recommendations.append(
            "Increase daily walking."
        )

    if heart_rate > 100:
        recommendations.append(
            "Monitor heart rate and reduce stress."
        )

    if screen > 8:
        recommendations.append(
            "Reduce screen time."
        )

    if len(recommendations) == 0:
        st.success(
            "Excellent lifestyle habits!"
        )
    else:
        for rec in recommendations:
            st.write("•", rec)

# Sidebar
st.sidebar.title("System Information")
st.sidebar.success(
    f"Model Accuracy: {accuracy:.2f}"
)