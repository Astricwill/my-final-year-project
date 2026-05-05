import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Lifestyle Predictor",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    BASE_DIR / "data" / "Cleaned_dataset.csv",
    BASE_DIR / "Cleaned_dataset.csv",
]
MODEL_PATH = BASE_DIR / "model" / "rf_model.pkl"

LIFESTYLE_LABELS = {
    0: "Unhealthy",
    1: "Moderately Healthy",
    2: "Healthy",
}

CATEGORY_OPTIONS = {
    "Age": ["16 - 18", "19 -21", "22 -24"],
    "Gender": ["Female", "Male"],
    "Level": ["100 Level", "200 Level", "300 Level", "400 Level", "500 Level"],
    "Faculty": [
        "College of Medical Sciences",
        "Computing and Applied Sciences",
        "Engineering",
        "Law",
        "Management and Social Sciences",
    ],
    "SleepHours": ["4-5 hours", "6-7 hours", "Less than 4 hours", "More than 7 hours"],
    "Rested": ["Always", "Rarely", "Sometimes"],
    "StudyHours": ["1-2 hours", "3-4 hours", "Less than 1 hour", "More than 4 hours"],
    "ScreenTime": ["2-4 hours", "5-7 hours", "Less than 2 hours", "more than 7 hours"],
    "ExerciseFreq": ["1-2 times", "3-4time", "5 or more times", "Never"],
    "ExerciseDuration": ["20-40 minutes", "40-60 minutes", "Less then 20 minutes", "More than 60 minutes"],
    "DietQuality": ["Fairly healthy", "Unhealthy", "Very healthy", "Very unhealthy"],
    "MealSkipping": ["Always", "Never", "Often", "Sometimes"],
    "StressFrequency": ["Always", "Never", "Often", "Sometimes"],
    "StressManagement": ["Fairly well", "Poorly", "Very poorly", "Very well"],
}

ENCODING_MAPS = {
    key: {option: idx for idx, option in enumerate(options)}
    for key, options in CATEGORY_OPTIONS.items()
}

FEATURE_ORDER = [
    "Age",
    "Gender",
    "Level",
    "Faculty",
    "SleepHours",
    "Rested",
    "StudyHours",
    "ScreenTime",
    "ExerciseFreq",
    "ExerciseDuration",
    "DietQuality",
    "MealSkipping",
    "StressFrequency",
    "StressManagement",
]

@st.cache_data
def load_data() -> pd.DataFrame:
    for path in DATA_PATHS:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Cleaned_dataset.csv not found in project root or data folder.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found at model/rf_model.pkl. Run preprocessing and save the model first.")
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_lifestyle_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["Lifestyle"]
        .value_counts()
        .rename(index=LIFESTYLE_LABELS)
        .rename_axis("Lifestyle")
        .reset_index(name="Count")
    )

@st.cache_data
def get_summary_metrics(df: pd.DataFrame) -> dict:
    return {
        "Total Responses": len(df),
        "Healthy Students": int((df["Lifestyle"] == 2).sum()),
        "Moderately Healthy": int((df["Lifestyle"] == 1).sum()),
        "Unhealthy": int((df["Lifestyle"] == 0).sum()),
        "Average Score": float(df["score"].mean()),
    }


def encode_inputs(inputs: dict) -> pd.DataFrame:
    encoded = {key: ENCODING_MAPS[key][value] for key, value in inputs.items()}
    return pd.DataFrame([encoded], columns=FEATURE_ORDER)


def predict_lifestyle(model, input_df: pd.DataFrame) -> tuple[int, float, dict]:
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    proba_by_label = {LIFESTYLE_LABELS[i]: float(probabilities[i]) for i in range(len(probabilities))}
    confidence = float(probabilities[prediction])
    return prediction, confidence, proba_by_label


def main():
    st.title("Student Lifestyle Prediction App")
    st.write(
        "Use this app to explore the student lifestyle dataset, review model performance, "
        "and predict a lifestyle category from student survey responses."
    )

    df = load_data()
    model = load_model()

    overview, explorer, predictor = st.tabs(["Overview", "Explore Data", "Predict Lifestyle"])

    with overview:
        st.subheader("Dataset summary")
        metrics = get_summary_metrics(df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Responses", metrics["Total Responses"])
        col2.metric("Healthy", metrics["Healthy Students"])
        col3.metric("Moderately Healthy", metrics["Moderately Healthy"])
        col4.metric("Unhealthy", metrics["Unhealthy"])

        st.markdown("### Lifestyle distribution")
        dist = get_lifestyle_distribution(df)
        st.bar_chart(data=dist.set_index("Lifestyle"))

        st.markdown("### Model and dataset information")
        st.write(
            "This app uses the cleaned student lifestyle dataset and a pre-trained Random Forest model."
        )
        stats = df[FEATURE_ORDER + ["score", "Lifestyle"]].describe().T
        st.dataframe(stats.style.format({"mean": "{:.2f}", "std": "{:.2f}"}))

    with explorer:
        st.subheader("Cleaned dataset preview")
        st.dataframe(df.head(20))

        st.markdown("### Filter and inspect")
        fac_filter = st.multiselect("Filter by faculty", CATEGORY_OPTIONS["Faculty"], default=CATEGORY_OPTIONS["Faculty"])
        age_filter = st.multiselect("Filter by age group", CATEGORY_OPTIONS["Age"], default=CATEGORY_OPTIONS["Age"])
        fac_values = [ENCODING_MAPS["Faculty"][x] for x in fac_filter] if fac_filter else list(ENCODING_MAPS["Faculty"].values())
        age_values = [ENCODING_MAPS["Age"][x] for x in age_filter] if age_filter else list(ENCODING_MAPS["Age"].values())
        filtered = df[df["Faculty"].isin(fac_values) & df["Age"].isin(age_values)]
        st.write(f"Showing {len(filtered)} rows after filtering.")
        st.dataframe(filtered.head(50))

        st.markdown("### Feature counts")
        chart_feature = st.selectbox(
            "Choose feature to plot",
            [
                "SleepHours",
                "Rested",
                "StudyHours",
                "ScreenTime",
                "ExerciseFreq",
                "ExerciseDuration",
                "DietQuality",
                "MealSkipping",
                "StressFrequency",
                "StressManagement",
            ],
            index=0,
        )
        feature_counts = filtered[chart_feature].value_counts().sort_index()
        if chart_feature in CATEGORY_OPTIONS:
            reverse_map = {v: k for k, v in ENCODING_MAPS[chart_feature].items()}
            feature_counts.index = feature_counts.index.map(reverse_map)
        st.bar_chart(feature_counts)

    with predictor:
        st.subheader("Predict student lifestyle")
        st.write("Enter survey responses below to predict the lifestyle category using the trained Random Forest model.")

        with st.form(key="lifestyle_form"):
            cols = st.columns(2)
            inputs = {}
            for idx, feature in enumerate(FEATURE_ORDER):
                target_col = cols[idx % 2]
                inputs[feature] = target_col.selectbox(feature, CATEGORY_OPTIONS[feature])
            submit = st.form_submit_button("Predict")

        if submit:
            input_df = encode_inputs(inputs)
            prediction, confidence, probabilities = predict_lifestyle(model, input_df)
            st.success(f"Predicted lifestyle: {LIFESTYLE_LABELS[prediction]}")
            st.write(f"Confidence: {confidence:.1%}")
            prob_df = pd.DataFrame.from_dict(probabilities, orient="index", columns=["Probability"]).rename_axis("Lifestyle").reset_index()
            st.bar_chart(prob_df.set_index("Lifestyle"))
            st.markdown("### Encoded model input")
            st.write(input_df)

    st.sidebar.title("About")
    st.sidebar.write(
        "Student Lifestyle Predictor is built for your final year project using the cleaned Topfaith University student dataset. "
        "The app supports dataset exploration and lifestyle classification with a Random Forest model."
    )
    st.sidebar.markdown("**How to run**: `streamlit run ui.py`")


if __name__ == "__main__":
    main()
