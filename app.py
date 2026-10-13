import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Manila Weather AI Assistant",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Manila Weather AI Assistant")
st.write(
    "Explore the cleaned Manila weather dataset and predict "
    "rain frequency based on temperature."
)

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_manila_weather_data.csv")
    return data

df = load_data()

# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------
X = df[["Temperature"]]
y = df["Rain Frequency"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------------------
# SUMMARY METRICS
# ----------------------------------------
average_temperature = df["Temperature"].mean()
average_rain = df["Rain Frequency"].mean()
correlation = df["Temperature"].corr(df["Rain Frequency"])

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Temperature",
    f"{average_temperature:.2f} °C"
)

col2.metric(
    "Average Rain Frequency",
    f"{average_rain:.2f}"
)

col3.metric(
    "Temperature-Rain Correlation",
    f"{correlation:.3f}"
)

# ----------------------------------------
# DISPLAY DATASET
# ----------------------------------------
with st.expander("View Cleaned Manila Dataset"):
    st.dataframe(df, use_container_width=True)

# ----------------------------------------
# WHAT-IF WEATHER SIMULATOR
# ----------------------------------------
st.subheader("🌡️ What-If Weather Simulator")

st.write(
    "Adjust the temperature to explore how the trained regression "
    "model changes its estimated rain frequency."
)

min_temperature = float(df["Temperature"].min())
max_temperature = float(df["Temperature"].max())
default_temperature = float(df["Temperature"].mean())

temperature = st.slider(
    "Select a Manila temperature:",
    min_value=min_temperature,
    max_value=max_temperature,
    value=default_temperature,
    step=0.5,
    format="%.1f °C"
)

prediction_input = pd.DataFrame({
    "Temperature": [temperature]
})

predicted_rain = float(model.predict(prediction_input)[0])

historical_average = float(df["Rain Frequency"].mean())
difference = predicted_rain - historical_average

prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

prediction_col1.metric(
    "Selected Temperature",
    f"{temperature:.1f} °C"
)

prediction_col2.metric(
    "Predicted Rain Frequency",
    f"{predicted_rain:.2f}"
)

prediction_col3.metric(
    "Difference from Historical Average",
    f"{difference:+.2f}"
)

if difference > 0:
    st.info(
        "The predicted rain frequency is above the historical "
        "average in the cleaned Manila dataset."
    )
elif difference < 0:
    st.info(
        "The predicted rain frequency is below the historical "
        "average in the cleaned Manila dataset."
    )
else:
    st.info(
        "The predicted rain frequency is equal to the historical "
        "average in the cleaned Manila dataset."
    )

st.caption(
    "This is a model estimate based only on temperature. "
    "It is not an operational weather forecast."
)

# ----------------------------------------
# INTERACTIVE PREDICTIVE VISUALIZATION
# ----------------------------------------
st.subheader("📈 Actual vs Predicted Rain Frequency")

visual_df = df.sort_values("Temperature").copy()

visual_df["Predicted Rain Frequency"] = model.predict(
    visual_df[["Temperature"]]
)

chart_df = visual_df[
    [
        "Temperature",
        "Rain Frequency",
        "Predicted Rain Frequency"
    ]
].copy()

chart_df = chart_df.rename(
    columns={
        "Rain Frequency": "Actual Rain Frequency"
    }
)

chart_df = chart_df.set_index("Temperature")

st.line_chart(chart_df)

st.write(
    "**How to read the chart:** The actual values represent the "
    "cleaned historical observations. The predicted values represent "
    "the regression model's estimated rain frequency."
)

with st.expander("View Actual and Predicted Values"):
    comparison_df = visual_df[
        [
            "Month",
            "Temperature",
            "Rain Frequency",
            "Predicted Rain Frequency"
        ]
    ].copy()

    comparison_df["Predicted Rain Frequency"] = (
        comparison_df["Predicted Rain Frequency"].round(2)
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------------
# DOWNLOAD CLEANED DATA
# ----------------------------------------
cleaned_csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Cleaned Manila Dataset",
    data=cleaned_csv,
    file_name="cleaned_manila_weather_data.csv",
    mime="text/csv",
    use_container_width=True
)


# ----------------------------------------
# CHATBOT RESPONSE FUNCTION
# ----------------------------------------
def answer_question(question):
    question = question.lower().strip()

    if (
        "how does temperature affect" in question
        or "relationship" in question
        or "correlation" in question
    ):
        if correlation > 0:
            direction = "positive"
        elif correlation < 0:
            direction = "negative"
        else:
            direction = "neutral"

        return (
            f"The cleaned Manila dataset has a {direction} relationship "
            f"between temperature and rain frequency. "
            f"The calculated correlation is {correlation:.3f}. "
            "This result describes the supplied dataset and should not be "
            "treated as a complete operational weather forecast."
        )

    elif (
        "highest" in question
        and ("rain" in question or "frequency" in question)
    ):
        highest_row = df.loc[df["Rain Frequency"].idxmax()]

        highest_rain = float(highest_row["Rain Frequency"])
        highest_temperature = float(highest_row["Temperature"])
        highest_month = str(highest_row["Month"])

        return (
            f"The highest recorded rain frequency is "
            f"{highest_rain:.2f} in {highest_month}. "
            f"The corresponding temperature is "
            f"{highest_temperature:.1f} °C."
        )

    elif "average temperature" in question:
        return (
            f"The average temperature in the cleaned Manila dataset is "
            f"{average_temperature:.2f} °C."
        )

    elif "average rain" in question:
        return (
            f"The average rain frequency in the cleaned Manila dataset is "
            f"{average_rain:.2f}."
        )

    elif "predict" in question:
        numbers = [
            float(word.replace("°c", "").replace("c", ""))
            for word in question.replace(",", " ").split()
            if word.lower()
            .replace("°c", "")
            .replace("c", "")
            .replace(".", "", 1)
            .isdigit()
        ]

        if numbers:
            chatbot_temperature = numbers[0]

            chatbot_input = pd.DataFrame({
                "Temperature": [chatbot_temperature]
            })

            chatbot_prediction = float(
                model.predict(chatbot_input)[0]
            )

            return (
                f"At {chatbot_temperature:.1f} °C, the regression "
                f"model predicts a rain frequency of "
                f"{chatbot_prediction:.2f}. This estimate uses "
                "temperature as the only predictive feature."
            )

        return (
            "Include a temperature in your question, such as: "
            "'Predict rain frequency at 28°C.' You can also use "
            "the What-If Weather Simulator above."
        )

    else:
        return (
            "I can answer questions about the relationship between "
            "temperature and rain frequency, averages, the month with the "
            "highest rain frequency, and model predictions."
        )

# ----------------------------------------
# CHATBOT
# ----------------------------------------
st.subheader("Ask About the Dataset")

st.caption("Suggested questions:")

st.code(
    "How does temperature affect rain frequency in Manila "
    "based on the dataset?"
)

st.code(
    "Which month in Manila has the highest recorded rain "
    "frequency, and what is its temperature?"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question about the Manila weather dataset")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    response = answer_question(prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.write(response)

