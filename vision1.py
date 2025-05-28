# --- Imports ---
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import pandas as pd
import requests
from textblob import TextBlob
import matplotlib.pyplot as plt

# --- Load environment variables ---
load_dotenv()

# --- Configure Gemini API ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Nutritionix API Details ---
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_API_KEY = os.getenv("NUTRITIONIX_API_KEY")

# --- Load Gemini model ---
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Helper Functions ---

def detect_food_name(image):
    response = model.generate_content([
        "Identify the food item and briefly mention toppings/sauces if visible.",
        image
    ])
    return response.text.strip()

def get_nutrition_info(food_name):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json"
    }
    body = {"query": food_name, "timezone": "US/Eastern"}

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        if 'foods' in data:
            nutrition = data['foods'][0]
            return {
                "Food": nutrition['food_name'].title(),
                "Calories": nutrition.get('nf_calories'),
                "Proteins (g)": nutrition.get('nf_protein'),
                "Sweetness (g)": nutrition.get('nf_sugars') or 0,
                "Sodium (mg)": nutrition.get('nf_sodium'),
                "Total Fat (g)": nutrition.get('nf_total_fat'),
                "Cholesterol (mg)": nutrition.get('nf_cholesterol'),
                "Potassium (mg)": nutrition.get('nf_potassium'),
                "Total Carbohydrates (g)": nutrition.get('nf_total_carbohydrate')
            }
    return None

def extract_health_goals(prompt):
    prompt = prompt.lower()
    return {
        "diabetes": "diabetes" in prompt,
        "weight_loss": "weight loss" in prompt or "losing weight" in prompt,
        "high_bp": "blood pressure" in prompt or "bp" in prompt or "hypertension" in prompt
    }

def give_recommendations(nutrition, health_goals=None):
    advice = []
    health_goals = health_goals or {}

    calories = nutrition.get('Calories') or 0
    sweetness = nutrition.get('Sweetness (g)') or 0
    sodium = nutrition.get('Sodium (mg)') or 0
    protein = nutrition.get('Proteins (g)') or 0
    fat = nutrition.get('Total Fat (g)') or 0
    cholesterol = nutrition.get('Cholesterol (mg)') or 0
    potassium = nutrition.get('Potassium (mg)') or 0
    carbs = nutrition.get('Total Carbohydrates (g)') or 0

    if calories > 500:
        msg = "⚠️ High in calories."
        if health_goals.get("weight_loss"):
            msg += " Try to limit this if you're trying to lose weight."
        advice.append(msg)

    if sweetness > 20:
        msg = "🍬 High sweetness level."
        if health_goals.get("diabetes"):
            msg += " This may spike blood sugar — go for lower glycemic index foods."
        else:
            msg += " Consume moderately to avoid sugar crashes."
        advice.append(msg)

    if sodium > 1500:
        msg = "🧂 Very salty!"
        if health_goals.get("high_bp"):
            msg += " High sodium may increase blood pressure — consider low-sodium alternatives."
        advice.append(msg)

    if protein > 15:
        advice.append("💪 Good protein source — helpful for muscle maintenance.")

    if fat > 20:
        advice.append("⚡ High in fat — consume moderately if aiming for weight control.")

    if cholesterol > 200:
        advice.append("🫀 High cholesterol content — reduce intake for heart health.")

    if potassium and potassium > 400:
        advice.append("🍌 Good potassium source — helps with muscle function and blood pressure regulation.")

    if carbs > 50:
        advice.append("🥖 High in carbs — if you're watching carbs (e.g., keto diet), consider alternatives.")

    if not advice:
        advice.append("✅ Balanced food. Looks like a healthy option!")

    return advice

def analyze_sentiment(comment):
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "😊 Positive"
    elif polarity < 0:
        return "😔 Negative"
    else:
        return "😐 Neutral"

def save_feedback(user_name, user_age, satisfaction, comment):
    sentiment = analyze_sentiment(comment)
    feedback_file = "feedback.csv"
    new_feedback = pd.DataFrame([{
        "Name": user_name,
        "Age": user_age,
        "Satisfaction": satisfaction,
        "Comment": comment,
        "Sentiment": sentiment
    }])

    if os.path.exists(feedback_file):
        existing = pd.read_csv(feedback_file)
        all_feedback = pd.concat([existing, new_feedback], ignore_index=True)
    else:
        all_feedback = new_feedback

    all_feedback.to_csv(feedback_file, index=False)

def show_sentiment_graph():
    if os.path.exists("feedback.csv"):
        feedback = pd.read_csv("feedback.csv")
        sentiment_counts = feedback['Sentiment'].value_counts()

        st.subheader("📊 Sentiment Analysis Overview")
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=["lightgreen", "lightcoral", "lightgrey"])
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("No feedback available yet to show sentiment graph.")

# --- Streamlit App Setup ---
st.set_page_config(page_title="Smart Food Analyzer", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("🍽️ Navigation")
page = st.sidebar.radio("Go to", ["Analyze Food", "Give Feedback"])
st.session_state.page = "analyze" if page == "Analyze Food" else "feedback"

# --- Analyze Food Page ---
if st.session_state.page == "analyze":
    st.title("🍴 Smart Food Analyzer")

    uploaded_file = st.file_uploader("Upload a food image (or skip to use prompt):", type=["jpg", "jpeg", "png"])
    image = None
    input_text = st.text_input("Input prompt (e.g., 'I have diabetes, suggest meals')")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("✨ Analyze Food"):
        if image:
            with st.spinner('Detecting food from image...'):
                food_name = detect_food_name(image)
            st.success(f"🍽️ Detected Food: **{food_name}**")

            # Allow user to edit food name
            food_name = st.text_input("📝 Confirm or edit the food name:", value=food_name)

            with st.spinner('Fetching nutrition information...'):
                nutrition = get_nutrition_info(food_name)

            if nutrition:
                st.subheader("🥗 Nutrition Facts:")
                for key, value in nutrition.items():
                    st.write(f"**{key}**: {value}")

                health_goals = extract_health_goals(input_text)

                with st.expander("🔍 Detected Health Context"):
                    for k, v in health_goals.items():
                        if v:
                            st.write(f"✔️ {k.replace('_', ' ').title()}")

                st.subheader("🩺 Dietary Recommendations:")
                recommendations = give_recommendations(nutrition, health_goals)
                for rec in recommendations:
                    st.info(rec)

                # Meal suggestions
                if input_text.strip():
                    st.subheader("🍽️ Meal Suggestions Based on Your Prompt")
                    full_context = f"""
                    The user has uploaded a food item with the following nutrition:

                    {nutrition}

                    User's message: "{input_text}"

                    Please provide meal suggestions or healthy alternatives based on the user's dietary needs or preferences.
                    """
                    with st.spinner("Generating meal suggestions..."):
                        meal_response = model.generate_content(full_context)
                    st.write(meal_response.text.strip())

                st.session_state.nutrition_info = nutrition
            else:
                st.error("⚠️ Could not fetch nutrition information. Try another image or a more specific name.")

        elif input_text:
            st.subheader("📝 Response:")
            with st.spinner("Generating response to your prompt..."):
                response = model.generate_content(input_text)
            st.write(response.text)
        else:
            st.warning("Please upload an image or enter a prompt.")

# --- Feedback Page ---
elif st.session_state.page == "feedback":
    st.title("📝 Share Your Feedback")

    st.subheader("Your Nutrition Info (Summary):")
    nutrition = st.session_state.get("nutrition_info", {})
    if nutrition:
        for key, value in nutrition.items():
            st.write(f"**{key}**: {value}")
    else:
        st.info("No nutrition data available yet. Please analyze food first.")

    st.markdown("---")
    st.subheader("Feedback Form:")

    user_name = st.text_input("👤 Your Name")
    user_age = st.number_input("🎂 Your Age", min_value=1, max_value=120, step=1)
    satisfaction = st.slider("⭐ Rate Your Satisfaction (1 - 10)", 1, 10, 5)
    comment = st.text_area("🖋️ Your Comment")

    if st.button("✅ Submit Feedback"):
        if user_name and comment:
            save_feedback(user_name, user_age, satisfaction, comment)
            st.success("🎉 Thanks for your feedback!")
            st.balloons()
            st.toast("Feedback Submitted Successfully!", icon="✅")
        else:
            st.error("⚠️ Please fill in both Name and Comment.")

    # Sentiment Graph
    show_sentiment_graph()

    

