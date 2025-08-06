import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Load Data
df = pd.read_csv("transfusion.csv")  # Ensure the correct path

# Title
st.title("🩸 ThalaMitra - AI Blood Donation Assistant")
st.write("Smart prediction, donor tracking, and reward insights for Thalassemia support.")

# Data Preview
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Features and Target
X = df.drop('whether he/she donated blood in March 2007', axis=1)
y = df['whether he/she donated blood in March 2007']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc*100:.2f}%")

# Sidebar input
st.sidebar.header("Donor Eligibility Checker")
recency = st.sidebar.slider("Recency (months)", 0, 50, 5)
frequency = st.sidebar.slider("Frequency (times)", 0, 50, 2)
monetary = st.sidebar.slider("Monetary (c.c. blood)", 250, 12500, 750)
time = st.sidebar.slider("Time (months)", 0, 100, 20)

# Predict Eligibility
input_data = np.array([[recency, frequency, monetary, time]])
prediction = model.predict(input_data)

# Calculate Donor Tier
def calculate_tier(frequency):
    if frequency >= 20:
        return "Gold Donor"
    elif frequency >= 10:
        return "Silver Donor"
    elif frequency >= 5:
        return "Bronze Donor"
    else:
        return "New Donor"

donor_tier = calculate_tier(frequency)

# Output
if st.button("Predict Donation Possibility"):
    if prediction[0] == 1:
        st.success("✅ This donor is likely to donate blood again.")
    else:
        st.warning("⚠️ This donor is unlikely to donate blood again.")
    st.info(f"🎖️ Donor Tier: {donor_tier}")

# Daily Eligible Donors Count
today = datetime.now()
eligible_donors = df[df['Recency (months)'] >= 4]  # assuming minimum 4 months gap
st.markdown("---")
st.subheader("📅 Daily Insights")
st.write(f"🔢 Approx. Eligible Donors Today: {len(eligible_donors)}")

# Placeholder for SMS Notification Feature
st.markdown("---")
st.subheader("📱 SMS Notification Preview")
if prediction[0] == 1:
    st.code(f"Hello! You are eligible to donate blood again. Kindly visit your nearest blood bank or book through ThalaMitra app.")

# Placeholder for e-RaktKosh Integration
st.markdown("---")
st.subheader("🧬 e-RaktKosh / Blood Bridge Status")
st.write("Integration with live e-RaktKosh API coming soon.")
st.info("This module will auto-fetch real-time blood requests and match them to eligible donors.")

# Leaderboard (Simulated)
st.markdown("---")
st.subheader("🏆 Top Donors Leaderboard (Simulated)")
leaderboard = pd.DataFrame({
    'Donor Name': ['Ravi', 'Sneha', 'Amit', 'Fatima'],
    'Total Donations': [24, 19, 17, 12],
    'Tier': ['Gold', 'Gold', 'Silver', 'Silver']
})
st.dataframe(leaderboard)

# Eligibility Self-Check Summary
st.markdown("---")
st.subheader("🔍 Self Check Summary")
st.write(f"You donated {frequency} time(s) in total.")
st.write(f"Your last donation was {recency} month(s) ago.")
st.write(f"You belong to: **{donor_tier}**")
