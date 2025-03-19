import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb

# Load trained XGBoost model
model = joblib.load("final_xgb_model.pkl")

# Set Page Config
st.set_page_config(page_title="YouTube Revenue Predictor", page_icon="📊", layout="centered")

# App Title
st.title("📊 YouTube Revenue Predictor")
st.write("### Predict your estimated YouTube video revenue based on key metrics.")

# Sidebar Section
st.sidebar.title("⚙️ Enter Video Details")

# Function to display label correctly
def labeled_input(label, emoji, description, key, min_value=0, step=1):
    st.sidebar.markdown(f"**{emoji} {label}**  \n*{description}*", unsafe_allow_html=True)
    return st.sidebar.number_input("", min_value=min_value, step=step, key=key)

# User Inputs with Proper Labels
views = labeled_input("Total Video Views", "👀", "How many people watched the video?", "views", step=1000)
subscribers = labeled_input("Channel Subscribers", "📌", "How many people subscribed?", "subscribers", step=100)
likes = labeled_input("Total Likes", "👍", "Total likes received on the video", "likes", step=10)
shares = labeled_input("Total Shares", "🔄", "How many times the video was shared", "shares", step=1)
comments = labeled_input("New Comments", "💬", "Number of new comments received", "comments", step=1)
engagement_rate = labeled_input("Engagement Rate (%)", "📊", "Likes, Shares, Comments per View", "engagement_rate", min_value=0.0, step=0.1)
ad_impressions = labeled_input("Ad Impressions", "📺", "Total number of ads displayed", "ad_impressions", step=10)
monetized_playbacks = labeled_input("Monetized Playbacks", "💰", "Videos where ads were played", "monetized_playbacks", step=10)
playback_cpm = labeled_input("Playback-Based CPM (USD)", "💵", "Earnings per 1,000 ad playbacks", "playback_cpm", min_value=0.0, step=0.1)
cpm = labeled_input("Overall CPM (USD)", "💲", "Total earnings per 1,000 impressions", "cpm", min_value=0.0, step=0.1)
revenue_per_1000_views = labeled_input("Revenue per 1000 Views (USD)", "💰", "Estimated earnings per 1,000 views", "revenue_per_1000_views", min_value=0.0, step=0.1)

# Prediction Button
if st.sidebar.button("🚀 Predict Revenue", key="predict_button"):
    st.subheader("📌 Prediction Results")

    # Create input DataFrame
    input_data = pd.DataFrame([[views, subscribers, likes, shares, comments, engagement_rate,
                                revenue_per_1000_views, ad_impressions, monetized_playbacks,
                                playback_cpm, cpm]],
                              columns=['Views', 'Subscribers', 'Likes', 'Shares', 'New Comments',
                                       'Engagement Rate', 'Revenue per 1000 Views (USD)',
                                       'Ad Impressions', 'Monetized Playbacks (Estimate)',
                                       'Playback-Based CPM (USD)', 'CPM (USD)'])

    # Make prediction
    try:
        predicted_revenue = model.predict(input_data)[0]
        st.success(f"💲 **Predicted Revenue:** ${predicted_revenue:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
