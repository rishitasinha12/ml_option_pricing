# streamlit run app.py
# app.py
# Import necessary libraries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("ML-Driven Option Pricing & Trading Strategy Dashboard")

# Load pre-processed stock and options data (update file paths as needed)
df_stock = pd.read_csv("stock_data.csv", parse_dates=['Date'])
df_options = pd.read_csv("options_data.csv")

st.sidebar.header("Dashboard Controls")
date_range = st.sidebar.date_input("Select Date Range")
show_sentiment = st.sidebar.checkbox("Show News Sentiment")

st.subheader("Stock Data Overview")
st.write(df_stock.head())

st.subheader("Options Data Overview")
st.write(df_options.head())

if show_sentiment:
    # Load sentiment data (ensure you have a CSV or replace with a function call)
    df_sentiment = pd.read_csv("sentiment_data.csv")
    st.subheader("News Sentiment Over Time")
    st.line_chart(df_sentiment['Sentiment'])

if "Cumulative_PnL" in df_options.columns:
    st.subheader("Cumulative PnL")
    st.line_chart(df_options['Cumulative_PnL'])

st.subheader("Model Performance")
st.write("MSE:", 0.123)  # Replace with dynamic model performance metrics if available
