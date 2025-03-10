# sentiment.py
import requests
from textblob import TextBlob
import pandas as pd

def fetch_news(api_key, query="Nvidia", page_size=20):
    """
    Fetch news articles related to the query using NewsAPI.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get("articles", [])
    return articles

def analyze_sentiment(articles):
    """
    Analyze sentiment of articles using TextBlob.
    Returns the average polarity.
    """
    sentiments = []
    for article in articles:
        text = article["title"]  # You can combine title, description, etc.
        analysis = TextBlob(text)
        sentiments.append(analysis.sentiment.polarity)
    return sum(sentiments) / len(sentiments) if sentiments else 0

if __name__ == "__main__":
    API_KEY = "YOUR_NEWSAPI_KEY"  # Replace with your actual NewsAPI key
    articles = fetch_news(API_KEY, query="Nvidia")
    avg_sentiment = analyze_sentiment(articles)
    print("Average News Sentiment for Nvidia:", avg_sentiment)
