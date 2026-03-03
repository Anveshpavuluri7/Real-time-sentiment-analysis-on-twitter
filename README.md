# Real-Time Sentiment Analysis on Twitter

A real-time sentiment analysis project that fetches live tweet streams using the Twitter API and classifies each tweet into sentiment categories such as **Positive**, **Neutral**, or **Negative**. This project is built using Python and focuses on natural language processing (NLP) to transform live Twitter data into actionable insights.

---

## 📌 Project Overview

This repository demonstrates how to:

- 🐦 Connect to the Twitter API to receive real-time tweets
- 🧠 Perform sentiment analysis on streamed tweets using NLP
- 📊 Display sentiment results dynamically  
- 📁 Process text data for meaningful insight extraction

The primary goal is to understand the emotional tone of Twitter data in near-real time — a valuable tool for brand monitoring, public opinion tracking, market research, and social analytics.

---

## 🔧 Features

✔️ Connects to the Twitter API via `tweepy`  
✔️ Real-time tweet fetching based on keywords/filters  
✔️ Cleans and preprocesses tweet text  
✔️ Classifies tweets into sentiment labels  
✔️ Outputs sentiment distribution and trending metrics  
✔️ Can be extended to show visualizations or dashboards

---

## 🛠️ Built With

- **Python** – Core language  
- **Tweepy** – Twitter API integration  
- **NLTK / TextBlob / Transformers** – Natural Language Processing  
- **Pandas** – Data handling  
- **Matplotlib / Plotly / Streamlit** *(optional)* – Visualization

---

## 🐍 Requirements

Make sure you have:

- Python 3.7 or higher  
- Twitter Developer account (API keys / Bearer Token)
- Required Python libraries (listed below)

---

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Anveshpavuluri7/Real-time-sentiment-analysis-on-twitter.git
cd Real-time-sentiment-analysis-on-twitter
```

### 2. Install required libraries:
```bash
pip install tweepy pandas textblob nltk
```

### 3. 🔑 Twitter API Setup

You’ll need Twitter API access keys:
Go to https://developer.twitter.com

Create a new project and app
Generate the following credentials:

API Key
API Secret Key
Access Token
Access Token Secret
Bearer Token
Save them in a config file or export as environment variables:
```bash
export API_KEY="your_api_key"
export API_SECRET="your_api_secret"
export ACCESS_TOKEN="your_access_token"
export ACCESS_SECRET="your_access_secret"
export BEARER_TOKEN="your_bearer_token"
```
### How to run
```bash
python app.py
```

### Example Output 
```bash
Tweet: "I love the new update!" | Sentiment: Positive
Tweet: "This app keeps crashing…" | Sentiment: Negative
Tweet: "Not sure how I feel… 🤔" | Sentiment: Neutral
```
