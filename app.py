import streamlit as st
from gdeltdoc import GdeltDoc, Filters
import pandas as pd
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import google.generativeai as genai
import numpy as np
from matplotlib.dates import DateFormatter

# Set up Streamlit app in wide mode and force light theme
st.set_page_config(page_title="Global News Explorer", page_icon="📰", layout="wide")

# Apply custom styling for the containers and overall look
st.markdown("""
    <style>
    .container {
        padding: 15px;
        border: 2px solid #cccccc;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        padding: 0.5em 1em;
        border-radius: 5px;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .reportview-container {
        background-color: #F0F2F6;
    }
    h1 {
        color: #333333;
        text-align: center;
        font-size: 2.5em;
        margin-top: 20px;
    }
    .summary {
        font-size: 1.2em;
        font-family: Arial, sans-serif;
        line-height: 1.5;
        color: #444444;
    }
    a.article-link {
        color: #007BFF;
        text-decoration: none;
    }
    a.article-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("<h1>📰 Global News Explorer 🌍</h1>", unsafe_allow_html=True)

# User input for search parameters
with st.sidebar:
    st.header("🔍 Search Parameters")
    
    keyword = st.text_input("Enter Keyword", "Climate Change")
    
    lookback_options = ["1 week", "1 month", "3 months", "6 months", "1 year"]
    lookback_period = st.selectbox("Select Time Range", lookback_options, index=1)

    # Checkbox to limit search to English articles only
    limit_to_english = st.checkbox("Limit to English Articles", value=True)

    search_button = st.button("Search Articles")

# Set up GDELT and Generative AI API
gd = GdeltDoc()
genai.configure(api_key=st.secrets["api_key"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to query GDELT API for articles, tone chart, and timeline data
def query_gdelt_data(keyword, lookback_period, limit_to_english):
    # Get date range based on lookback period
    end_date = datetime.now()
    if lookback_period == "1 week":
        start_date = end_date - pd.DateOffset(weeks=1)
    elif lookback_period == "1 month":
        start_date = end_date - pd.DateOffset(months=1)
    elif lookback_period == "3 months":
        start_date = end_date - pd.DateOffset(months=3)
    elif lookback_period == "6 months":
        start_date = end_date - pd.DateOffset(months=6)
    elif lookback_period == "1 year":
        start_date = end_date - pd.DateOffset(years=1)

    # Format date for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Search filters
    filters = Filters(keyword=keyword, start_date=start_date_str, end_date=end_date_str)
    
    # Get articles
    articles = gd.article_search(filters)
    articles_df = pd.DataFrame(articles)
    
    if limit_to_english:
        articles_df = articles_df[articles_df['language'] == 'English']
    
    return articles_df, start_date_str, end_date_str

# Function to generate word cloud
def generate_wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stopwords,
        colormap='viridis',
        collocations=False
    ).generate(text)
    
    return wordcloud

# Plot timeline chart (with moving average)
def plot_timeline(timeline_df):
    timeline_df['date'] = pd.to_datetime(timeline_df['date'])
    timeline_df['value'] = pd.to_numeric(timeline_df['value'])
    timeline_df['moving_avg'] = timeline_df['value'].rolling(window=7).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(timeline_df['date'], timeline_df['moving_avg'], label='7-Day Moving Average', color='orange')
    plt.fill_between(timeline_df['date'], timeline_df['value'].min(), timeline_df['value'], color='blue', alpha=0.3)
    plt.title('Timeline Volume with 7-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Volume (Percentage of Global Coverage)')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    st.pyplot(plt)

# Plot tone chart
def plot_tone_chart(tone_df):
    tone_df['bin'] = pd.to_numeric(tone_df['bin'])
    tone_df['count'] = pd.to_numeric(tone_df['count'])

    colors = ['red' if x < 0 else 'gray' if x == 0 else 'green' for x in tone_df['bin']]
    
    total_articles = tone_df['count'].sum()
    negative_articles = tone_df[tone_df['bin'] < 0]['count'].sum()
    positive_articles = tone_df[tone_df['bin'] > 0]['count'].sum()
    neutral_articles = tone_df[tone_df['bin'] == 0]['count'].sum()

    negative_percentage = (negative_articles / total_articles) * 100
    positive_percentage = (positive_articles / total_articles) * 100
    neutral_percentage = (neutral_articles / total_articles) * 100

    plt.figure(figsize=(10, 6))
    plt.bar(tone_df['bin'], tone_df['count'], color=colors)
    plt.title(f'Tone Distribution (Negative: {negative_percentage:.2f}%, Neutral: {neutral_percentage:.2f}%, Positive: {positive_percentage:.2f}%)')
    plt.xlabel('Tone (Negative to Positive)')
    plt.ylabel('Article Count')
    plt.grid(True)
    st.pyplot(plt)

# App Logic
if search_button:
    with st.spinner("Retrieving news articles, please wait..."):
        articles_df, start_date, end_date = query_gdelt_data(keyword, lookback_period, limit_to_english)
    
    if not articles_df.empty:
        # First Row: Summary
        with st.container(border = True):
            st.markdown("<div class='container'><h3>🗑 Summary of Articles</h3></div>", unsafe_allow_html=True)
            articles_list = [f"Title: {row['title']}, URL: {row['url']}" for _, row in articles_df.iterrows()]
            articles_text = "\n".join(articles_list)
        
            prompt = (
                f"Summarize significant events related to '{keyword}' from {start_date} to {end_date}."
                f" Here are the top articles: {articles_text}"
            )
        
            try:
                response = model.generate_content(prompt)
                summary = response.text
            except Exception as e:
                st.error("An error occurred during summarization.")
                summary = "Summary not available due to an error."
            
            st.markdown(f"<div class='summary'>{summary}</div>", unsafe_allow_html=True)

        # Second Row: Word Cloud
        with st.container(border = True):
            st.markdown("<div class='container'><h3>☁️ Word Cloud of Headlines</h3></div>", unsafe_allow_html=True)
            if 'title' in articles_df.columns:
                wordcloud = generate_wordcloud(" ".join(articles_df['title']))
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

        # Third Row: Timeline
        with st.container(border = True):
            st.markdown("<div class='container'><h3>📊 Timeline Chart</h3></div>", unsafe_allow_html=True)
            # Generate fake timeline data (replace with actual API query)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            timeline_data = pd.DataFrame({"date": dates, "value": np.random.randint(0, 100, size=len(dates))})
            plot_timeline(timeline_data)
        
        # Fourth Row: Tone Chart
        with st.container(border = True):
            st.markdown("<div class='container'><h3>📉 Tone Chart</h3></div>", unsafe_allow_html=True)
            # Generate fake tone chart data (replace with actual API query)
            tone_data = pd.DataFrame({"bin": range(-10, 11), "count": np.random.randint(1, 100, size=21)})
            plot_tone_chart(tone_data)

        # Fifth Row: Article Headlines
        with st.container(border = True):
            st.markdown("<div class='container'><h3>📰 Article Headlines</h3></div>", unsafe_allow_html=True)
            for index, row in articles_df.iterrows():
                st.markdown(f"[{row['title']}]({row['url']})", unsafe_allow_html=True)

    else:
        st.warning("🤔 No articles found for the given search parameters.")