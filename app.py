import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import google.generativeai as genai
import numpy as np
from matplotlib.dates import DateFormatter
from googletrans import Translator

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def main():
    # Page configuration
    st.set_page_config(page_title="Global News Explorer", page_icon="📰", layout="wide")
    apply_custom_styling()

    # Adding a header and a description for the app
    st.markdown("<h1>📰 Global News Explorer 🌍</h1>", unsafe_allow_html=True)
    st.markdown("""
        Welcome to the Global News Explorer. This application allows you to search global news articles on specific topics, visualize key trends, 
        and gain insights into significant events worldwide. Simply enter your search criteria and explore the results with intuitive visualizations.
    """)

    # 'About this app' dropdown with details about the tech stack
    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown("""
        **Global News Explorer** was developed using:
        - **Streamlit** for the web interface
        - **GDELT API** for retrieving global news data
        - **Matplotlib** and **WordCloud** for data visualization
        - **Google Generative AI** for automatic text summarization
        - **Googletrans** for translation services
        
        This app was built to help users easily explore news articles and visualize trends in the global media landscape. 
        ChatGPT-4 was used as a research assistant in developing and enhancing the app's features.
        """)

    # Sidebar input
    with st.sidebar:
        st.header("🔍 Search Parameters")
        keyword_input, lookback_period = get_user_input()
        search_button = st.button("Search Articles")

    # Configure Generative AI API
    genai.configure(api_key=st.secrets["api_key"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Process search on button click
    if search_button:
        if len(keyword_input) < 5:
            st.warning("🔍 Please enter at least 5 characters for the keyword.")
        else:
            query = build_query(keyword_input)
            st.spinner("Retrieving news articles, please wait...")

            start_datetime, end_datetime = get_start_date(lookback_period)
            articles_df, timeline_df, tone_df = aggregate_gdelt_data(query, start_datetime, end_datetime)

            # Descriptive text about the results
            st.markdown(f"### Results for '{keyword_input.replace(';', ' OR ')}'")
            st.markdown("Explore the news articles retrieved below, along with insights into the timeline and overall sentiment.")

            # Display results if available
            if not articles_df.empty:
                display_summary(model, query, start_datetime.strftime('%Y-%m-%d'), end_datetime.strftime('%Y-%m-%d'), articles_df)
                display_wordcloud(articles_df)
                display_timeline(timeline_df)
                display_tone_chart(tone_df)
                display_article_headlines(articles_df)
            else:
                st.warning("🤔 No articles found for the given search parameters.")

def ensure_keyword_in_quotes(keyword):
    """
    Ensure each keyword or keyphrase is enclosed in double quotes unless it's already quoted.
    """
    keyword = keyword.strip()
    if not (keyword.startswith('"') and keyword.endswith('"')):
        keyword = f'"{keyword}"'
    return keyword

def get_start_date(lookback_period):
    end_datetime = datetime.now()
    if lookback_period == '1 week':
        start_datetime = end_datetime - timedelta(weeks=1)
    elif lookback_period == '1 month':
        start_datetime = end_datetime - timedelta(days=30)
    elif lookback_period == '3 months':
        start_datetime = end_datetime - timedelta(days=90)
    elif lookback_period == '6 months':
        start_datetime = end_datetime - timedelta(days=180)
    elif lookback_period == '1 year':
        start_datetime = end_datetime - timedelta(days=365)
    return start_datetime, end_datetime

def apply_custom_styling():
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
    </style>
    """, unsafe_allow_html=True)


def get_user_input():
    keyword_input = st.text_input("Enter Keywords (semicolon-separated)", "Climate Change; Global Warming; Greenhouse Effect")
    lookback_options = ["1 week", "1 month", "3 months", "6 months", "1 year"]
    lookback_period = st.selectbox("Choose Lookback Period", lookback_options, index=1)
    return keyword_input, lookback_period


def query_gdelt_data(query, mode, start_datetime=None, end_datetime=None):
    params = {
        'mode': mode,
        'format': 'json',
        'maxrecords': 250,
    }

    if start_datetime:
        params['STARTDATETIME'] = start_datetime.strftime('%Y%m%d%H%M%S')
    if end_datetime:
        params['ENDDATETIME'] = end_datetime.strftime('%Y%m%d%H%M%S')

    request_url = f"{BASE_URL}?query={query}&mode={mode}&format=json&maxrecords=250"
    if start_datetime:
        request_url += f"&STARTDATETIME={params['STARTDATETIME']}"
    if end_datetime:
        request_url += f"&ENDDATETIME={params['ENDDATETIME']}"

    response = requests.get(request_url)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def aggregate_gdelt_data(query, start_datetime, end_datetime):
    articles_df, timeline_df, tone_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    timeline_data = query_gdelt_data(query, 'timelinevol', start_datetime, end_datetime)
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data['timeline'][0]['data'])
        timeline_df['date'] = pd.to_datetime(timeline_df['date'])
        timeline_df['value'] = pd.to_numeric(timeline_df['value'])
        timeline_df['moving_avg'] = timeline_df['value'].rolling(window=7).mean()

    tone_data = query_gdelt_data(query, 'tonechart', start_datetime, end_datetime)
    if tone_data:
        tone_df = pd.DataFrame(tone_data['tonechart'])

    articles_data = query_gdelt_data(query, 'artlist', start_datetime, end_datetime)
    if articles_data:
        articles_df = pd.DataFrame(articles_data['articles'])

    return articles_df, timeline_df, tone_df


def display_summary(model, keyword_input, start_date, end_date, articles_df):
    st.markdown("<div class='container'><h3>🗑 Summary of Articles</h3></div>", unsafe_allow_html=True)
    articles_list = [f"Title: {row['title']}, URL: {row['url']}" for _, row in articles_df.iterrows()]
    articles_text = "\n".join(articles_list)

    prompt = (
        f"Summarize significant events related to the search for '{keyword_input.replace(';', ' OR ')}' "
        f"from {start_date} to {end_date}. Here are the top articles: {articles_text}. "
    )

    try:
        response = model.generate_content(prompt)
        summary = response.text
    except Exception as e:
        st.error("An error occurred during summarization.")
        summary = "Summary not available due to an error."
    
    st.markdown(f"<div class='summary'>{summary}</div>", unsafe_allow_html=True)


def display_wordcloud(articles_df):
    st.markdown("<div class='container'><h3>☁️ Word Cloud of Headlines</h3></div>", unsafe_allow_html=True)
    if 'title' in articles_df.columns:
        wordcloud = generate_wordcloud(" ".join(articles_df['title']))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


def generate_wordcloud(text):
    translator = Translator()
    try:
        translated_text = translator.translate(text, dest='en').text
    except Exception as e:
        translated_text = text

    filtered_words = [word for word in translated_text.split() if len(word) > 2]
    cleaned_text = " ".join(filtered_words)

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stopwords,
        collocations=False
    ).generate(cleaned_text)
    
    return wordcloud

def build_query(keyword_input):
    """
    Build the full query string from the user input.
    If the user provides multiple keywords separated by semicolons, 
    parentheses are added only around OR'd terms.
    """
    keywords = keyword_input.split(";")
    
    # If there are multiple keywords, apply parentheses and OR logic
    if len(keywords) > 1:
        keywords = [ensure_keyword_in_quotes(kw.strip()) for kw in keywords]
        query = " OR ".join(keywords)
        return f"({query})"  # Wrap the OR'd terms in parentheses
    else:
        # Single keyword: no parentheses needed
        return ensure_keyword_in_quotes(keywords[0].strip())



def display_timeline(timeline_df):
    
    st.markdown("<div class='container'><h3>📊 Timeline Chart</h3></div>", unsafe_allow_html=True)
    with st.container(border = True):
        if not timeline_df.empty:
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

def display_tone_chart(tone_df):
    
    st.markdown("<div class='container'><h3>📉 Tone Chart</h3></div>", unsafe_allow_html=True)
    with st.container(border = True):
        if not tone_df.empty:
            tone_df['bin'] = pd.to_numeric(tone_df['bin'])
            tone_df['count'] = pd.to_numeric(tone_df['count'])
            colors = ['red' if x < 0 else 'gray' if x == 0 else 'green' for x in tone_df['bin']]

            plt.figure(figsize=(10, 6))
            plt.bar(tone_df['bin'], tone_df['count'], color=colors)
            plt.xlabel('Tone (Negative to Positive)')
            plt.ylabel('Article Count')
            plt.grid(True)
            st.pyplot(plt)

def display_article_headlines(articles_df):
    
    st.markdown("<div class='container'><h3>📰 Article Headlines</h3></div>", unsafe_allow_html=True)
    with st.container(border = True):
        for index, row in articles_df.iterrows():
            st.markdown(f"[{row['title']}]({row['url']})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
