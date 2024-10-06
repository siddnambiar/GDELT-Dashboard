import streamlit as st
from gdeltdoc import GdeltDoc, Filters
import pandas as pd
import random
from datetime import datetime
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

st.set_page_config(page_title="Global News Explorer", page_icon="📰")

st.markdown("""
    <style>
    .reportview-container {
        background-color: #FAFAFA;
        color: #333;
        font-family: "Arial", sans-serif;
    }
    .stButton>button {
        background-color: #007BFF;
        color: #FFF;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-size: 1em;
    }
    .stButton>button:hover {
        background: #0056b3;
        transform: scale(1.02);
    }
    .summary-container {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

title_html = "<h1 style='text-align: center; font-size: 3em;'>📰 Global News Explorer 🌍</h1>"
st.markdown(title_html, unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Discover how past events were portrayed in the news during specific months</h3>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; font-size: 1.2em; margin-bottom: 20px;'>
        Welcome to Global News Explorer! This app allows you to search and analyze news articles from around the world, providing insights and visual summaries. Note that since this is a prototype, it only allows you to go back until 2020. Feel free to use this app to explore how quirky, weird, and niche events were portrayed in the news at the time.
        Simply enter a keyword, select a date range, and click 'Search Articles' to get a summary of recent news and a word cloud visualization.
        Here are some fun, quirky keywords you could try: 'UFO sightings', 'unusual animal behavior', 'bizarre inventions', 'strange coincidences', 'mystery events'.
    </div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.header("🔍 Search Parameters")
    st.write("Use the filters below to specify your search criteria.")

    keyword = st.text_input(
        "Keyword", "world news",
        help="Enter keywords to search for articles. Use quotes for exact phrases, 'OR' for alternatives, and '-' to exclude terms."
    )

    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    years = list(range(2020, current_year + 1))[::-1]
    year = st.selectbox("Select Year", years, index=0)

    months_full = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    if year == current_year:
        months = months_full[:current_month]
    else:
        months = months_full

    month = st.selectbox("Select Month", months, index=len(months)-1)

    quarters = ["None"]
    if year < current_year:
        quarters.extend(["Q1", "Q2", "Q3", "Q4"])
    else:
        if current_month >= 3:
            quarters.append("Q1")
        if current_month >= 6:
            quarters.append("Q2")
        if current_month >= 9:
            quarters.append("Q3")
        if current_month >= 12:
            quarters.append("Q4")

    quarter = st.selectbox("Select Quarter (Optional)", quarters, index=0)

st.markdown("### 🔎 Ready to Search?")
search_button = st.button("Search Articles")

start_date = f"{year}-01-01"
end_date = f"{year}-12-31"
if quarter != "None":
    if quarter == "Q1":
        start_date = f"{year}-01-01"
        end_date = f"{year}-03-31"
    elif quarter == "Q2":
        start_date = f"{year}-04-01"
        end_date = f"{year}-06-30"
    elif quarter == "Q3":
        start_date = f"{year}-07-01"
        end_date = f"{year}-09-30"
    elif quarter == "Q4":
        start_date = f"{year}-10-01"
        end_date = f"{year}-12-31"
else:
    month_number = months_full.index(month) + 1
    start_date = f"{year}-{month_number:02d}-01"
    if month in ["April", "June", "September", "November"]:
        end_day = 30
    elif month == "February":
        end_day = 29 if (int(year) % 4 == 0) else 28
    else:
        end_day = 31
    end_date = f"{year}-{month_number:02d}-{end_day}"

current_date_str = current_date.strftime("%Y-%m-%d")
if end_date > current_date_str:
    end_date = current_date_str

# Guidance for using AND vs OR in the keyword query
st.markdown("""
    **Keyword Guidance:**
    - Use quotes (`"..."`) to search for exact phrases. For example: `"climate change"`
    - Use `OR` to search for articles that mention any of the specified terms. For example: `(economy OR finance OR stocks)`
    - Use `-` to exclude terms. For example: `technology -smartphones` (articles about technology excluding smartphones)
""")

filters = Filters(
    keyword=keyword,
    start_date=start_date,
    end_date=end_date
)

gd = GdeltDoc()

with open('Data/api_key', 'r') as key_file:
    api_key = key_file.readline().strip()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

with st.container(border=True):
    if search_button:
        with st.spinner(random.choice([
            "🌍 Searching the world...", "🔎 Finding news nuggets...",
            "📰 Scouring the headlines...", "🖼️ Gathering the latest scoops..."
        ])):
            articles = gd.article_search(filters)
            if not articles.empty:
                articles_df = pd.DataFrame(articles)

                if 'language' in articles_df.columns:
                    articles_df = articles_df[articles_df['language'] == 'English']

                articles_list = []
                for _, row in articles_df.iterrows():
                    articles_list.append(f"Title: {row['title']}, URL: {row['url']}")
                articles_text = "\n".join(articles_list)

                prompt = (
                    f"You are an expert news analyst. Provide an objective, concise summary of the most significant events related to '{keyword}' "
                    f"during the period from {start_date} to {end_date}. Focus on recent developments with substantial impact. "
                    "Organize the summary with clear headings for each major theme, and include specific details such as dates, names, statistics, "
                    "and significant quotes to support each point. Write in a neutral, professional tone appropriate for a news summary. "
                    "Exclude minor or unrelated topics to maintain focus. Your summary should be detailed and comprehensive, covering all major themes without a strict word limit.\n\n"
                    "Here are the top articles:\n"
                    f"{articles_text}"
                )

                try:
                    response = model.generate_content(prompt)
                    summary = response.text
                except Exception as e:
                    st.error("An error occurred during summarization.")
                    summary = "Summary not available due to an error."

                st.markdown("## 🗝 Summary of Articles")
                st.markdown(f"{summary}")

                if 'title' in articles_df.columns:
                    st.markdown("## ☁️ Word Cloud of Headlines")
                    text = " ".join(title for title in articles_df['title'])
                    stopwords = set(STOPWORDS)
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        stopwords=stopwords,
                        colormap='viridis',
                        collocations=False
                    ).generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
            else:
                st.warning("😕 No articles found for the given search parameters.")

with st.expander("🤔 About Me"):
    st.markdown(
        "Hi, I'm Sidd Nambiar! I'm passionate about Data Science and Science Communication. I built this toy prototype to explore the possibilities of using AI and data visualization to make sense of global events. I hope this app provides an engaging way to interact with the news!"
    )

st.markdown(f"""
    <footer style="text-align: center; padding: 20px; font-size: 0.9em; background: #F1F1F1; margin-top: 20px;">
        📰 Global News Explorer - Empowering global insights | Built with Streamlit & Plotly ❤️<br>
        Developed by <a href="https://www.linkedin.com/in/siddnambiar/" target="_blank">Sidd Nambiar</a>. Connect with me on <a href="https://www.linkedin.com/in/siddnambiar/" target="_blank">LinkedIn</a>.
    </footer>
""", unsafe_allow_html=True)
