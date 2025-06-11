import pandas as pd
import os
import time
import logging
import nltk

from google_play_scraper import Sort, reviews_all
from transformers import pipeline
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ---------------- NLTK Setup ----------------
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# ---------------- Constants ----------------
BANK_APP_IDS = {
    'Commercial Bank of Ethiopia': 'com.cbe.mobilebanking',
    'Bank of Abyssinia': 'com.boa.boadigital',
    'Dashen Bank': 'com.dashen.dashensuperapp'
}

THEME_KEYWORDS = {
    'Account Access': ['login', 'password', 'signin', 'register', 'otp', 'authentication', 'lock', 'block'],
    'Transaction & Performance': ['slow', 'crash', 'fail', 'stuck', 'transfer', 'payment', 'transaction', 'error', 'loading', 'speed'],
    'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'simple', 'confusing', 'dark mode', 'update', 'look', 'feel'],
    'Customer Support': ['support', 'help', 'contact', 'call center', 'response', 'agent', 'service'],
    'Feature Request & Functionality': ['feature', 'add', 'need', 'fingerprint', 'biometric', 'statement', 'limit', 'option', 'suggestion']
}

# ---------------- Tasks ----------------

def scrape_reviews(app_ids: dict, review_count: int) -> pd.DataFrame:
    all_reviews = []
    for bank_name, app_id in app_ids.items():
        logging.info(f"Scraping reviews for {bank_name}...")
        try:
            results = reviews_all(app_id, sleep_milliseconds=0, lang='en', country='us', sort=Sort.NEWEST)
            sliced_results = results[:review_count]
            for review in sliced_results:
                review['bank'] = bank_name
                review['source'] = 'Google Play'
            all_reviews.extend(sliced_results)
            logging.info(f"Collected {len(sliced_results)} reviews for {bank_name}")
            time.sleep(2)
        except Exception as e:
            logging.error(f"Failed to scrape {bank_name}: {e}")
    return pd.DataFrame(all_reviews)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preprocessing data...")
    df = df[['content', 'score', 'at', 'bank', 'source']].rename(columns={
        'content': 'review', 'score': 'rating', 'at': 'date'
    })
    initial_rows = len(df)
    df.dropna(subset=['review'], inplace=True)
    df = df[df['review'].str.strip() != '']
    logging.info(f"Removed {initial_rows - len(df)} empty reviews.")
    df.drop_duplicates(subset=['review', 'rating', 'bank'], inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Final dataset size: {len(df)}")
    return df

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Analyzing sentiment...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = sentiment_pipeline(df['review'].tolist())
    df['sentiment_label_model'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    def adjust_sentiment(row):
        if row['rating'] <= 2:
            return 'NEGATIVE'
        elif row['rating'] == 3:
            return 'NEUTRAL'
        return 'POSITIVE'

    df['sentiment_label'] = df.apply(adjust_sentiment, axis=1)
    logging.info("Sentiment analysis complete.")
    return df

def identify_themes(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Classifying themes...")
    def assign_theme(text):
        text = text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return theme
        return 'General Feedback'
    df['theme'] = df['review'].apply(assign_theme)
    logging.info("Theme classification complete.")
    return df

def store_data_in_oracle(df: pd.DataFrame, table_name: str):
    logging.info(f"Saving data to Oracle DB table '{table_name}'...")
    db_user = os.getenv("DB_USER", "your_user")
    db_password = os.getenv("DB_PASSWORD", "your_password")
    db_dsn = os.getenv("DB_DSN", "localhost:1521/XEPDB1")

    if "your_user" in db_user or "your_password" in db_password:
        logging.warning("Default DB credentials detected. Skipping DB storage.")
        return

    try:
        engine = create_engine(f"oracle+oracledb://{db_user}:{db_password}@{db_dsn}")
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=100)
        logging.info(f"Saved {len(df)} rows to table '{table_name}'.")
    except Exception as e:
        logging.error(f"Failed to write to Oracle DB: {e}")

def create_visualizations(df: pd.DataFrame, output_dir: str = "output_visuals"):
    logging.info("Creating visualizations...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory '{output_dir}'.")

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='bank', hue='sentiment_label', palette={'POSITIVE': 'green', 'NEUTRAL': 'grey', 'NEGATIVE': 'red'})
    plt.title('Sentiment Distribution by Bank')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_sentiment_distribution.png'))
    plt.close()

    negative_reviews = df[df['sentiment_label'] == 'NEGATIVE']
    plt.figure(figsize=(10, 6))
    sns.countplot(data=negative_reviews, y='theme', order=negative_reviews['theme'].value_counts().index, color='salmon')
    plt.title('Most Common Themes in Negative Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_top_negative_themes.png'))
    plt.close()

    avg_rating_theme = df.groupby('theme')['rating'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    avg_rating_theme.plot(kind='barh', color='skyblue')
    plt.title('Average Rating by Theme')
    plt.xlim(1, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_avg_rating_by_theme.png'))
    plt.close()
    logging.info("Visualizations created successfully.")

# ---------------- Main Execution ----------------

if __name__ == "__main__":
    logging.info("Starting analysis pipeline...")

    raw_df = scrape_reviews(BANK_APP_IDS, review_count=450)
    if raw_df.empty:
        logging.warning("No reviews collected. Aborting.")
        exit()

    clean_df = preprocess_data(raw_df)
    clean_df.to_csv("task_1_clean_reviews.csv", index=False)

    analyzed_df = analyze_sentiment(clean_df)
    analyzed_df = identify_themes(analyzed_df)
    analyzed_df.to_csv("task_2_analyzed_reviews.csv", index=False)

    store_data_in_oracle(analyzed_df, table_name="BANK_APP_REVIEWS")

    create_visualizations(analyzed_df)

    logging.info("All tasks completed.")
