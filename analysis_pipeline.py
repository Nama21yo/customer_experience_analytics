import pandas as pd
from google_play_scraper import Sort, reviews_all
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import logging
import time
from sqlalchemy import create_engine
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Download NLTK stopwords if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Constants ----------------------------------------------------------------

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

# --- Functions ----------------------------------------------------------------

def scrape_reviews(app_ids: dict, review_count: int) -> pd.DataFrame:
    all_reviews = []
    for bank_name, app_id in app_ids.items():
        logging.info(f"Scraping reviews for {bank_name} ({app_id})...")
        try:
            results = reviews_all(
                app_id,
                sleep_milliseconds=0,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
            )
            sliced_results = results[:review_count]
            for review in sliced_results:
                review['bank'] = bank_name
                review['source'] = 'Google Play'
            all_reviews.extend(sliced_results)
            logging.info(f"Scraped {len(sliced_results)} reviews for {bank_name}.")
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error scraping {bank_name}: {e}")
    return pd.DataFrame(all_reviews)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting data preprocessing...")
    df = df[['content', 'score', 'at', 'bank', 'source']].rename(columns={
        'content': 'review',
        'score': 'rating',
        'at': 'date'
    })
    initial = len(df)
    df.dropna(subset=['review'], inplace=True)
    df = df[df['review'].str.strip() != '']
    logging.info(f"Removed {initial - len(df)} rows with empty reviews.")
    df.drop_duplicates(subset=['review', 'rating', 'bank'], inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Preprocessing complete. Final dataset: {len(df)} rows.")
    return df

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Performing sentiment analysis...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    sentiments = sentiment_pipeline(df['review'].tolist())
    df['sentiment_label_model'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    def adjust_label(row):
        if row['rating'] <= 2:
            return 'NEGATIVE'
        elif row['rating'] == 3:
            return 'NEUTRAL'
        return 'POSITIVE'
    df['sentiment_label'] = df.apply(adjust_label, axis=1)
    logging.info("Sentiment analysis complete.")
    return df

def identify_themes(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Identifying themes...")
    def get_theme(text: str) -> str:
        review = text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            if any(k in review for k in keywords):
                return theme
        return 'General Feedback'
    df['theme'] = df['review'].apply(get_theme)
    logging.info("Theme identification complete.")
    return df

def store_data_in_oracle(df: pd.DataFrame, table_name: str):
    logging.info(f"Preparing to store data in Oracle table: {table_name}...")
    db_user = os.getenv("DB_USER", "your_user")
    db_password = os.getenv("DB_PASSWORD", "your_password")
    db_dsn = os.getenv("DB_DSN", "localhost:1521/XEPDB1")
    if db_user == "your_user" or db_password == "your_password":
        logging.warning("Database credentials not set. Skipping Oracle DB storage.")
        return
    try:
        engine = create_engine(f"oracle+oracledb://{db_user}:{db_password}@{db_dsn}")
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=100)
        logging.info(f"Successfully wrote {len(df)} rows to Oracle table '{table_name}'.")
    except Exception as e:
        logging.error(f"Failed to store data in Oracle. Error: {e}")

# --- Main Execution ----------------------------------------------------------

if __name__ == "__main__":
    logging.info("Starting Task 1: Data Collection & Preprocessing")
    raw_reviews = scrape_reviews(BANK_APP_IDS, review_count=450)
    if raw_reviews.empty:
        logging.warning("No data scraped. Exiting.")
        exit()

    clean_reviews = preprocess_data(raw_reviews)
    task1_output_path = "task_1_clean_reviews.csv"
    clean_reviews.to_csv(task1_output_path, index=False, encoding='utf-8')
    logging.info(f"Task 1 output saved to: {task1_output_path}")

    logging.info("Starting Task 2: Sentiment and Thematic Analysis")
    sentiment_df = analyze_sentiment(clean_reviews)
    final_df = identify_themes(sentiment_df)
    task2_output_path = "task_2_analyzed_reviews.csv"
    final_columns = ['bank', 'date', 'rating', 'review', 'sentiment_label', 'sentiment_score', 'theme']
    final_df[final_columns].to_csv(task2_output_path, index=False, encoding='utf-8')
    logging.info(f"Task 2 output saved to: {task2_output_path}")

    logging.info("Starting Task 3: Storing in Oracle DB")
    store_data_in_oracle(final_df[final_columns], table_name="BANK_APP_REVIEWS")

    logging.info("All tasks completed successfully.")
