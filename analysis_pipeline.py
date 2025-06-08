import pandas as pd
from google_play_scraper import Sort, reviews_all
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import logging
import time

# Intial Setup

# Configure logging to monitor the script's progress and any issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)

#  necessary NLTK data for text processing
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Constants ----------------------------------------------------------------

# Dictionary of Bank App IDs on the Google Play Store
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

# --- Data Collection & Preprocessing ----------------------------------

def scrape_reviews(app_ids: dict, review_count: int) -> pd.DataFrame:
    """
    Scrapes a specified number of reviews for each app in the dictionary.

    Args:
        app_ids (dict): A dictionary mapping bank names to their app IDs.
        review_count (int): The number of reviews to scrape per app.

    Returns:
        pd.DataFrame: A DataFrame containing raw review data.
    """
    all_reviews = []
    for bank_name, app_id in app_ids.items():
        logging.info(f"Starting to scrape reviews for {bank_name} ({app_id})...")
        try:
            # `reviews_all` fetches all available reviews. We'll slice it.
            # Using `reviews` with a count is also a good option.
            results = reviews_all(
                app_id,
                sleep_milliseconds=0,  # No delay between requests
                lang='en',             # English reviews
                country='us',          # US region
                sort=Sort.NEWEST,      # Get the most recent reviews
            )
            
            # Slice the results to get the desired number of reviews
            sliced_results = results[:review_count]

            for review in sliced_results:
                review['bank'] = bank_name
                review['source'] = 'Google Play'
            all_reviews.extend(sliced_results)
            logging.info(f" Scraped {len(sliced_results)} reviews for {bank_name}.")
            time.sleep(2) # A small polite delay between scraping different apps
        except Exception as e:
            logging.error(f"Could not scrape reviews for {bank_name}. Error: {e}")
    
    return pd.DataFrame(all_reviews)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the raw review data.

    Args:
        df (pd.DataFrame): The raw review DataFrame.

    Returns:
        pd.DataFrame: A cleaned and structured DataFrame.
    """
    logging.info("🧹 Starting data preprocessing...")
    
    # 1. Select and rename columns
    df = df[['content', 'score', 'at', 'bank', 'source']].rename(columns={
        'content': 'review',
        'score': 'rating',
        'at': 'date'
    })

    # 2. Handle missing data (especially empty reviews)
    initial_rows = len(df)
    df.dropna(subset=['review'], inplace=True)
    df = df[df['review'].str.strip() != '']
    rows_after_na = len(df)
    logging.info(f"Dropped {initial_rows - rows_after_na} rows with missing review text.")

    # 3. Remove duplicate reviews
    initial_rows = len(df)
    df.drop_duplicates(subset=['review', 'rating', 'bank'], inplace=True, keep='first')
    rows_after_duplicates = len(df)
    logging.info(f"Dropped {initial_rows - rows_after_duplicates} duplicate reviews.")

    # 4. Normalize dates to YYYY-MM-DD format
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # 5. Reset index for a clean DataFrame
    df.reset_index(drop=True, inplace=True)
    
    logging.info(f"Preprocessing complete. Final dataset has {len(df)} rows.")
    return df

# --- Task 2: Sentiment and Thematic Analysis ----------------------------------

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs sentiment analysis on the review text.

    Args:
        df (pd.DataFrame): The cleaned review DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added sentiment label and score.
    """
    logging.info("Performing sentiment analysis...")
    
    # Load a pre-trained sentiment analysis model
    # This model is great for general sentiment but is binary (POSITIVE/NEGATIVE)
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Apply the pipeline to the review text
    # Note: For very large datasets, process in batches. For 1200 reviews, this is fine.
    sentiments = sentiment_pipeline(df['review'].tolist())
    
    # Add results to the DataFrame
    df['sentiment_label_model'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    # Business Rule: Adjust labels based on rating for a more nuanced classification
    # 1-2 stars -> Negative, 3 -> Neutral, 4-5 -> Positive
    def adjust_sentiment(row):
        if row['rating'] <= 2:
            return 'NEGATIVE'
        elif row['rating'] == 3:
            return 'NEUTRAL'
        else: # 4 or 5
            return 'POSITIVE'

    df['sentiment_label'] = df.apply(adjust_sentiment, axis=1)
    
    logging.info("Sentiment analysis complete.")
    return df

def identify_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies themes in reviews using a keyword-based approach.

    Args:
        df (pd.DataFrame): DataFrame with review text.

    Returns:
        pd.DataFrame: DataFrame with an added 'theme' column.
    """
    logging.info("Identifying review themes...")

    def assign_theme(review_text: str) -> str:
        """Helper function to assign a theme based on keyword matching."""
        review_lower = review_text.lower()
        for theme, keywords in THEME_KEYWORDS.items():
            if any(keyword in review_lower for keyword in keywords):
                return theme
        return 'General Feedback' # Default theme if no keywords match

    df['theme'] = df['review'].apply(assign_theme)
    logging.info("Thematic analysis complete.")
    return df

# --- Main Execution Block ----------------------------------------------------

if __name__ == "__main__":
    # --- Execute Task 1 ---
    logging.info("--- Starting Task 1: Data Collection & Preprocessing ---")
    raw_reviews_df = scrape_reviews(BANK_APP_IDS, review_count=450) # Scrape a bit more to account for cleaning
    
    if not raw_reviews_df.empty:
        clean_reviews_df = preprocess_data(raw_reviews_df)
        
        # Save the Task 1 deliverable
        task1_output_path = "task_1_clean_reviews.csv"
        clean_reviews_df.to_csv(task1_output_path, index=False, encoding='utf-8')
        logging.info(f"💾 Task 1 output saved to: {task1_output_path}")
        print("\n--- Task 1 Final DataFrame Head ---")
        print(clean_reviews_df.head())
        print(f"\nValue counts per bank:\n{clean_reviews_df['bank'].value_counts()}")

        # ---  Task 2 ---
        logging.info("\n--- Starting Task 2: Sentiment & Thematic Analysis ---")
        sentiment_df = analyze_sentiment(clean_reviews_df)
        final_analyzed_df = identify_themes(sentiment_df)
        
        # Save the Task 2 deliverable
        task2_output_path = "task_2_analyzed_reviews.csv"
        # Reordering columns for clarity in the final output
        final_columns = [
            'bank', 'date', 'rating', 'review', 
            'sentiment_label', 'sentiment_score', 'theme'
        ]
        final_analyzed_df[final_columns].to_csv(task2_output_path, index=False, encoding='utf-8')
        logging.info(f"💾 Task 2 output saved to: {task2_output_path}")
        print("\n--- Task 2 Final DataFrame Head ---")
        print(final_analyzed_df[final_columns].head())
        print(f"\nTheme distribution:\n{final_analyzed_df['theme'].value_counts()}")
        
    else:
        logging.warning("Scraping returned no data. Halting execution.")

    logging.info("\n All tasks completed successfully!")
