# Mobile Banking App Review Analysis for Ethiopian Banks

This project is a data analysis initiative simulating a consulting engagement by **Omega Consultancy**. The goal is to analyze customer satisfaction with the mobile banking apps of three major Ethiopian banks: **Commercial Bank of Ethiopia (CBE)**, **Bank of Abyssinia (BOA)**, and **Dashen Bank**.

The analysis involves scraping user reviews from the Google Play Store, performing Natural Language Processing (NLP) to understand sentiment and recurring themes, and deriving actionable insights to help the banks improve their digital offerings, enhance user retention, and manage customer complaints effectively.

## Business Objective

As a Data Analyst at Omega Consultancy, the mission is to:

1.  **Scrape** user reviews from the Google Play Store to build a relevant dataset.
2.  **Analyze** review sentiment (positive, negative, neutral) and extract key themes (e.g., "UI/UX", "Performance", "Bugs").
3.  **Identify** specific satisfaction drivers and pain points for each bank's app.
4.  **Store** the cleaned and analyzed data in a structured database (future task).
5.  **Deliver** a report with visualizations and data-driven recommendations for app improvement.

## Tech Stack

- **Language:** Python 3.9+
- **Data Scraping:** `google-play-scraper`
- **Data Manipulation:** `pandas`
- **NLP & Analysis:**
  - `transformers` (Hugging Face) for sentiment analysis (`distilbert-base-uncased-finetuned-sst-2-english`).
  - `nltk` for text preprocessing (e.g., stop-word removal).
  - `scikit-learn` for potential keyword extraction via TF-IDF (the current implementation uses a rule-based approach).
- **Version Control:** `Git` & `GitHub`

## Project Structure

```
.
├── .gitignore               # Specifies files for Git to ignore
├── README.md                # This file
├── analysis_pipeline.py     # Main script for scraping, preprocessing, and analysis
├── requirements.txt         # Required Python packages for the project
│
└── output/                  # (Generated after running the script)
    ├── task_1_clean_reviews.csv      # Deliverable for Task 1
    └── task_2_analyzed_reviews.csv   # Deliverable for Task 2
```

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/mobile-banking-analysis.git
    cd mobile-banking-analysis
    ```

2.  **Create a virtual environment (recommended):**

    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## How to Run the Pipeline

Execute the main analysis script from your terminal. The script will perform all tasks sequentially and save the output CSV files in the root directory.

```sh
python analysis_pipeline.py
```

Upon successful execution, you will see logs in your console detailing the progress of each step and two new files will be created: `task_1_clean_reviews.csv` and `task_2_analyzed_reviews.csv`.

## Data Pipeline Overview

The `analysis_pipeline.py` script automates the end-to-end process from data collection to analysis.

### Task 1: Data Collection & Preprocessing

1.  **Web Scraping:** The script uses `google-play-scraper` to collect ~400 recent reviews for each of the three bank apps.
2.  **Data Cleaning:** A multi-step preprocessing function ensures data quality:
    - **Removes Duplicates:** Ensures each review is unique.
    - **Handles Missing Data:** Drops rows where the review text is missing.
    - **Normalizes Dates:** Converts all date formats into a consistent `YYYY-MM-DD` format.
    - **Standardizes Schema:** Renames columns to `review`, `rating`, `date`, `bank`, and `source`.
3.  **Output:** A clean dataset is saved as `task_1_clean_reviews.csv`.

### Task 2: Sentiment and Thematic Analysis

1.  **Sentiment Analysis:**

    - The pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face is used to assign a `POSITIVE` or `NEGATIVE` label and a confidence score.
    - A business rule is applied to create a more nuanced three-tier sentiment (`POSITIVE`, `NEUTRAL`, `NEGATIVE`) based on the star rating (1-2 stars: Negative, 3: Neutral, 4-5: Positive).

2.  **Thematic Analysis:**

    - A rule-based approach is used to categorize each review into a predefined theme.
    - Keywords and common phrases are grouped into 5 overarching themes:
      - `Account Access` (login, password, OTP)
      - `Transaction & Performance` (slow, crash, error, transfer)
      - `User Interface & Experience` (UI, design, easy, update)
      - `Customer Support` (support, help, call center)
      - `Feature Request & Functionality` (feature, fingerprint, statement)
    - Reviews that do not match any theme are labeled as `General Feedback`.

3.  **Output:** The fully analyzed dataset is saved as `task_2_analyzed_reviews.csv`.

## Git Workflow

This project follows a feature-branching workflow to maintain a clean and stable `main` branch.

1.  **Task 1 Branch (`task-1`):**

    - All development for data scraping and preprocessing was done on this branch.
    - Commits were made incrementally (e.g., `"feat: Implement scraping function"`, `"fix: Handle duplicate reviews"`).
    - Once complete, the branch was merged into `main` via a pull request.

2.  **Task 2 Branch (`task-2`):**
    - Development for sentiment and thematic analysis was isolated to this branch.
    - Commits followed a similar logical pattern.
    - The branch was merged into `main` after the analysis pipeline was successfully validated.

This approach ensures that the `main` branch always contains a working version of the project at the end of each major task.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
