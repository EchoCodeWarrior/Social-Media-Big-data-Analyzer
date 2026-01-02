# Social-Media-Big-data-Analyzer


# 🐦 X-Ray Analytics Engine

**A Streamlit-first Data Engineering & NLP application for real-time X (Twitter) sentiment analysis.**

This application provides a secure, blocking UI workflow that fetches data via the **X API v2**, processes text using **NLTK**, and performs advanced analytics (TF-IDF, Frequency Distribution) on financial and sector-specific trends.

---

## 🚀 Key Features

* **🔐 Secure, Blocking Auth Flow:** The UI prevents any data fetching or processing until a valid X API Bearer Token is authenticated.
* **📡 Smart Data Fetching:**
* Automated pagination to handle large datasets.
* Rate-limit handling to protect API quotas.
* Intelligent stopping criteria (fetches until target word count is met).


* **🧠 Advanced NLP Pipeline:**
* **Cleaning:** Removal of URLs, mentions, emojis, and symbols.
* **Normalization:** Lowercasing and Lemmatization (NLTK).
* **Stopwords:** Filters out generic English stopwords.


* **📊 Analytics:**
* **TF-IDF:** Identifies unique/important terms relative to the dataset.
* **Word Frequency:** Counts top recurring terms.


* **📂 Export:** Download processed analytics as CSV.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **API Wrapper:** [Tweepy](https://www.tweepy.org/)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
* **NLP:** [NLTK](https://www.nltk.org/) (Stopwords, Lemmatizer)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (TfidfVectorizer)

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/x-ray-analytics.git
cd x-ray-analytics

```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

Create a `requirements.txt` file (or use the command below):

```bash
pip install streamlit tweepy pandas nltk scikit-learn

```

---

## 🖥️ Usage

### 1. Run the Application

```bash
streamlit run app.py

```

### 2. Authentication (Step 1)

* The app will launch in your browser.
* You will be greeted by a **Lock Screen**.
* Enter your **X API Bearer Token** (from the [X Developer Portal](https://developer.twitter.com/)).
* *Note: Credentials are stored in session state only and are never saved to disk.*

### 3. Configure Search (Step 2)

Once validated, the dashboard unlocks:

* **Sector:** Choose from Banking, IT, Pharma, etc.
* **Word Target:** Set the minimum amount of text you want to analyze (e.g., 5,000 words).
* **Stock Symbol:** (Optional) Add a specific ticker like `TSLA` or `HDFCBANK`.

### 4. Analyze Results

Click **Start Extraction**. The app will:

1. Fetch tweets in pages.
2. Clean and Lemmatize text in memory.
3. Display **TF-IDF Tables** and **Frequency Charts**.

---

## 📂 Project Structure

```text
x-ray-analytics/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
└── .gitignore            # Ignored files (venv, secrets)

```

---

## 🧠 How It Works (NLP Logic)

1. **Ingestion:** The app constructs a query using Boolean operators (e.g., `(AI OR Cloud OR SaaS) -is:retweet`).
2. **Pre-processing:**
* *Regex:* Removes `http://...`, `@user`, and non-alpha characters.
* *NLTK:* Tokenizes text and converts words to their root form (e.g., "paying" -> "pay").


3. **Vectorization (TF-IDF):**
* Uses `TfidfVectorizer` to calculate the weight of words.
* **Formula:** 
* *Why?* This helps highlight words that are specific to the current market trend rather than just common words like "the" or "is".



---

## ⚠️ Important Notes

* **API Quotas:** This app uses `search_recent_tweets`.
* **Free/Basic Tier:** Very limited access.
* **Pro Tier:** Recommended for analyzing large datasets (>10k tweets).


* **Rate Limits:** The app includes a `time.sleep(1.1)` delay between pagination requests to stay within the standard 1 request/second safety limit.

---

## 📜 License

[MIT License](https://www.google.com/search?q=LICENSE)
