import streamlit as st
import tweepy
import pandas as pd
import nltk
import re
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# --- NLTK Setup (Cached) ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# --- Configuration & State Management ---
st.set_page_config(page_title="X-Ray Analytics Engine", layout="wide", page_icon="🐦")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'api_client' not in st.session_state:
    st.session_state['api_client'] = None
if 'data_fetched' not in st.session_state:
    st.session_state['data_fetched'] = False
if 'analytics_data' not in st.session_state:
    st.session_state['analytics_data'] = {}

# --- Helper: Seed Keywords ---
SECTOR_KEYWORDS = {
    "Banking": ["interest rates", "inflation", "RBI", "credit", "loan", "liquidity", "banknifty"],
    "IT & Tech": ["AI", "generative", "SaaS", "cloud", "earnings", "layoffs", "innovation"],
    "Pharma": ["FDA", "drug approval", "clinical trial", "generic", "biotech", "health"],
    "Automotive": ["EV", "sales", "vehicle", "export", "supply chain", "chips"],
    "General": ["market", "economy", "growth", "stocks", "investing"]
}

# --- 1. NLP Processing Engine ---
def clean_text(text):
    """
    Heavy duty cleaning: URLs, Mentions, Emojis, Symbols, Stopwords, Lemmatization.
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove non-alphabetic characters (emojis, numbers, symbols)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization & Stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

# --- 2. API Validation (UPDATED: OAuth 1.0a) ---
def validate_token(api_key, api_secret, access_token, access_token_secret):
    try:
        # Initialize Client with User Context (OAuth 1.0a)
        client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
        
        # Test call - fetch the authenticated user's details
        # This now works because we have User Context
        me = client.get_me()
        
        if me.data:
            st.session_state['api_client'] = client
            st.session_state['authenticated'] = True
            st.toast(f"✅ Logged in as @{me.data.username}", icon="🔓")
            time.sleep(1)
            st.rerun()
            
    except tweepy.errors.Unauthorized:
        st.error("⛔ Authentication Failed: Check your Keys and Tokens.")
    except Exception as e:
        st.error(f"⚠️ Connection Error: {str(e)}")

# --- 3. Data Fetching Logic (Optimized) ---
def fetch_twitter_data(sector, stock_symbol, target_words, client):
    """
    Fetches data until word count target is met or limits reached.
    """
    # Construct Query
    keywords = SECTOR_KEYWORDS.get(sector, [])
    if stock_symbol:
        keywords.append(stock_symbol)
        keywords.append(f"#{stock_symbol}")
        keywords.append(f"${stock_symbol}") # Cashtag
    
    # Create a safe OR query
    query = f"({' OR '.join(keywords[:10])}) -is:retweet lang:en"
    
    collected_texts = []
    current_word_count = 0
    next_token = None
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Initializing search for: {query}...")
    
    try:
        while current_word_count < target_words:
            # Rate limit handling: API v2 allows 100 tweets per req usually
            response = client.search_recent_tweets(
                query=query,
                max_results=100, # Max allowed for standard tier
                next_token=next_token,
                tweet_fields=['text', 'created_at']
            )
            
            if not response.data:
                break
                
            for tweet in response.data:
                clean = clean_text(tweet.text)
                word_len = len(clean.split())
                
                if word_len > 0:
                    collected_texts.append({
                        "raw_text": tweet.text,
                        "clean_text": clean,
                        "word_count": word_len,
                        "sector": sector,
                        "query_tag": stock_symbol if stock_symbol else sector
                    })
                    current_word_count += word_len
            
            # Update Progress
            progress = min(current_word_count / target_words, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Fetched {len(collected_texts)} tweets | Words: {current_word_count}/{target_words}")
            
            # Pagination
            if 'next_token' in response.meta:
                next_token = response.meta['next_token']
                time.sleep(1.1) # Basic Rate Limit Protection
            else:
                break
                
            # Safety break for demo purposes
            if len(collected_texts) > 2000 and current_word_count < target_words:
                 status_text.warning("High fetch count, stopping to preserve quota/time.")
                 break

    except tweepy.errors.TooManyRequests:
        st.error("Rate Limit Hit! X API Quota Exceeded. Processing what we have...")
    except Exception as e:
        st.error(f"API Error: {e}")
        
    return pd.DataFrame(collected_texts)

# --- 4. Analytics Computation ---
def run_analytics(df):
    if df.empty:
        return None, None

    # --- Task 1: TF-IDF ---
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum tfidf scores for each word across all docs
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    df_tfidf = pd.DataFrame(denselist, columns=feature_names)
    
    # Calculate average importance
    word_importance = df_tfidf.mean(axis=0).sort_values(ascending=False)
    tfidf_result = pd.DataFrame({
        "Term": word_importance.index,
        "TF-IDF Score": word_importance.values,
        "Tag": df['query_tag'].iloc[0]
    })

    # --- Task 2: Frequency ---
    all_words = " ".join(df['clean_text']).split()
    word_counts = Counter(all_words)
    freq_df = pd.DataFrame(word_counts.most_common(200), columns=['Word', 'Frequency'])
    freq_df['Tag'] = df['query_tag'].iloc[0]
    
    return tfidf_result, freq_df

# ==========================================
# MAIN UI FLOW
# ==========================================

def main():
    # --- PHASE 1: AUTHENTICATION (BLOCKING) ---
    if not st.session_state['authenticated']:
        st.title("🔐 X-Ray Analytics: Secure Login")
        st.markdown("Enter your **X API Keys & Tokens** (OAuth 1.0a) to access the engine.")
        st.info("Credentials are stored in session RAM only and are wiped on refresh.")

        with st.form("auth_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                api_key = st.text_input("API Key (Consumer Key)", type="password")
                api_secret = st.text_input("API Secret (Consumer Secret)", type="password")
            
            with col2:
                access_token = st.text_input("Access Token", type="password")
                access_secret = st.text_input("Access Token Secret", type="password")
            
            submit = st.form_submit_button("Validate & Connect")
            
            if submit:
                if not (api_key and api_secret and access_token and access_secret):
                    st.warning("⚠️ All 4 fields are required.")
                else:
                    validate_token(api_key, api_secret, access_token, access_secret)
        
        st.stop() # Halts app execution here until authenticated

    # --- PHASE 2: INPUTS (Only visible after Auth) ---
    st.title("📈 Market Sentiment Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sector_input = st.selectbox("Select Sector", list(SECTOR_KEYWORDS.keys()))
        target_words = st.number_input("Target Word Count", min_value=1000, value=5000, step=1000, 
                                     help="System stops fetching once this volume is cleaned and processed.")
        
    with col2:
        stock_input = st.text_input("Stock Symbol / Ticker (Optional)", placeholder="e.g., TSLA, INFY")
        
    start_btn = st.button("🚀 Start Extraction & Processing", type="primary")

    # --- PHASE 3: PROCESSING ---
    if start_btn:
        with st.status("Orchestrating Pipeline...", expanded=True) as status:
            st.write("📡 Connecting to X API Endpoint...")
            
            # Fetch
            df = fetch_twitter_data(sector_input, stock_input, target_words, st.session_state['api_client'])
            
            if not df.empty:
                st.write(f"✅ Data Acquisition Complete. Raw Tweets: {len(df)}")
                
                st.write("🧠 Vectorizing Text & Calculating TF-IDF...")
                tfidf_res, freq_res = run_analytics(df)
                
                # Store in session state
                st.session_state['analytics_data'] = {
                    'tfidf': tfidf_res,
                    'freq': freq_res,
                    'raw': df
                }
                st.session_state['data_fetched'] = True
                status.update(label="Processing Complete!", state="complete", expanded=False)
            else:
                st.error("No data found or API error occurred.")
                status.update(label="Failed", state="error")

    # --- PHASE 4: RESULTS DISPLAY ---
    if st.session_state['data_fetched'] and st.session_state['analytics_data']:
        data = st.session_state['analytics_data']
        
        st.divider()
        st.subheader(f"📊 Analytics Results: {stock_input if stock_input else sector_input}")
        
        tab1, tab2, tab3 = st.tabs(["🔥 TF-IDF Impact", "🔢 Word Frequency", "📥 Raw Data"])
        
        with tab1:
            st.markdown("**Top Terms by TF-IDF Importance**")
            st.dataframe(data['tfidf'], use_container_width=True)
            
            # Download CSV
            csv_tfidf = data['tfidf'].to_csv(index=False).encode('utf-8')
            st.download_button("Download TF-IDF Data", csv_tfidf, "tfidf_analysis.csv", "text/csv")
            
        with tab2:
            st.markdown("**Most Frequent Terms (Stopwords Removed)**")
            st.dataframe(data['freq'], use_container_width=True)
            
            # Bar Chart Visual
            st.bar_chart(data['freq'].set_index('Word').head(20))
            
            csv_freq = data['freq'].to_csv(index=False).encode('utf-8')
            st.download_button("Download Frequency Data", csv_freq, "frequency_analysis.csv", "text/csv")

        with tab3:
            st.markdown("**Processed Tweet Data**")
            st.dataframe(data['raw'][['created_at', 'raw_text', 'clean_text', 'word_count']], use_container_width=True)

if __name__ == "__main__":
    main()
