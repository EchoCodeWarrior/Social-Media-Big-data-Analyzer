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
    # Remove mentions (@user) and hashtags (#topic) symbols only, keeping text if needed? 
    # Usually better to remove the @handle entirely to avoid analyzing usernames.
    text = re.sub(r'@\w+', '', text)
    # Remove non-alphabetic characters (emojis, numbers, symbols)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization & Stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

# --- 2. API Validation ---
def validate_token(bearer_token):
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        # Test call - fetch the authenticated user's details
        me = client.get_me()
        if me.data:
            st.session_state['api_client'] = client
            st.session_state['authenticated'] = True
            st.session_state['bearer_token'] = bearer_token
            st.success(f"Authenticated as ID: {me.data.id}")
            time.sleep(1)
            st.rerun()
    except Exception as e:
        st.error(f"Validation Failed: {str(e)}")

# --- 3. Data Fetching Logic (Optimized) ---
def fetch_twitter_data(sector, stock_symbol, target_words, client):
    """
    Fetches data until word count target is met or limits reached.
    Optimized for X API quotas.
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
                time.sleep(1.1) # Basic Rate Limit Protection (1 request/sec roughly)
            else:
                break
                
            # Safety break for demo purposes to avoid infinite loops if API returns little data
            if len(collected_texts) > 5000 and current_word_count < target_words:
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
        st.markdown("Enter your X API v2 Bearer Token to access the analysis engine.")
        
        token_input = st.text_input("Bearer Token", type="password", help="From developer.twitter.com")
        
        if st.button("Validate & Connect"):
            if not token_input:
                st.warning("Please enter a token.")
            else:
                validate_token(token_input)
        
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
