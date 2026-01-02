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
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        st.warning(f"NLTK Download Warning: {e}")

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
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

# --- 2. API Validation (FIXED FOR BEARER TOKEN) ---
def validate_token(bearer_token):
    try:
        # Initialize Client with Bearer Token
        client = tweepy.Client(bearer_token=bearer_token)
        
        # CRITICAL FIX: instead of get_me() (which needs user keys),
        # we fetch the official 'X' account to test the connection.
        response = client.get_user(username="X")
        
        if response.data:
            st.session_state['api_client'] = client
            st.session_state['authenticated'] = True
            st.toast("✅ Connection Successful!", icon="🟢")
            time.sleep(1)
            st.rerun()
            
    except tweepy.errors.Unauthorized:
        st.error("⛔ Authorization Failed: Invalid Bearer Token or Free Tier limitation.")
    except Exception as e:
        st.error(f"⚠️ Validation Error: {str(e)}")

# --- 3. Data Fetching Logic ---
def fetch_twitter_data(sector, stock_symbol, target_words, client):
    keywords = SECTOR_KEYWORDS.get(sector, [])
    if stock_symbol:
        keywords.append(stock_symbol)
        keywords.append(f"#{stock_symbol}")
    
    # Query Construction
    query = f"({' OR '.join(keywords[:8])}) -is:retweet lang:en"
    
    collected_texts = []
    current_word_count = 0
    next_token = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Searching: {query}...")
    
    try:
        while current_word_count < target_words:
            # SEARCH API CALL
            response = client.search_recent_tweets(
                query=query,
                max_results=50, # Safer batch size
                next_token=next_token,
                tweet_fields=['text', 'created_at']
            )
            
            if not response.data:
                status_text.warning("No tweets found for this query.")
                break
                
            for tweet in response.data:
                clean = clean_text(tweet.text)
                word_len = len(clean.split())
                
                if word_len > 0:
                    collected_texts.append({
                        "raw_text": tweet.text,
                        "clean_text": clean,
                        "word_count": word_len,
                        "query_tag": stock_symbol if stock_symbol else sector
                    })
                    current_word_count += word_len
            
            # Progress Update
            progress = min(current_word_count / target_words, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Fetched {len(collected_texts)} tweets | Words: {current_word_count}")
            
            if 'next_token' in response.meta:
                next_token = response.meta['next_token']
                time.sleep(1.2) # Rate Limit Safety
            else:
                break
                
            if len(collected_texts) > 2000: # Safety cap
                 break

    except tweepy.errors.Forbidden as e:
        st.error(f"⛔ 403 Forbidden: You are likely on the Free Tier. Search is not allowed. ({e})")
        return pd.DataFrame()
    except tweepy.errors.Unauthorized as e:
        st.error(f"⛔ 401 Unauthorized: Check your Bearer Token. ({e})")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(collected_texts)

# --- 4. Analytics Computation ---
def run_analytics(df):
    if df.empty: return None, None

    vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    df_tfidf = pd.DataFrame(tfidf_matrix.todense(), columns=feature_names)
    word_importance = df_tfidf.mean(axis=0).sort_values(ascending=False)
    
    tfidf_result = pd.DataFrame({
        "Term": word_importance.index,
        "Score": word_importance.values
    })

    all_words = " ".join(df['clean_text']).split()
    freq_df = pd.DataFrame(Counter(all_words).most_common(200), columns=['Word', 'Frequency'])
    
    return tfidf_result, freq_df

# --- Main UI ---
def main():
    if not st.session_state['authenticated']:
        st.title("🔐 X-Ray Analytics: Login")
        st.markdown("Enter your **App Bearer Token** (Basic Tier Required).")
        
        token = st.text_input("Bearer Token", type="password")
        if st.button("Connect"):
            if token: validate_token(token)
            else: st.warning("Token required.")
        st.stop()

    st.title("📈 Market Sentiment Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        sector = st.selectbox("Sector", list(SECTOR_KEYWORDS.keys()))
        target = st.number_input("Target Words", value=1000, step=500)
    with col2:
        stock = st.text_input("Stock Symbol (Optional)")
        
    if st.button("🚀 Start Analysis", type="primary"):
        with st.status("Processing...", expanded=True):
            df = fetch_twitter_data(sector, stock, target, st.session_state['api_client'])
            
            if not df.empty:
                tfidf, freq = run_analytics(df)
                st.session_state['analytics_data'] = {'tfidf': tfidf, 'freq': freq, 'raw': df}
                st.session_state['data_fetched'] = True
            
    if st.session_state['data_fetched']:
        data = st.session_state['analytics_data']
        st.divider()
        t1, t2 = st.tabs(["TF-IDF", "Frequency"])
        with t1: st.dataframe(data['tfidf'], use_container_width=True)
        with t2: st.bar_chart(data['freq'].set_index('Word').head(20))

if __name__ == "__main__":
    main()
