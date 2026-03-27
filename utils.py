import re
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import pandas as pd

# Suppress the annoying BeautifulSoup warning for plain text emails
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# High-risk keywords often found in phishing campaigns
RISK_KEYWORDS = [
    'urgent', 'verify', 'login', 'password', 'bank', 
    'suspend', 'account', 'invoice', 'payment', 'winner'
]

def clean_text(text):
    """Removes HTML tags and normalizes text."""
    if not isinstance(text, str):
        return ""
    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_features(df):
    """
    Extracts custom security features from a dataframe containing a 'text' column.
    Returns a DataFrame with numerical features.
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. URL Scanning: Count URLs (Phishing emails often have multiple/obfuscated links)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    features['url_count'] = df['text'].apply(lambda x: len(re.findall(url_pattern, str(x))))
    
    # 2. Email Length
    features['email_length'] = df['text'].apply(lambda x: len(str(x)))
    
    # 3. Keyword Risk Scoring
    def calculate_risk_score(text):
        text_lower = str(text).lower()
        score = sum(1 for word in RISK_KEYWORDS if word in text_lower)
        return score
    
    features['risk_score'] = df['text'].apply(calculate_risk_score)
    
    # 4. HTML Content Presence
    features['has_html'] = df['text'].apply(lambda x: 1 if bool(BeautifulSoup(str(x), "html.parser").find()) else 0)
    
    return features