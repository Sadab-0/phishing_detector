import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from utils import clean_text, extract_features

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv('dataset/email_data.csv')
    except FileNotFoundError:
        print("Error: dataset/email_data.csv not found. Please add the dataset.")
        return

    # Clean the text data
    df['clean_text'] = df['text'].apply(clean_text)

    X = df[['clean_text', 'text']] # Keep original text for custom feature extraction
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Building Machine Learning Pipeline...")
    
    # We use a custom transformer to run our extract_features function
    feature_extractor = FunctionTransformer(extract_features, validate=False)

    # ColumnTransformer applies TF-IDF to 'clean_text', and custom extraction to 'text'
    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english'), 'clean_text'),
            ('custom_features', feature_extractor, ['text']) # Passes the text column to extract_features
        ]
    )

    # Combine preprocessor with Random Forest Classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    
    print("\n--- Model Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nSaving model to model/phishing_model.pkl...")
    joblib.dump(model_pipeline, 'model/phishing_model.pkl')
    print("Training complete!")

if __name__ == '__main__':
    main()