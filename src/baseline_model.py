import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from src.evaluate import evaluate_model
import json
import os

CONFIG = {
    'input_file': r'C:\Users\Lenovo\Desktop\PD ML SET 2\data\processed_abstracts.csv',
    'output_dir': r'C:\Users\Lenovo\Desktop\PD ML SET 2\results',
    'random_state': 42,
    'test_size': 0.2
}

def load_split_data():
    """
    Loads processed abstracts and performs an 80/20 train/test split correctly.
    """
    df = pd.read_csv(CONFIG['input_file'])
    
    # Split FIRST before any feature engineering
    X = df['abstract_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], stratify=y, random_state=CONFIG['random_state']
    )
    
    return X_train, X_test, y_train, y_test

def train_logreg(X_train, y_train_enc, le):
    print("Training TF-IDF + Logistic Regression Pipeline with CV Sanity Check...")
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, random_state=CONFIG['random_state'], C=1.0, multi_class='multinomial', solver='lbfgs'))
    ])
    
    # Cross-validation sanity check on training data only
    cv_scores = cross_val_score(lr_pipeline, X_train, y_train_enc, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"LR Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    lr_pipeline.fit(X_train, y_train_enc)
    return lr_pipeline

def train_nb(X_train, y_train_enc, le):
    print("Training TF-IDF + Multinomial NB Pipeline...")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    
    nb_pipeline.fit(X_train, y_train_enc)
    return nb_pipeline

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    X_train, X_test, y_train, y_test = load_split_data()
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_.tolist()
    
    # Model 1
    mod1 = train_logreg(X_train, y_train_enc, le)
    y_pred1_enc = mod1.predict(X_test)
    y_prob1 = mod1.predict_proba(X_test)
    y_pred1 = le.inverse_transform(y_pred1_enc)
    metrics1 = evaluate_model(y_test, y_pred1, y_prob1, classes, 'Logistic Regression', CONFIG['output_dir'])
    
    # Model 2
    mod2 = train_nb(X_train, y_train_enc, le)
    y_pred2_enc = mod2.predict(X_test)
    y_prob2 = mod2.predict_proba(X_test)
    y_pred2 = le.inverse_transform(y_pred2_enc)
    metrics2 = evaluate_model(y_test, y_pred2, y_prob2, classes, 'Multinomial NB', CONFIG['output_dir'])
    
    # Save metrics
    results = [metrics1, metrics2]
    with open(os.path.join(CONFIG['output_dir'], 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation complete. Metrics saved to results/ folder.")

if __name__ == "__main__":
    main()
