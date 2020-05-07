import sys

# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from custom_transformer import StartingVerbExtractor

nltk.download(['words', 'punkt', 'stopwords',
               'averaged_perceptron_tagger',
               'maxent_ne_chunker', 'wordnet'])


def load_data(database_filepath):
    # Create connection to the database
    conn = sqlite3.connect(database_filepath)
    # Read message_category table from the database
    df = pd.read_sql('SELECT * FROM message_category', conn)
    # features
    X = df['message'].values
    # labels
    Y = df.loc[:, 'related':'direct_report'].values

    return X, Y


def tokenize(text):

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # replace urls with placeholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # convert to lowercase and remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    # lemmatize verbs by specifying pos
    lemmed_tokens = [WordNetLemmatizer().lemmatize(w, pos='v')
                     for w in lemmed]
    # stem tokens
    cleaned_tokens = [PorterStemmer().stem(w) for w in lemmed_tokens]

    return cleaned_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
