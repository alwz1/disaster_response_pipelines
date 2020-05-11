import sys

# import libraries
from joblib import dump, load

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['words', 'punkt', 'stopwords',
               'averaged_perceptron_tagger', 'wordnet'])


def load_data(database_filepath):
    """
        Read table 'message_category' from 'DisasterResponse.db'
        Args: database_filepath
        Returns:
            X: ndarray of messages
            Y: ndarray of category labels
            category_names
    """
    # Create connection to the database
    engine = create_engine('sqlite:///'+database_filepath)
    # Read message_category table from the database
    df = pd.read_sql('SELECT * FROM message_category', engine)
    # features
    X = df['message'].values
    # labels
    Y = df.loc[:, 'related':'direct_report'].values

    category_names = df.loc[:, 'related':'direct_report'].columns

    return X, Y, category_names


def tokenize(text):
    """
        1. Replace url in the text with 'urlplaceholder'
        2. Remove punctuations and use lower cases
        3. Remove stopwords and lemmatize tokens

        Args: text
        Returns: cleaned tokens of text
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok)
                    for tok in tokens if tok not in stop_words]

    return clean_tokens


# Add two customer transformers

def tokenize_2(text):
    """
        Tokenize the input text. This function is called in StartingVerbExtractor.
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(
        tok).lower().strip() for tok in tokens]

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """ return true if the first word is an appropriate verb or RT for retweet """
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize_2(sentence))
            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]

            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """ Fit """
        return self

    def transform(self, X):
        """ Transform """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# Count the number of tokens
class TextLengthExtractor(BaseEstimator, TransformerMixin):

    def text_len_count(self, text):
        """ Count the number of tokens """
        text_length = len(tokenize(text))
        return text_length

    def fit(self, x, y=None):
        """ Fit """
        return self

    def transform(self, X):
        """ Transform """
        X_text_len = pd.Series(X).apply(self.text_len_count)
        return pd.DataFrame(X_text_len)


def build_model():
    """
        Returns a pipeline that applies FeatureUnion, CountVectorizer,
        TfidfTransformer, StartingVerbExtractor,
        and MultiOutputClassifier(XGBClassifier)
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,
                                         max_features=6000,
                                         max_df=0.75)),
                ('tfidf', TfidfTransformer(use_idf=True))
            ])),

            ('txt_length', TextLengthExtractor()),
            ('start_verb', StartingVerbExtractor())

        ])),

        ('norm',  Normalizer()),

        ('clf', MultiOutputClassifier(XGBClassifier(
            max_depth=4,
            # learning_rate=0.2,
            max_delta_step=3,
            colsample_bytree=0.5,
            colsample_bylevel=0.5,
            subsample=0.8,
            # n_estimators=150,
            tree_method='hist')))
    ])

    parameters = {
        'clf__estimator__learning_rate': [0.2, 0.5],
        'clf__estimator__n_estimators': [100, 150]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1)

    return cv


def display_results(y_test, y_pred, category_name):
    """
        Display f1 score, precision, recall and confusion_matrix
        for each category of the test dataset
    """

    clf_report = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print('\n')
    print(category_name, ":")
    print('\n')
    print(clf_report)
    print('confusion_matrix')
    print(confusion_mat)
    print('\n')
    print('-'*65)


def evaluate_model(model, X_test, Y_test, category_names):
    """
       Evaluate model and display f1 score, precision, recall and
       confusion_matrix for each category of the test dataset
    """

    Y_pred = model.predict(X_test)

    for i in range(Y_test.shape[1]):
        display_results(Y_test[:, i], Y_pred[:, i], category_names[i])


def save_model(model, model_filepath):
    """
        Save model to a pickle file
    """
    dump(model, open(model_filepath, 'wb'))


def main():
    """ Build, train, evaluate, and save trained model """
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
