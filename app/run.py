import json
import plotly
import pandas as pd
import re
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sqlalchemy import create_engine

import plotly.express as px


app = Flask(__name__)


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


def tokenize_2(text):
    """ Tokenize input text. This function is called in StartingVerbExtractor. """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """ Return true if the first word is an appropriate verb or RT for retweet """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize_2(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_category', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.iloc[:, 4:].columns
    category_counts = (df.iloc[:, 4:]).sum(
    ).sort_values(ascending=False).values

    # Calculate message count by genre and related status
    genre_related = df[df['related'] == 1].groupby('genre').count()[
        'message']
    genre_not_rel = df[df['related'] == 0].groupby('genre').count()[
        'message']

    # create visuals

    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # GRAPH 2 - category graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        },
        # GRAPH 3 - category graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_related,
                    name='Related'
                ),

                Bar(
                    x=genre_names,
                    y=genre_not_rel,
                    name='Not Related'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre and Related Status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'stack'
            }
        }


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(
        zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ Run app on host='0.0.0.0' and port=3001 """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
