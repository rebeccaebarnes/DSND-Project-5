import json
import sys
from string import punctuation

import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sqlalchemy import create_engine

# Create app
app = Flask(__name__)

# Create classes
class WordCount(BaseEstimator, TransformerMixin):
    def word_count(self, text):
        table = text.maketrans(dict.fromkeys(punctuation))
        words = word_tokenize(text.lower().strip().translate(table))
        return len(words)
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.word_count)
        return pd.DataFrame(count)


class CharacterCount(BaseEstimator, TransformerMixin):
    def character_count(self, text):
        return len(text)
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.character_count)
        return pd.DataFrame(count)


class NounCount(BaseEstimator, TransformerMixin):
    def noun_count(self, text):
        count = 0
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for _, tag in pos_tags:
                if tag in ['PRP', 'NN']:
                    count += 1
        
        return count
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.noun_count)
        return pd.DataFrame(count)


class VerbCount(BaseEstimator, TransformerMixin):
    def verb_count(self, text):
        count = 0
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for _, tag in pos_tags:
                if tag in ['VB', 'VBP']:
                    count += 1
        
        return count
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.verb_count)
        return pd.DataFrame(count)

# Create functions
def load_data(database_filepath):
    engine_location = 'sqlite:///' + database_filepath
    engine = create_engine(engine_location)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    table = text.maketrans(dict.fromkeys(punctuation))
    words = word_tokenize(text.lower().strip().translate(table))
    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    lemmed = [WordNetLemmatizer().lemmatize(word, pos='v') for word in lemmed]
    stemmed = [PorterStemmer().stem(word) for word in lemmed]
    
    return stemmed

# load data
engine = create_engine('sqlite:///./data/disaster_messages.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("./models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()