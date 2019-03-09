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
from plotly.graph_objs import Bar, Heatmap
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
    data = df.iloc[:, 4:].drop('child_alone', axis=1)
    related_counts = data.related.value_counts().tolist()
    related_msg_counts = data[data.related == 1].sum().tolist()[1:]
    
    corr_list = []
    correl = data.corr().values
    for row in correl:
        corr_list.append(list(row))
    
    col_names = [col.replace('_', ' ').title() for col in data.columns]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=['Related', 'Not Related'],
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'How Many Messages were Related to Disasters?',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                },
                'margin': dict(
                    l = 350,
                    r = 350, 
                )
            }
        },
        {
            'data': [
                Bar(
                    x=col_names[1:],
                    y=related_msg_counts
                )
            ],

            'layout': {
                'title': 'Types of Disaster Messages were Received?',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                },
                'height': 620,
                'margin': dict(
                    b = 230,
                    pad = 4
                    ),
            }
        },
        {
            'data': [
                Heatmap(
                    z=corr_list, 
                    x=col_names,
                    y=col_names,
                    colorscale='Viridis',
                )
            ],

            'layout': {
                'title': 'What Types of Messages Occur Together?',
                'height': 750,
                'margin': dict(
                    l = 150,
                    r = 30, 
                    b = 160,
                    t = 30,
                    pad = 4
                    ),
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