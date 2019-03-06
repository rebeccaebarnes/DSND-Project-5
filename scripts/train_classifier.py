import argparse
import json
import pickle
from string import punctuation

import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

# Create argparser
parser = argparse.ArgumentParser(description='Categorize train ml/nlp pipeline')
parser.add_argument("database_filepath", help="File path for database")
parser.add_argument("model_filepath", help="File path for saving model")
parser.add_argument('-d', '--params_dict', 
                    help='Dictionary of model parameters. Dictionary should be\
                          passed in string form with values in a list, e.g. \
                          "{key: [value(s)]}". To see available params, use \
                          "train_classifer.py database/filepath model/filepath\
                          -p"', 
                    type=json.loads)
parser.add_argument('-p', '--available_params', action='store_true',
                    help='Details of model parameter keys')
args = parser.parse_args()

# Set up classes
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
    engine = create_engine(database_filepath)
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


def build_model(X_train, Y_train, params=None):
    if not params:
        params = {
            'clf__estimator__max_depth': [500],
            'clf__estimator__min_samples_split': [25],
            'clf__estimator__n_estimators': [300],
            'features__text__max_df': [0.5],
            'features__text__max_features': [5000],
            'features__text__ngram_range': [(1, 2)],
            'features__text__use_idf': [False]
        }
    
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("text", TfidfVectorizer(tokenizer=tokenize)),
                ("word_count", WordCount()),
                ("character_count", CharacterCount()),
                ("noun_count", NounCount()),
                ("verb_count", VerbCount())
            ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42, 
                                                             verbose=1)))
        ])

    scorer = make_scorer(f1_score, average='micro')

    cv = GridSearchCV(pipeline, params, cv=5, n_jobs=3, scoring=scorer)
    print('Training model...')
    cv.fit(X_train, Y_train)
    return cv


def evaluate_model(model, X_test, Y_test, col_names):
    y_preds = model.predict(X_test)
    for label, pred, col in zip(Y_test.values.transpose(), y_preds.transpose(), 
                                col_names):
        print(col)
        print(classification_report(label, pred))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main(database_filepath=args.database_filepath, 
         model_filepath=args.model_filepath, params=args.params_dict):
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
    print('Building model...')
    model = build_model(X_train, Y_train, params)
        
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    if args.available_params:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("text", TfidfVectorizer(tokenizer=tokenize)),
                ("word_count", WordCount()),
                ("character_count", CharacterCount()),
                ("noun_count", NounCount()),
                ("verb_count", VerbCount())
            ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42, 
                                                             verbose=1)))
        ])

        print(pipeline.get_params())
    
    else:
        main()