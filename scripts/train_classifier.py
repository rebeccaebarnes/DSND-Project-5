import json
import os
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

# Set up classes
class WordCount(BaseEstimator, TransformerMixin):
    '''
    Custom scikit-learn transformer to count the number of words in text.
    '''
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
    '''
    Custom scikit-learn transformer to count the number of characters in text, 
    including spaces and punctuation.
    '''
    def character_count(self, text):
        return len(text)
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        count = pd.Series(x).apply(self.character_count)
        return pd.DataFrame(count)


class NounCount(BaseEstimator, TransformerMixin):
    '''
    Custom scikit-learn transformer to count the number of nouns in text after
    tokenization including removal of stop words, lemmatization of nouns and 
    verbs, and stemming, using nltk's WordNetLemmatizer and PorterStemmer.
    '''
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
    '''
    Custom scikit-learn transformer to count the number of nouns in text after
    tokenization using a custom "tokenize" function.
    '''
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
    '''
    Load 'messages' table from a database and extract X and Y values and 
    category names.
    '''
    engine_location = 'sqlite:///' + database_filepath
    engine = create_engine(engine_location)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizes text after standardizing text, removing punctuation and stop words
    by lemmatizing nouns and verbs, and stemming, using nltk's WordNetLemmatizer
    and PorterStemmer.
    '''
    table = text.maketrans(dict.fromkeys(punctuation))
    words = word_tokenize(text.lower().strip().translate(table))
    words = [word for word in words if word not in stopwords.words('english')]
    lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]
    lemmed = [WordNetLemmatizer().lemmatize(word, pos='v') for word in lemmed]
    stemmed = [PorterStemmer().stem(word) for word in lemmed]
    
    return stemmed


def build_model(X_train, Y_train, params=None):
    '''
    Create a multi-output Random Forest classifier machine learning pipeline for
    natural language processing with tdidf, word_count, character_count, 
    noun_count, and verb_count features. If params are provided, grid search is 
    conducted for optimization. Default paramas are max_df=0.5, 
    max_features=5000, ngram_range=(1, 2), use_idf=False for tfidf feature and 
    min_samples_split=25, max_depth=500, n_estimators=300 for the classifier.

    Args:
        X_train: Array-like. Text to be analayzed. 
        Y_train: Array-like. Classification labels.
        params: Optionall. Dictionary. Range of parameters to search with grid 
                search.
    Returns:
        Fitted Random Forest classifer.
    '''
    if not params:
        model = Pipeline([
            ("features", FeatureUnion([
                ("text", TfidfVectorizer(tokenizer=tokenize, max_df=0.5, 
                                 max_features=5000, ngram_range=(1, 2),
                                 use_idf=False)),
                ("word_count", WordCount()),
                ("character_count", CharacterCount()),
                ("noun_count", NounCount()),
                ("verb_count", VerbCount())
            ])),
            ("clf", MultiOutputClassifier(RandomForestClassifier(
                min_samples_split=25, random_state=42, 
                max_depth=500, n_estimators=300)))
            ])

    else:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("text", TfidfVectorizer(tokenizer=tokenize)),
                ("word_count", WordCount()),
                ("character_count", CharacterCount()),
                ("noun_count", NounCount()),
                ("verb_count", VerbCount())
            ])),
            ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42)))
            ])

        scorer = make_scorer(f1_score, average='micro')

        model = GridSearchCV(pipeline, params, cv=5, n_jobs=3, scoring=scorer)

    print('Training model...')
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test, col_names):
    '''
    Print the precision, recall and f1-scores for a multi-output 
    classification.
    '''
    y_preds = model.predict(X_test)
    for label, pred, col in zip(Y_test.values.transpose(), y_preds.transpose(), 
                                col_names):
        print(col)
        print(classification_report(label, pred))


def save_model(model, model_filepath):
    '''
    Pickle model in specified location.
    '''
    # Assume maximum depth of one directory for location
    if model_filepath.find('/'):
        folder_name = model_filepath.split('/')[0]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main(database_filepath, model_filepath, params):
    '''
    Extract datafrom database, train a multi-output Random Forrest classifier, 
    print evaluation statistics, and save the model.
    '''
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
    # Create argparser
    import argparse
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

    if args.available_params:
        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("text", TfidfVectorizer(tokenizer=tokenize)),
                ("word_count", WordCount()),
                ("character_count", CharacterCount()),
                ("noun_count", NounCount()),
                ("verb_count", VerbCount())
            ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42)))
        ])

        print(pipeline.get_params())
    
    else:
        main(database_filepath=args.database_filepath, 
             model_filepath=args.model_filepath, params=args.params_dict)