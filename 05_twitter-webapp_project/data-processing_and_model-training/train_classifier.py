# import libraries
import sys
import joblib
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Loads data from database filepath
    
    :Input:
        :database_filepath: String, path to database to connect
    :Returns:
        :X: Dataframe, X data for training and testing
        :Y: Dataframe, labels for X data, also for training and testing
        :category_names: Names for categories in Y
    """
    
    print("Creating connection to database...")
    
    # Create connection to database
    engine = create_engine('sqlite:///'+database_filepath)
    con = engine.connect()
    
    # Read in data to pandas dataframe
    df = pd.read_sql_table("messages", con)

    print("Database connection successful! Returning appropriate data for further processing...")

    X = df["message"]
    Y = df.drop(labels=["message", "original", "genre"], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Normalizes, tokenizes and lemms text
    
    :Input:
        :text: String, tweet from a supposed natural disaster event
    :Returns:
        :clean_tokens: List of strings, tokenized and cleaned form of the message
    """
    
    # Normalise by setting to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Create tokens 
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatise words
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return clean_tokens


def build_model():
    """
    Creates machine learning pipeline for learning
    
    :Input:
        :None: Doesn't require an input
    :Returns:
        :pipeline: Machine Learning pipeline with fit/predict methods
    """
    
    basic_logit = LogisticRegression(random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(basic_logit, n_jobs=-1))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Takes model and evaluates it against X and Y test with the category names
    
    Note that the category names are necessary because Y_test and Y_pred will
    need to be evaluated one at a time as y_test and y_pred respectively for
    each category
    
    :Input:
        :model: Model trained on X_train and Y_train
        :X_test: Dataframe, validation data for model
        :Y_test: Dataframe, actual labels for the test data in X
        :category_names: List of strings, categories to be evaluated
    :Returns:
        :None: Prints out report to terminal
    """
    
    # Create Y_pred
    Y_pred = model.predict(X_test)
    
    # Evaluate Model
    for i, col in enumerate(category_names):
    
        print("Column {}: {}".format(i, col))
    
        y_true = list(Y_test.values[:, i])
        y_pred = list(Y_pred[:, i])
        target_names = ['is_{}'.format(col), 'is_not_{}'.format(col)]
        print(classification_report(y_true, y_pred, target_names=target_names))
        
    return


def save_model(model, model_filepath):
    """
    Saves model as a pickle file to model_filepath
    
    :Input:
        :model: pipeline/model, to be pickled for later use
        :model_filepath: String, filepath where model will be saved
    :Returns:
        :None: Pickle file will be created at model_path
    """
    
    joblib.dump(model, model_filepath)
    
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()