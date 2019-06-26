import json
import plotly
import pandas as pd
import joblib
import operator
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

# Update tokenize function
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

# Get table name
table_name = engine.table_names()[0]

df = pd.read_sql_table(table_name, engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Get most appearing categories
    category_names = list(df.iloc[:, 3:].sum().sort_values(ascending=False).index)
    category_counts = list(df.iloc[:, 3:].sum().sort_values(ascending=False).values)
    
    # Get top 10 tokens             
    sorted_d = joblib.load("sorted_d.pkl")
    
    token_names = [x[0] for x in sorted_d[-10:]]
    token_counts = [x[1] for x in sorted_d[-10:]]
    
    # create visuals
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
        },
        
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
                    'title': "Category"
                }
            }
        },
       
        {
            'data': [
                Bar(
                    x=token_names,
                    y=token_counts
                )
            ],

            'layout': {
                'title': 'Top 10 Tokens',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Tokens"
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()