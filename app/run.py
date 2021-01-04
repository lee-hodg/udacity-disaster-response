import json
import plotly
import pandas as pd
import joblib
import sys

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

sys.path.insert(0, '..')

from settings import DATABASE_FILENAME, TABLE_NAME, MODEL_PICKLE_FILENAME

app = Flask(__name__)

# load data
engine = create_engine(f'sqlite:///{DATABASE_FILENAME}')
df = pd.read_sql_table(TABLE_NAME, engine)

# load model
model = joblib.load(MODEL_PICKLE_FILENAME)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Class distribution
    class_dist = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()/len(df)
    class_dist = class_dist.sort_values(ascending=False)*100
    class_name = list(class_dist.index)

    # How many labels does each message have?
    label_dist = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum(axis=1)
    label_counts = (label_dist.value_counts()/len(df))*100
    label_count, message_count = label_counts.index, label_counts.values

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
                    x=class_name,
                    y=class_dist,
                    name='Class ',
                    marker=dict(
                        color='rgb(212, 228, 247)'
                    )
                )
            ],
            'layout': {
                'title': 'Distribution of Category Labels',
                'yaxis': {
                    'title': "% of messages with this label"
                },
                'xaxis': {
                    'title': "Label",
                },
            }
        },
        {
            'data': [
                Bar(
                    x=label_count,
                    y=message_count
                )
            ],
            'layout': {
                'title': 'Distribution of Number of Category Labels Per Message',
                'yaxis': {
                    'title': "% of messages with this many labels"
                },
                'xaxis': {
                    'title': "Number of Labels",
                },
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


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
