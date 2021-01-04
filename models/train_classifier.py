import pickle
import pandas as pd
import numpy as np
import nltk
import argparse
import sys

from sqlalchemy import create_engine


from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from typing import Tuple

nltk.data.find('tokenizers/punkt')
nltk.data.find('tokenizers/punkt')
nltk.data.find('tokenizers/punkt')


# Just to import settings from parent
sys.path.insert(0, '..')

from settings import DATABASE_FILENAME, MODEL_PICKLE_FILENAME, TABLE_NAME
from utils import tokenize


def up_sample(df):
    """
    The dataset is very imbalanced with many more messages having the aid category label
    than for example the shop label. This can lead to issues if the ML model just tries
    to classify based on accuracy, since for example if it always set shop=0 it would be accurate
    maybe 90% of the time, but not because it's a good model, just because 90% (or w/e)
    of the messages don't have that label.

    We can try to address this by up-sampling: we look for messages that have at least 1 category but not
    any of the N most popular categories, then we resample those messages to the level of the most popular category

    See also https://elitedatascience.com/imbalanced-classes

    :return: upsampled df
    """
    # Shuffle
    df_temp = df.sample(frac=1, random_state=0)

    # Only related == 1 messages have categories anyway
    categories = df_temp[df_temp.related == 1].drop(columns=['id', 'message', 'original', 'genre', 'related'])

    # How many messages labeled by each category
    cat_counts = categories.sum().sort_values(ascending=False)

    # Set the sampling number equal to the most popular category
    # (aid 11048, weather 7485,....shops 308, offer 306)
    upsample_number = cat_counts[0]

    # Choose the most  popular categories
    popular_cats = cat_counts.index[:3].to_list()

    # Messages with no labels in popular cats but at least 1 category (approx 362 of these)
    minority_messages = categories[(~categories[popular_cats].any(axis=1)) & (categories.sum(axis=1) > 0)]

    df_minority = df.loc[minority_messages.index]
    df_majority = df.loc[~df.index.isin(minority_messages.index)]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,                 # sample with replacement
                                     n_samples=upsample_number,    # to match majority class
                                     )

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled


def load_data(database_filepath: str, up_sample: bool) -> Tuple[pd.Series, pd.Series, np.array]:
    """
    Load data from the sqlite file at `database_filepath`.
    Split into features and labels, then return these along with category columns

    :param database_filepath: location of the sqlite db file
    :param up_sample: should we upsample the minority category messages in the df?
    :return: the features (the message) along with the target labels (the categories)
    and the list of categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(TABLE_NAME, engine)
    if up_sample:
        df = up_sample(df)
    # Split into features and labels
    x = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return x, y, y.columns


def build_model(optimize_parameters=False):
    """
    Build the ML Pipeline. First using CountVector and TfidfTransformer
    and then using RandomForest and a MultiOutputClassifier

    Main parameters for tuning the RF model are n_estimators (10-500),
     max_depth (2-30 or None), min_samples_split (2-30) the ranges vary by
     the type of problem you have and cannot be easily predefined.

    :param optimize_parameters: boolean to control whether to grid search or not
    :return: the pipeline
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                # ('vect_tfdif', TfidfVectorizer(tokenizer=tokenize()))
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])

    if optimize_parameters:
        print('Optimizing parameters...')
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 10000),
            'tfidf_vect__max_df': (0.75, 1.0),
            # 'tfidf_vect__max_features': (None, 5000, 10000),
            # 'tfidf_vect__ngram_range': ((1, 1), (1, 2)),   # unigrams or bigrams
            'clf__estimator__min_samples_split': [2, 3, 4],
            'clf__estimator__n_estimators': [10, 30],
        }

        cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
        return cv

    return pipeline


def evaluate_model(model, x_test, y_test, category_names):
    """
    Make predictions on the test set using the trained model.
    For each category use classification_report to evaluate the precision, recall and f1-score

    :param model: the trained model
    :param x_test:  the features of test set
    :param y_test:  the targets of test set
    :param category_names:  the category column names
    :return: None
    """

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file

    :param model: trained model
    :param model_filepath: where to save the pickle file
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def parse_input_arguments():
    """
    Use argparse to parse the command line arguments

    Returns:
        database_filename (str): database filename. Default value DATABASE_FILENAME
        model_pickle_filename (str): pickle filename. Default value MODEL_PICKLE_FILENAME
        optimize_params (bool): If True perform grid search of the parameters
    """
    parser = argparse.ArgumentParser(description="Disaster Response ML Pipeline")
    parser.add_argument('--database_filename', type=str, default=DATABASE_FILENAME,
                        help='Database filename (cleaned messages)')
    parser.add_argument('--model_pickle_filename', type=str, default=MODEL_PICKLE_FILENAME,
                        help='Pickle file to save model weights')
    parser.add_argument('--optimize_params', action="store_true", default=False,
                        help='Search parameters to find best or not')
    parser.add_argument('--upsample', action="store_true", default=False,
                        help='Upsample the minority messages')
    args = parser.parse_args()
    return args.database_filename, args.model_pickle_filename, args.optimize_params, args.upsample


def main():
    database_filepath, model_filepath, optimize_parameters, up_sample = parse_input_arguments()

    print(f'Loading data...\n    DATABASE: {database_filepath}')
    x_data, y_data, category_names = load_data(database_filepath, up_sample=up_sample)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    print('Building model...')
    model = build_model(optimize_parameters=optimize_parameters)

    print('Training model...')
    model.fit(x_train, y_train)

    if optimize_parameters:
        print('Optimized params were: \n')
        print(model.best_estimator_.get_params())

    print('Evaluating model...')
    evaluate_model(model, x_test, y_test, category_names)

    print(f'Saving model...\n    MODEL: {model_filepath}')
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
