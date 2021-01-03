import pickle
import pandas as pd
import numpy as np
import nltk
import argparse

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from settings import DATABASE_FILENAME, MODEL_PICKLE_FILENAME, TABLE_NAME

from typing import Tuple

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.Series, np.array]:
    """
    Load data from the sqlite file at `database_filepath`.
    Split into features and labels, then return these along with category columns

    :param database_filepath: location of the sqlite db file
    :return: the features (the message) along with the target labels (the categories)
    and the list of categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(TABLE_NAME, engine)
    # Split into features and labels
    x = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return x, y, y.columns


def tokenize(text: str):
    """
    NLP pre-processing. First tokenize the message text using NLTK word_tokenize.
    Next use the WordNetLemmatizer and lower-case/strip the lemmatized tokens

    :param text: the text document (message)
    :return: a list of cleaned tokens representing the message
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


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
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])

    if optimize_parameters:
        print('Optimizing parameters...')
        parameters = {
            # 'vect__ngram_range': ((1, 1), (1, 2)),
            # 'vect__max_df': (0.5, 0.75, 1.0),
            # 'vect__max_features': (None, 5000, 10000),
            'tfidf__use_idf': (True, False),
            'clf__estimator__min_samples_split': [2, 3, 4],
            'clf__estimator__n_estimators': [10, 20],
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
    # for i, category in enumerate(category_names):
    #     print(f'For category {category}:')
    #     print(classification_report(y_test[category], y_pred[:, i]))
    #     print('-'*50)


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
    args = parser.parse_args()
    return args.database_filename, args.model_pickle_filename, args.optimize_params


def main():
    database_filepath, model_filepath, optimize_parameters = parse_input_arguments()

    print(f'Loading data...\n    DATABASE: {database_filepath}')
    x_data, y_data, category_names = load_data(database_filepath)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    print('Building model...')
    model = build_model(optimize_parameters=optimize_parameters)

    print('Training model...')
    model.fit(x_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, x_test, y_test, category_names)

    print(f'Saving model...\n    MODEL: {model_filepath}')
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
