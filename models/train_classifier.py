import sys
import pickle
import pandas as pd
import nltk

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

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load data from the sqlite file at `database_filepath`.
    Split into features and labels, then return these along with category columns

    :param database_filepath: location of the sqlite db file
    :return: the features (the message) along with the target labels (the categories)
    and the list of categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    # Split into features and labels
    x = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return x, y, y.columns


def tokenize(text):
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


def build_model():
    """
    Build the ML Pipeline. First using CountVector and TfidfTransformer
    and then using RandomForest and a MultiOutputClassifier
    :return: the pipeline
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])

    # How should I choose these?
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_estimators': [10, 20],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


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

    for i, category in enumerate(category_names):
        print(f'For category {category}:')
        print(classification_report(y_test[category], y_pred[:, i]))
        print('-'*50)


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file

    :param model: trained model
    :param model_filepath: where to save the pickle file
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        x_data, y_data, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(x_train, y_train)

        print('Grid cv results...')
        print(sorted(model.cv_results_.keys()))
        
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
