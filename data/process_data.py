import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories CSV files into dataframes.
    Merge them on `id` column with an inner join. Each id in the left frame
    corresponds to one in the right frame so "inner" join is fine.

    :param messages_filepath: the path on disk for the messages CSV
    :param categories_filepath: the path on disk for the categories CSV
    :return: The pandas dataframe created by merging the 2 datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df


def clean_data(df):
    """
    We want to one-hot encode the categories ultimately having each as its own column and each row with a 0/1
    if that example belongs to the category.

    We needed to merge initially to ensure categories matched with messages, but now we form a new categories df
    from the merged df categories column, splitting the categories on ';', creating column names, and then retaining
    only the binary 0/1 in as the row values. We finally drop the categories column in the original df and concat our
    new one-hot encoded categories df with it.


    :param df: the dataframe from load_data()
    :return: A new pandas dataframe with our categories one-hot encoded.
    """
    # We split the categories on the ; delimiter expanding into their own columns with expand=True
    categories = df.categories.str.split(';', expand=True)

    # Take the first row and remove the -N from each category to serve as column names
    row = categories.iloc[0, :]
    # use this row to extract a list of new column names for categories.
    category_column_names = row.apply(lambda x: x[:-2]).values
    categories.columns = category_column_names

    # For each column we want the row values to be just the binary numeric part 1/0
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)

    # Now we drop the categories from the merged df and concat with the
    #  new categories frame (note axis=1 meaning concat on columns)
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Check we now have no duplicates
    assert df[df.duplicated()].shape[0] == 0

    # Set 2 to 1 in the related column
    df.loc[df.related == 2] = 1

    # Also child_alone always 0 so useless
    df.drop(columns=['child_alone'], inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the pandas dataframe `df` to an sqlite database with filename `database_filename`

    :param df: pandas dataframe to be saved
    :param database_filename: string database filename
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the file paths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
