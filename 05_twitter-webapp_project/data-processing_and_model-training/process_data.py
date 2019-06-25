import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories from CSV files and returns them as pandas dataframes
    
    :Input:
        :messages_filepath: CSV of the twitter messages
        :categories_filepath: CSV of categories
    :Returns:
        :df: Joined but uncleaned version of the two CSV files
    """
    
    # Import files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge dataframes
    df = pd.merge(messages, categories, how='left', left_on='id', right_on='id').drop(labels=["id"], axis=1)

    return df


def clean_data(df):
    """
    Takes loaded data frame from `load_data` and cleans it for use.
    
    :Input:
        :df: Data frame created from `load_data`
    :Returns:
        :df: Cleaned data frame.
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]


    # use this row to extract a list of new column names for categories.
    category_colnames = []
    for i in row:
        category_colnames.append(i.split("-")[0])
        
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Clean each value to leave only a numeric value
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[1])
 
    # Change columns from strings to numerical values
    categories = categories.astype(int)

    # drop the original categories column from `df`
    df = df.drop(labels=["categories"], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # Drop `child_alone` column
    df = df.drop(labels=["child_alone"], axis=1)
    
    # Change 2s in `related` to 1s
    df["related"] = df["related"].apply(lambda x: 1 if x > 0 else 0)
    
    return df

    
def save_data(df, database_filename):
    """
    Takes clean dataframe from `clean_data` and saves it inside of an SQLite database with a desired filename.
    
    :Input:
        :df: Clean data frame from `clean_data`
        :database_filename: String, file name of database
    :Returns:
        :None: Does not return anything but creates a SQLite database
    """
        
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, 
                                                                             categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    
    main()