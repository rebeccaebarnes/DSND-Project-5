import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Create dataframes for messages and categories data.
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return messages_df, categories_df


def clean_data(messages_df, categories_df):
    '''
    Create a clean, combined dataframe of messages and category dummy variables.

    Args:
        messages_df: DataFrame. Contains 'id' column for joining.
        categories_df: DataFrame. Contains 'id' column for joining and 
        'categories' column with strings of categories separated by ;.
    '''
    # Merge datasets
    df = pd.merge(messages_df, categories_df, on='id')

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Create categories columns
    new_cat_df = df.categories.str.split(';', expand=True)

    col_names = df.iloc[0].str[:-2]
    new_cat_df.columns = col_names

    for col in col_names:
        new_cat_df[col] = new_cat_df[col].str[-1].astype(int)

    df = pd.concat([df.drop('categories', axis=1), new_cat_df], axis=1)

    # Clean 'related' values
    df.loc[(df.related == 2), 'related'] = 1

    return df


def save_data(df, database_filepath):
    '''
    Save dataframe to database in 'messages' table. Replace any existing data.
    '''
    conn = sqlite3.connect(database_filepath)
    df.to_sql('messages', con=conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


def main(messages_filepath, categories_filepath, database_filepath):
    '''
    Extract messages and categories data from csv files, cleans the data and 
    saves merged data into database. Checks if data already saved.
    '''
    # Load data
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
    messages_df, categories_df = load_data(messages_filepath, 
                                           categories_filepath)    

    
    conn = sqlite3.connect(database_filepath)
    # Check if data already in db
    try:
        row_count = pd.read_sql('SELECT COUNT(*) FROM messages', conn).iloc[0][0]
        if categories_df.drop_duplicates().shape[0] == row_count:
            print('Database is up to date')
            conn.close()
        else:
            print('Cleaning data...')
            df = clean_data(messages_df, categories_df)

            print('Saving data...')
            save_data(df, database_filepath)
            conn.close()

    except sqlite3.OperationalError:
        print('Cleaning data...')
        df = clean_data(messages_df, categories_df)

        print('Saving data...')
        save_data(df, database_filepath)
        conn.close()

    

if __name__ == '__main__':
    # Create argparser
    import argparse
    parser = argparse.ArgumentParser(description='Categorize disaster messages')
    parser.add_argument("messages_filepath", help="File path for messages csv")
    parser.add_argument("categories_filepath", help="File path for categories csv")
    parser.add_argument("database_filepath", help="File path for database")
    args = parser.parse_args()

    main(messages_filepath=args.messages_filepath, 
         categories_filepath=args.categories_filepath, 
         database_filepath=args.database_filepath)