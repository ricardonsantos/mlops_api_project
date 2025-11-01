import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
def data_preprocessing(input_file: str, output_file: str, target_col : str = 'salary'):
    """
    Preprocess raw data (cleaning and value inputers) 
    """
    # load data already removing trailing spaces and identifying Null values
    logging.info('loading and checking raw data')
    df = pd.read_csv(input_file, skipinitialspace=True, na_values=[' ','?'])
    logging.info(df.info(verbose=True))
    logging.info('done!')

    logging.info('cleaning special characters in dataframe')
    # remove hyphens from column names
    df.columns = df.columns.str.replace("-", "_")
    # identify categorical and numerical values
    categorical_cols = df.select_dtypes('O').columns
    numerical_cols = df.select_dtypes(include='number').columns
    # remove hyphens from categorical values
    for c in categorical_cols:
        df[c] = df[c].str.replace('-','_').str.lower()
    logging.info('done!')

    logging.info('checking for columns with Null values')
    if df.isnull().any().any():
        logging.info('Null values found. Using value inputers')
        for c in categorical_cols.drop('salary'):
            df[c] = df[c].fillna(df[c].mode()[0])
        for n in numerical_cols:
            df[n] = df[n].fillna(df[n].mean())
    logging.info('done')
    
    logging.info(f'persisting file output as {output_file}')
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    data_preprocessing('data/census.csv', 'data/census_preprocessed.csv')
