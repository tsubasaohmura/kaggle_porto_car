import pandas as pd
import numpy as np 
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression as lr

# Custom classes
from dataloads import load_train_data, load_test_data

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'

if __name__ == '__main__':
    # Logging section.
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)
    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')
    # Define arguments
    logger.info('Define arguments')

    logger.info('Define training arguments.')
    df_train = load_train_data()
    x_train = df_train.drop('target', axis=1)
    y_train = df_train['target'].values
    cols_train = x_train.columns.values

    logger.info('Define testing arguments.')
    df_test = load_test_data()
    print(df_test)
    x_test = df_test.sort_values('id')

    logger.info(f'Done. \ntraining data is in {df_train.shape}. \ntesting data is in {df_test.shape}.')


    # Training section
    logger.info('Training started.')

    clf = lr(random_state=0)
    clf.fit(x_train, y_train)

    logger.info('Done.')


    # Forecast Testing section
    logger.info('Forecast Testing started.')

    pred_test = clf.predict_proba(x_test)

    logger.info('Done.')

    # Extraction section
    logger.info('Extract section started.')

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test
    print(df_submit.head())

    logger.info('Extracting submit files in .csv format...')
    df_submit.to_csv('submit.csv', index=False)

    logger.info('Done')









