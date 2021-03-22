
import argparse
import os

import pandas as pd
import numpy as np

columns = ['CRIM', 'ZN', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target']

if __name__=='__main__':
    
    sagemaker_processing_input_path = '/opt/ml/processing/input'
    sagemaker_processing_output_path = '/opt/ml/processing/output'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='boston_train.csv')
    parser.add_argument('--test-file', type=str, default='boston_test.csv')
    parser.add_argument('--input-dir', type=str, default=sagemaker_processing_input_path)
    parser.add_argument('--output-dir', type=str, default=sagemaker_processing_output_path)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.input_dir, args.train_file))
    test_df = pd.read_csv(os.path.join(args.input_dir, args.test_file))
        
    cols_xgboost = columns[-1:] + columns[:-1]
    
    train_df = train_df[cols_xgboost]
    test_df = test_df[cols_xgboost]
    
    # Create local output directories
    if not os.path.exists(os.path.join(args.output_dir,'train')):
        os.makedirs(os.path.join(args.output_dir,'train'))
        print('creating the processed train directory')

    if not os.path.exists(os.path.join(args.output_dir,'test')):
        os.makedirs(os.path.join(args.output_dir,'test'))
        print('creating the processed test directory')
    
    output_train_data_path = os.path.join(args.output_dir,'train',args.train_file)
    train_df.to_csv(output_train_data_path,header=False,index=False)
    print('Saved the processed training dataset')

    
    output_test_data_path = os.path.join(args.output_dir,'test',args.test_file)
    test_df.to_csv(output_test_data_path,header=False,index=False)
    print('Saved the processed test dataset')
