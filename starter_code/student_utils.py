import pandas as pd
import numpy as np
import os
import tensorflow as tf

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_out = df.copy()

    # Normalize keys to strings for a safe join
    ndc_lookup = ndc_df.copy()
    ndc_lookup['NDC_Code'] = ndc_lookup['NDC_Code'].astype(str).str.strip()
    ndc_lookup['Non-proprietary Name'] = ndc_lookup['Non-proprietary Name'].astype(str).str.strip()

    df_out['ndc_code'] = df_out['ndc_code'].astype(str).str.strip()

    ndc_lookup = ndc_lookup[['NDC_Code', 'Non-proprietary Name']].drop_duplicates()
    ndc_lookup = ndc_lookup.rename(columns={'NDC_Code': 'ndc_code', 'Non-proprietary Name': 'generic_drug_name'})

    df_out = df_out.merge(ndc_lookup, on='ndc_code', how='left')
    df_out['generic_drug_name'] = df_out['generic_drug_name'].fillna('unknown')

    return df_out

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first')
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    unique_patients = df[patient_key].unique()
    np.random.seed(42)
    shuffled_patients = np.random.permutation(unique_patients)
    total_patients = len(shuffled_patients)
    train_size = int(0.6 * total_patients)
    val_size = int(0.2 * total_patients)
    
    train_patients = shuffled_patients[:train_size]
    val_patients = shuffled_patients[train_size:train_size + val_size]
    test_patients = shuffled_patients[train_size + val_size:]
    
    train = df[df[patient_key].isin(train_patients)].reset_index(drop=True)
    validation = df[df[patient_key].isin(val_patients)].reset_index(drop=True)
    test = df[df[patient_key].isin(test_patients)].reset_index(drop=True)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction
