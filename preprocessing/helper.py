import pandas as pd
def encode_categorical(dataframe:pd.DataFrame,
                       feature_name:str|list):
    if type(feature_name) is str:
        get_encoded = pd.get_dummies(dataframe.loc[:,[feature_name]],dtype=int) # Get Encoded data
        dataframe.drop(columns=[feature_name], axis=1, inplace=True)
        dataframe[get_encoded.columns] = get_encoded
    elif type(feature_name) is list:
        get_encoded = pd.get_dummies(dataframe.loc[:,feature_name],dtype=int) # Get Encoded data
        dataframe.drop(columns=feature_name, axis=1, inplace=True)
        dataframe[get_encoded.columns] = get_encoded