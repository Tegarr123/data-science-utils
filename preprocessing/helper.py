import pandas as pd

import pandas as pd
import torch.utils.data as data_utils
import torch
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

def df_to_dataloader(train_data:pd.DataFrame,
                     test_data:pd.DataFrame,
                     label_feature:str,
                     batch_size:int=None
                     ):
    # dataframe to tensor
    X_train = torch.tensor(train_data.drop(columns=label_feature,axis=1).values, dtype=torch.float32)
    y_train = torch.tensor(train_data.loc[:,label_feature].values, dtype=torch.float32)
    X_test = torch.tensor(test_data.drop(columns=label_feature, axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(test_data.loc[:, label_feature].values, dtype=torch.float32)
    # Create torch dataset
    train_tensor = data_utils.TensorDataset(X_train, y_train)
    test_tensor = data_utils.TensorDataset(X_test, y_test)
    # Create data loader
    train_loader = data_utils.DataLoader(dataset=train_tensor,
                                         batch_size=batch_size,
                                         shuffle=True)
    test_loader = data_utils.DataLoader(dataset=test_tensor,
                                        batch_size=batch_size,
                                        shuffle=True)
    return train_loader,test_loader