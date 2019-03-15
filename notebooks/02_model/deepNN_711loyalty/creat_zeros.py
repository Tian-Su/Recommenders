import pandas as pd
from itertools import product
# import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


if __name__ == "__main__":
    

    #############
    # Parameter setup
    #############
    necessary_col = ['userID', 'itemID', 'dayofweek', 'hourofday', 'monthofyear', 'quantity']
    test_size = 0.25
    root_path = "../data/"
    train_readin_path = root_path + "reward_transaction_subcat_train.csv"
    test_readin_path = root_path + "reward_transaction_subcat_test.csv"
    train_save_path = root_path + "reward_transaction_subcat_withZero_train.csv"
    test_save_path = root_path + "reward_transaction_subcat_withZero_test.csv"
    
    sample_zero_size = 2000000
    
    #############
    # Read in data
    #############
    train = pd.read_csv(train_readin_path)
    train = train[necessary_col]
    test = pd.read_csv(test_readin_path)
    test = test[necessary_col]
    
    
    print("train ", train.shape)
    print("test ", test.shape)
    
    #############
    # Create meshed df for 0 quantity records
    #############
    df = pd.concat([train, test],ignore_index=True).drop_duplicates().reset_index(drop=True)
    df_itemID = df["itemID"].unique()
    df_userID = df["userID"].unique()
    df_user_item_mesh = pd.DataFrame(product(df_userID, df_itemID))
    df_user_item_mesh.columns = ["userID", "itemID"]
    
    #############
    # Get the non-zero quantity user-item pairs
    #############
    df_trans_user_time_pair = df[["userID", "itemID"]].drop_duplicates()
    
    #############
    # Exclude the non-zero quantity user-item pairs from the mesh df
    # Get the true zero purchase user-item pairs
    #############
    df_user_item_null = df_user_item_mesh.merge(df_trans_user_time_pair, on=['userID','itemID'], 
                   how='left', indicator=True)
    df_user_item_null = df_user_item_null[df_user_item_null["_merge"] == "left_only"]
    
    
    #############
    # Down sample the zero quantity user-item pairs
    #############
    df_user_item_null_sample = df_user_item_null.sample(n=sample_zero_size, random_state=1)
    
    #############
    # Impute the values
    #############
    df_user_item_null_sample["dayofweek"] = -1
    df_user_item_null_sample["hourofday"] = -1
    df_user_item_null_sample["monthofyear"] = -1
    df_user_item_null_sample["quantity"] = 0
    
    #############
    # Split zero quantity df for train and test
    #############
    df_user_item_null_sample = df_user_item_null_sample[necessary_col]
    train_null, test_null = train_test_split(df_user_item_null_sample, test_size=test_size)
    
    print("train_null.shape ", train_null.shape)
    print("test_null.shape ", test_null.shape)

    #############
    # Combine non-zero with zeros
    #############
    train = pd.concat([train, train_null],ignore_index=True).drop_duplicates().reset_index(drop=True)
    test = pd.concat([test, test_null],ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    train = shuffle(train)
    test = shuffle(test)
    
    print("train ", train.shape)
    print("test ", test.shape)
    
    #############
    # Save the data
    #############
    train.to_csv(train_save_path, header=True)
    print("train saved to ", train_save_path)
    test.to_csv(test_save_path, header=True)
    print("test saved to ", test_save_path)
    
    
    

    
    
    
    
    
    
    
    
    