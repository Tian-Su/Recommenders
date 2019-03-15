import sys
sys.path.append("../../../")

import pandas as pd
from datetime import datetime

from reco_utils.dataset.python_splitters import python_chrono_split


if __name__ == "__main__":
    
    ############
    # Parameter set up
    ############
    
    root_path = "../data/"
    readin_path = root_path + "reward_transaction_subcat.csv"
    train_save_path = root_path + "reward_transaction_subcat_train_1.csv"
    test_save_path = root_path + "reward_transaction_subcat_test_1.csv"
    
    train_size = 0.75
    original_user_col = "loyalty_id"
    col_user="userID"
    original_item_col = "subcat_concat"
    col_item="itemID"
    original_time_col = "trns_timestamp"
    col_timestamp="datetime"
    
    
    ############
    # Readin data
    ############
    df = pd.read_csv(readin_path)
    print("readin data.shape ", df.shape)
    print("columns ", df.columns)
    
    
    ############
    # Create time variables
    ############
    df[col_timestamp] = df[original_time_col].apply(lambda x: datetime.strptime(x[:10] + ' ' + x[11:18], "%Y-%m-%d %H:%M:%S"))
    df["dayofweek"] = df[col_timestamp].apply(lambda x: x.weekday())
    df["hourofday"] = df[col_timestamp].apply(lambda x: x.hour)
    df["monthofyear"] = df[col_timestamp].apply(lambda x: x.month)
    
    ############
    # Convert the loyalty_id to number for the model
    ############
    df_user = df[[original_user_col]].drop_duplicates()
    df_user = df_user.reset_index().drop("index", axis=1).reset_index()
    df_user.columns = [col_user, original_user_col]
    print("unique user number ", df_user.shape[0])
    
    ############
    # Convert the subcategory to number for the model
    ############
    df_item = df[[original_item_col]].drop_duplicates()
    df_item = df_item.reset_index().drop("index", axis=1).reset_index()
    df_item.columns = [col_item, original_item_col]
    print("unique item number ", df_item.shape[0])
    
    ############
    # Merge userID and itemID back to original data
    ############
    df_order_user = df.merge(df_user, how="left", on=original_user_col)
    df_order_user_item = df_order_user.merge(df_item, how="left", on=original_item_col)
    
    
    ############
    # Time split of train test
    ############
    train, test = python_chrono_split(df_order_user_item, train_size, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp)
    
    print("train.shape ", train.shape)
    print("columns ", train.columns)
    print("test.shape ", test.shape)
    print("columns ", test.columns)
    
    ############
    # Save cleaned splited data
    ############
    train.to_csv(train_save_path, header=True)
    test.to_csv(test_save_path, header=True)
    
    print("train data saved to ", train_save_path)
    print("test data saved to ", test_save_path)
    
    
    
    
    
    
    
    
    
    