import sys
sys.path.append("../../")
sys.path.append("../../../")
import time
import os
import shutil
import papermill as pm
import pandas as pd
import numpy as np
import tensorflow as tf
from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_chrono_split
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)

from datetime import datetime

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Reshape
from keras.layers import Dot
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout



def model_structure_NN_MF(df):
    
    ###############
    # NN structure with contactual information
    # Day of week (as input to dense layer)
    # Time of day (as input to dense layer)
    ###############

    user_id_input = Input(shape=[1], name='user')
    item_id_input = Input(shape=[1], name='item')
    dayofweek_input = Input(shape=[1], name='dayofweek')
    hourofday_input = Input(shape=[1], name='hourofday')
    monthofyear_input = Input(shape=[1], name='monthofyear')

    user_count = df.userID.unique().shape[0]
    item_count = df.itemID.unique().shape[0]
    dayofweek_count = df.dayofweek.unique().shape[0]
    print (user_count, item_count, dayofweek_count)



    ##############
    # NN structure
    ##############
    user_embedding = Embedding(output_dim=embedding_size_user, input_dim=user_count,
                               input_length=1, name='user_embedding')(user_id_input)

    item_embedding = Embedding(output_dim=embedding_size_item, input_dim=item_count,
                               input_length=1, name='item_embedding')(item_id_input)

    user_vecs = Reshape([embedding_size_user])(user_embedding)
    item_vecs = Reshape([embedding_size_item])(item_embedding)

    dayofweek_dense = Dense(8, activation='relu')(dayofweek_input)
    hourofday_dense = Dense(16, activation='relu')(hourofday_input)
    monthofyear_dense = Dense(16, activation='relu')(monthofyear_input)

    input_vecs = Concatenate()([user_vecs, item_vecs, dayofweek_dense, hourofday_dense, monthofyear_dense])

    x = Dropout(0.3)(input_vecs)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)

    y = Dense(1)(x)

    model = Model(inputs=[user_id_input, item_id_input, dayofweek_input, hourofday_input, monthofyear_input], outputs=y)

    model.compile(loss='mse',
                  optimizer="adam"
                 )
    return model


def run_model_NN_FM(model, df, train):
    
    ###############
    # Run the network
    ###############

    import time
    from keras.callbacks import ModelCheckpoint

    save_path = mainpath + "/model"
    mytime = time.strftime("%Y_%m_%d_%H_%M")
    modname = 'matrix_facto_5_' + mytime 
    thename = save_path + '/' + modname + '.h5'
    mcheck = ModelCheckpoint(thename, monitor='val_loss', save_best_only=True)

    history = model.fit([train["userID"], train["itemID"], train["dayofweek"], train["hourofday"], train["monthofyear"]]
                        , target_col
                        , batch_size=batch_size, epochs=epochs
                        , validation_split=validation_split_perc
                        , callbacks=[mcheck]
                        , shuffle=True)

    import pickle
    with open(mainpath + '/histories/' + modname + '.pkl' , 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
        
if __name__ == "__main__":
    
    
#     print("System version: {}".format(sys.version))
#     print("Pandas version: {}".format(pd.__version__))
#     print("Tensorflow version: {}".format(tf.__version__))
    
#     necessary_col = ['userID', 'itemID', 'dayofweek', 'hourofday', 'monthofyear', 'quantity']

    root_path = "../data/"
    train_readin_path = root_path + "reward_transaction_subcat_withZero_train.csv"
    test_readin_path = root_path + "reward_transaction_subcat_withZero_test.csv"
    
    train = pd.read_csv(train_readin_path)
    test = pd.read_csv(test_readin_path)
    df = pd.concat([train, test],ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    print("/n" + "Finished loading")
    
    embedding_size_user = 10
    embedding_size_item = 10
    
    target_col = train["quantity"]
    validation_split_perc = 0.3
    batch_size = 32
    epochs = 5
    
    mainpath = '/home/admin711/notebooks/tian/result/deep_beer_try_out'

    model = model_structure_NN_MF(df)
    
    run_model_NN_FM(model, df, train)
    
        
        












