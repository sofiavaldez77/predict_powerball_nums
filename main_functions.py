"""IMPORT MODULES"""
import pandas as pd
import requests


from sklearn.preprocessing import StandardScaler
import numpy as np

from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

""""DOWNLAOD DATA + CREATE PANDAS DATAFRAME"""
def url_to_df(url,csv_path):
    #DOWNLOAD DATA
    #this will pull data from tx powerball website
    #it pulls the numbers in the order drawn 
    req=requests.get(url)
    url_content=req.content
    csv_file=open(csv_path,'wb')
    csv_file.write(url_content)
    csv_file.close()
    #CREATE PANDAS DATAFRAME
    raw_df=pd.read_csv(csv_path, header=None)
    return raw_df

def clean_df(raw_df):
    #combine day,month,year columns into one column
    date_col=raw_df[1].astype(str) + "-"+ raw_df[2].astype(str) +"-"+ raw_df[3].astype(str)
    #drop powerplay, day,month,year,columns & "Powerball" string column
    raw_df=raw_df.drop(raw_df.columns[10],axis=1) 
    raw_df=raw_df.drop(raw_df.columns[0:4],axis=1) 
    #insert date as first column
    #we dont need the date for training but i just wanna make sure data is in correct order
    raw_df.insert(loc=0,column='Date Drawn',value=date_col)
    #rename columns
    raw_df.columns=['Date Drawn','First','Second','Third','Fourth','Fifth','Powerball']
    #save this version dataframe as csv for exploratory analysis later (for fun)
    raw_df.to_csv('explore_powerball.csv')
    #ok now let's remove 'date drawn' & 'powerball' b/c wont train w these
    raw_df=raw_df.drop(['Date Drawn','Powerball'],axis=1)
    return raw_df

""" SPLIT INTO TRAIN / VALIDATION/ TEST DATAFRAMES
 use sklearn train_test_split twice """

 """NORMALIZE TRAIN / VALIDATION/ TEST DATAFRAMES"""

def norm_train_df(train_df,normalize):
    #fit to / transform train data
    if normalize:
        scaler=StandardScaler().fit(train_df.values)
        train_norm=scaler.transform(train_df.values)
        train_norm_df=pd.DataFrame(data=train_norm,index=train_df.index)
    else:
        scaler=False
    return train_norm_df, scaler
    
def norm_other_df(other_df,scaler):
    #now transform test data using norm. params from train data
    #other df= validation & test dfs
    if scaler:
        other_norm=scaler.transform(other_df.values)
        other_norm_df=pd.DataFrame(data=other_norm,index=other_df.index)
    else:
        other_norm_df=other_df

    return other_norm_df


"""CREATE NUMPY ARRAYS OF TRAIN / VALIDATION / TEST FEATURES & LABELS 
(AKA INPUT FORMAT OF DEEP LEARNING MODEL)"""

def DL_format(norm_df,window_len):
    #format for keras lstm = (# rows, window size, # balls)
    num_rows=len(norm_df)
    num_features=len(norm_df.columns)
    features=np.empty([num_rows-window_len,window_len,num_features],dtype=float)
    labels=np.empty([num_rows-window_len,num_features],dtype=float)
    for i in range(0,num_rows-window_len):
        features[i]=norm_df.iloc[i:i+window_len,0:num_features]
        labels[i]=norm_df.iloc[i+window_len: i+window_len+1, 0:num_features] 
    #verify shapes of train/label:
    print("features shape= ",str(features.shape),"\nlabel shape= " ,str(labels.shape))
    #verify content of train/label:
    #sample=1
    #print("features sample= \n",str(features[sample]),"\nlabel sample=",str(labels[sample]))
    return features,labels

"""CREATE MODEL"""

def create_model(train_features):
    window_len=train_features.shape[1]
    num_features=train_features.shape[2]
    #create model
    model=Sequential()
    model.add(LSTM(56,activation='relu',
                input_shape=(window_len,num_features),
                return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(56,activation='relu',
                return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_features))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    #print(model.summary())
    return model

"""TRAIN MODEL aka model.fit(---)"""


"""DIAGNOSE OVERFITTING/UNDERFITTING OF MODEL"""
def plot_train_val(trained_model):
    pyplot.plot(trained_model.history['loss'])
    pyplot.plot(trained_model.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

def eval_test(test_features,test_labels,normalize,scaler):
    scores=model.evaluate(test_features,test_labels)
    LSTM_accuracy = scores[1]*100
    print('Test accuracy: ', scores[1]*100)
    test_predictions=model.predict(test_features)
