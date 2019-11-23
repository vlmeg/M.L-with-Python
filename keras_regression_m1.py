# -*- coding: utf-8 -*-
'''
Created on Nov 17, 2018

@author: vm
Description:https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
'''


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set path to input
path_input = '/Users/vm/Desktop/nn_Regression/'


def get_data():
    # get train data
    train_data_path = path_input+'train.csv'
    train = pd.read_csv(train_data_path)
    # get test data
    test_data_path = path_input+'test.csv'
    test = pd.read_csv(test_data_path)
    return train , test


def get_combined_data():
    # reading train data
    train , test = get_data()
    target = train.SalePrice
    train.drop(['SalePrice'], axis = 1 , inplace = True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Id'], inplace=True, axis=1)
    return combined, target


# Load train and test data into pandas DataFrames
train_data, test_data = get_data()

# Combine train and test data to process them together
combined, target = get_combined_data()

# Generates descriptive statistics that summarize the central tendency,
# dispersion and shape of a dataset’s distribution, excluding NaN values
descriptive=combined.describe()


def get_cols_with_no_nans(df,col_type):
    """
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    """
    if col_type == 'num':
        predictors = df.select_dtypes(exclude=['object'])
    elif col_type == 'no_num':
        predictors = df.select_dtypes(include=['object'])
    elif col_type == 'all':
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')
print(num_cols,cat_cols)

combined = combined[num_cols + cat_cols]
combined.hist(figsize = (12,10))
# plt.show()


# Correlation between features
train_data = train_data[num_cols + cat_cols]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
# plt.show()


# Encoding categorical features using one hot encoder
def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)
            # drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df
    

print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(combined.shape[1]))


def split_combined():
    global combined
    train = combined[:1460]
    test = combined[1460:]
    return train , test 


train, test = split_combined()


# ===============================================================================
# Second : Make the Deep Neural Network
# ===============================================================================
'''
Define a sequential model
Add some dense layers
Use ‘relu’ as the activation function for the hidden layers
Use a ‘normal’ initializer as the kernal_intializer

We will use mean_absolute_error as a loss function
Define the output layer with only one node
Use ‘linear ’as the activation function for the output layer
'''
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# Define a check point callback
checkpoint_name = path_input+'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# ===============================================================================
# 3.TRAIN THE MODEL
# ===============================================================================


NN_model.fit(train, target, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Load wights file of the best model :
wights_file = path_input+'Weights-408--18717.35616.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# ===============================================================================
# 4. TEST THE MODEL
# ===============================================================================

def make_submission(prediction, sub_name):
    my_submission = pd.DataFrame({'Id':pd.read_csv(path_input+'test.csv').Id,'SalePrice':prediction})
    my_submission.to_csv('{}.csv'.format(sub_name),index=False)
    print('A submission file has been made')


predictions = NN_model.predict(test)
make_submission(predictions[:,0],path_input+'submission(NN).csv')


# ===============================================================================
# 5. TRY ANOTHER ML ALGORITHM
# ===============================================================================

# Use random forest

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.25, random_state = 14)

model = RandomForestRegressor()
model.fit(train_X, train_y)

# Get the mean absolute error on the validation data
predicted_prices = model.predict(val_X)
MAE = mean_absolute_error(val_y , predicted_prices)
print('Random forest validation MAE = ', MAE)
