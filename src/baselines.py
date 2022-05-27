import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# read MovieLens 1M dataset
ratings_df = pd.read_csv('../data/ratings.dat',sep="::",header=None)

# rename columns
ratings_df = ratings_df[[0,1,2]].rename(columns={0:'user_id',1:'movie_id',2:'rating'})

# split into train, valid and test sets
test_size, valid_size = 0.1, 0.1
test_split_random_state, valid_split_random_state= 42, 0 
train_valid_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=test_split_random_state)
train_df, valid_df = train_test_split(train_valid_df, test_size=valid_size, random_state=valid_split_random_state)

# performance metric root mean squared error
rmse = lambda y_true,y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

#Baseline 1: predict all ratings '3'
y_pred_dummy = np.ones((len(test_df.rating.values))) * 3
y_true = test_df.rating.values
print(f'baseline 1 test rmse: {rmse(y_true,y_pred_dummy)}')

#Baseline 2: predict all ratings train_df's mean rating
y_pred_dummy = np.ones((len(test_df.rating.values))) * np.mean(train_valid_df.rating)
y_true = test_df.rating.values
print(f'baseline 2 test rmse: {rmse(y_true,y_pred_dummy)}')

#Baseline 3: predict user mean rating
user_means = train_valid_df.groupby('user_id').agg({'rating':'mean'}).reset_index()
user_means = user_means.rename(columns={'rating':'preds'})
merged_df = test_df.merge(user_means)
print(f'baseline 3 test rmse: {rmse(merged_df.rating,merged_df.preds)}')

#Baseline 4: predict item mean rating
item_means = train_valid_df.groupby('movie_id').agg({'rating':'mean'}).reset_index()
item_means = item_means.rename(columns={'rating':'preds'})
merged_df = test_df.merge(item_means, how='left')
# fill movies not present in training set with average movie ratings
merged_df['preds'] = merged_df['preds'].fillna((merged_df['preds'].mean()))
print(f'baseline 4 test rmse: {rmse(merged_df.rating,merged_df.preds)}')