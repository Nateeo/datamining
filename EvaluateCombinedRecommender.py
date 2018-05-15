from CombinedRecommender import CombinedRecommender

import os
import sys
import pickle
import numpy as np
sys.path.append(os.getcwd())

raw_data = dict()
raw_data['train_data'] = np.load('dataset/citeulike/user_data_train.npy')
raw_data['val_data'] = np.load('dataset/citeulike/user_data_val.npy')
raw_data['test_data'] = np.load('dataset/citeulike/user_data_test.npy')

# construct CombinedRecommender with the same batch_size max_item max_user
# as the component recommenders

batch_size = 10
test_batch_size = 10
display_itr = 10

max_user = 5551
max_item = 16980

test_dat = dict()

test_dat['user_id_input'] = []
test_dat['item_id_input'] = []

for pair in raw_data['test_data']:
    test_dat['user_id_input'].append(pair[0])
    test_dat['item_id_input'].append(pair[1])


combined_recommender = CombinedRecommender(
    batch_size=batch_size, max_user=max_user, max_item=max_item)

print('constructed combined recommender')

print(combined_recommender.serve(test_dat))
