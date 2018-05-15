import os
import sys
import pickle
import numpy as np
sys.path.append(os.getcwd())

from openrec import ImplicitModelTrainer
from openrec.utils import ImplicitDataset
from openrec.recommenders import BPR
from BPR2 import BPR2
from openrec.recommenders import Recommender
from openrec.utils.evaluators import AUC
from openrec.utils.samplers import PairwiseSampler

import tensorflow as tf

sess_config = tf.ConfigProto()

raw_data = dict()
raw_data['max_user'] = 5551
raw_data['max_item'] = 16980

raw_data['train_data'] = np.load('dataset/citeulike/user_data_train.npy')
raw_data['val_data'] = np.load('dataset/citeulike/user_data_val.npy')
raw_data['test_data'] = np.load('dataset/citeulike/user_data_test.npy')

print(raw_data['test_data'])

pythonsucks = dict()

pythonsucks['user_id_input'] = []
pythonsucks['item_id_input'] = []

for pair in raw_data['test_data']:
    pythonsucks['user_id_input'].append(pair[0])
    pythonsucks['item_id_input'].append(pair[1])

batch_size = 10
test_batch_size = 10
display_itr = 10

train_dataset = ImplicitDataset(
    raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(
    raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(
    raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

max_user = train_dataset.max_user()
max_item = train_dataset.max_item()

# bpr_model = BPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
#                 dim_embed=20, opt='Adam', sess_config=sess_config)
# sampler = PairwiseSampler(batch_size=batch_size,
#                           dataset=train_dataset, num_process=5)
# model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
#                                      train_dataset=train_dataset, model=bpr_model, sampler=sampler)
# auc_evaluator = AUC()

# model_trainer.train(num_itr=int(10), display_itr=display_itr, eval_datasets=[val_dataset, test_dataset],
#                     evaluators=[auc_evaluator])

# print("Save")
# bpr_model.save("./model", 1)
# print("Saved")

big_bpr = BPR(batch_size=batch_size, max_user=max_user,
              max_item=max_item, dim_embed=20)
Recommender.load(big_bpr, "model-1")
print(big_bpr)
print(big_bpr.serve(pythonsucks))

# goodnight
tf.reset_default_graph()

big_bpr2 = BPR(batch_size=batch_size, max_user=max_user,
               max_item=max_item, dim_embed=20)
Recommender.load(big_bpr2, "model-1")
print(big_bpr2)
print(big_bpr2.serve(pythonsucks))

print(big_bpr)
print(big_bpr.serve(pythonsucks))

print(big_bpr2)
print(big_bpr2.serve(pythonsucks))
