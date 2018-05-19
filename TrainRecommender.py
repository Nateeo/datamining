# largely a copy of exp_pmf.py for testing

import os
import sys
import datetime
import pickle
import numpy as np
sys.path.append(os.getcwd())

from openrec import ImplicitModelTrainer
from openrec.utils import ImplicitDataset
from openrec.recommenders import CML
from openrec.recommenders import Recommender
from openrec.utils.evaluators import AUC, NDCG, Recall, Precision
from openrec.utils.samplers import PairwiseSampler

from ImplicitDatasetWithExplicitConversion import ImplicitDatasetWithExplicitConversion
from PairwiseSamplerWithExplicitConversion import PairwiseSamplerWithExplicitConversion

import tensorflow as tf

import training_config as tc

config = tf.ConfigProto()

raw_data = dict()
raw_data['max_user'] = tc.max_user
raw_data['max_item'] = tc.max_item
fileToLoad = 'move_merge_large_timebased.csv'
csv = np.genfromtxt(fileToLoad, delimiter=",", dtype=[
                    int, int, float, float, float, int, int, int, int, str, int], names=True, encoding='utf8')
#csv = np.genfromtxt('movies_medium.csv', delimiter=",", dtype='int,int,float,bool,float,float', names=True, encoding='ansi')
#csv = np.genfromtxt('Movies_ratings_small_merged_reduced.csv', delimiter=",", dtype='int,int,float,float,int,int,str,str,float,int,int,str,bool', names=True, encoding='ansi')

# Permute all the data, then subsection it off - using temp AND THEN numpy
np.random.shuffle(csv)
permuter = dict()
valTemp = []
testTemp = []
trainTemp = []
for sample_itr, entry in enumerate(csv):
    if(entry['rating'] >= 3):
        if(entry['user_id'] not in permuter):
            permuter[entry['user_id']] = 0
        index = permuter[entry['user_id']]
        if index == 0:
            trainTemp.append(entry)
            permuter[entry['user_id']] = 1
        elif index == 1:
            trainTemp.append(entry)
            permuter[entry['user_id']] = 2
        elif index == 2:
            valTemp.append(entry)
            permuter[entry['user_id']] = 3
        else:
            testTemp.append(entry)
            permuter[entry['user_id']] = 0


raw_data['train_data'] = np.array(trainTemp)
raw_data['val_data'] = np.array(valTemp)
raw_data['test_data'] = np.array(testTemp)

batch_size = tc.batch_size
test_batch_size = tc.test_batch_size
display_itr = tc.display_itr

train_dataset = ImplicitDatasetWithExplicitConversion(
    raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDatasetWithExplicitConversion(
    raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDatasetWithExplicitConversion(
    raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

model = CML(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
            dim_embed=20, opt='Adam', sess_config=config)
sampler = PairwiseSamplerWithExplicitConversion(
    dataset=train_dataset, batch_size=batch_size, num_process=3)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
                                     train_dataset=train_dataset, model=model, sampler=sampler)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
precision_evaluator = Precision(
    precision_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ndcg_evaluator = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

print(((str(datetime.datetime.now())).split('.')[0]).split(' ')[
      1] + ' ' + ((str(datetime.datetime.now())).split('.')[0]).split(' ')[0])
print(fileToLoad)
model_trainer.train(num_itr=int(10), display_itr=display_itr, eval_datasets=[val_dataset],
                    evaluators=[auc_evaluator, recall_evaluator, precision_evaluator, ndcg_evaluator])

print("Save")
model.save("./model", 3)
print("Saved")

# big_bpr = BPR(batch_size=batch_size, max_user=max_user,
#               max_item=max_item, dim_embed=20)
# Recommender.load(big_bpr, "model-1")

# print("model loaded")
# print(big_bpr)

# print(big_bpr.serve(pythonsucks))
