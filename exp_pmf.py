import os
import sys
sys.path.append(os.getcwd())

from openrec import ItrMLPModelTrainer
from openrec.utils import ImplicitDataset, Dataset
from openrec.recommenders import PMF
from openrec.utils.evaluators import AUC, Recall, MSE
from openrec.utils.samplers import PointwiseSampler
from config import sess_config
import numpy as np

if __name__ == "__main__":
    # raw_data = dict()
    # raw_data['max_user'] = 5551
    # raw_data['max_item'] = 16980

    # raw_data['train_data'] = np.load('dataset/citeulike/user_data_train.npy')
    # raw_data['val_data'] = np.load('dataset/citeulike/user_data_val.npy')
    # raw_data['test_data'] = np.load('dataset/citeulike/user_data_test.npy')

    # print(raw_data['test_data'])
    raw_data = dict()
    raw_data['max_user'] = 19
    raw_data['max_item'] = 40
    csv = np.genfromtxt('Movies_ratings_small_merged_reduced.csv', delimiter=",", dtype='int,int,float,float,int,int,str,str,float,int,int,str,bool', names=True, encoding='ansi')
    raw_data['train_data'] = csv
    raw_data['val_data'] = csv
    raw_data['test_data'] = csv
    print(csv)

    batch_size = 10
    test_batch_size = 1
    display_itr = 100

    train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
    
    val_dataset = Dataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
    test_dataset = Dataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')
    test_dataset.shuffle() # need this to set to Dataset#data field
    val_dataset.shuffle()
   
    model = PMF(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
                    dim_embed=50, opt='Adam', sess_config=sess_config)
    sampler = PointwiseSampler(dataset=train_dataset, batch_size=batch_size, pos_ratio=0.2, num_process=5)
    model_trainer = ItrMLPModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
        train_dataset=train_dataset, model=model, sampler=sampler)

    # auc_evaluator = AUC()
    # recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    mse_evaluator = MSE()

    model_trainer.train(num_itr=int(1000), display_itr=display_itr, update_itr=1000, eval_datasets=[val_dataset],
                        evaluators=[mse_evaluator])