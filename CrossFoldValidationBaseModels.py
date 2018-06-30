import os
import sys
import datetime

import numpy as np
from openrec import ImplicitModelTrainer, ItrMLPModelTrainer
from openrec.recommenders import BPR, CDL, CML, NCML, PMF, ItrMLP, VisualCML, Recommender
from openrec.utils import Dataset, ImplicitDataset
from openrec.utils.evaluators import AUC, Recall, Precision, NDCG
from openrec.utils.samplers import PairwiseSampler, PointwiseSampler

from config import config
from ImplicitDatasetWithExplicitConversion import ImplicitDatasetWithExplicitConversion
from PairwiseSamplerWithExplicitConversion import PairwiseSamplerWithExplicitConversion
from CMLTime import CMLTime

sys.path.append(os.getcwd())


if __name__ == "__main__":
    raw_data = dict()
    raw_data['max_user'] = 10657
    raw_data['max_item'] = 4251
    fileToLoad = 'rec3_popularity_main_shuffled.csv'
    # for rec1 csv = np.genfromtxt(fileToLoad, delimiter=",", dtype=[int,int,float,int,int,int,int,int,int,int], names=True, encoding='utf8')
    # for rec2 csv = np.genfromtxt(fileToLoad, delimiter=",", dtype=[int,int,float,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int], names=True, encoding='utf8')
    csv = np.genfromtxt(fileToLoad, delimiter=",", dtype=[int,int,float,int,int,int,int,int,int,float,int,float,int], names=True, encoding='utf8')

    number_of_folds = 5
    current_fold = 0
    fold_data = dict()
    fold_data[0] = []
    fold_data[1] = []
    fold_data[2] = []
    fold_data[3] = []
    fold_data[4] = []

    for sample_itr, entry in enumerate(csv):
        if(entry['rating'] >= 3):
            fold_data[current_fold].append(entry)
            current_fold = current_fold + 1
            if(current_fold > 4):
                current_fold = 0

                

    raw_data[0] = np.array(fold_data[0])
    raw_data[1] = np.array(fold_data[1])
    raw_data[2] = np.array(fold_data[2])
    raw_data[3] = np.array(fold_data[3])
    raw_data[4] = np.array(fold_data[4])
    
    print(len(raw_data[0]))
    print(raw_data[0])
    print(len(raw_data[1]))
    print(raw_data[1])
    print(len(raw_data[2]))
    print(len(raw_data[3]))
    print(len(raw_data[4]))

    batch_size = 1000
    test_batch_size = 100
    display_itr = 10000

    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    precision_evaluator = Precision(precision_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ndcg_evaluator = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    model = CML(batch_size=batch_size, max_user=raw_data['max_user'], max_item=raw_data['max_item'], 
                    dim_embed=20, opt='Adam', sess_config=config)

    print(((str(datetime.datetime.now())).split('.')[0]).split(' ')[1] + ' ' + ((str(datetime.datetime.now())).split('.')[0]).split(' ')[0])
    print(fileToLoad)

    Recommender.load(model, "model-53")

    for x in range(4,5):
        # Use the CURRENT ITERATIOn to determine the fold to leave out, concat the rest.
        val_fold = raw_data[x]
        init_value = x + 1
        if(init_value > 4):
            init_value = 0
        training_folds = raw_data[init_value]
               
        print('TESTING FOLD AND LENGTH (TO BE EXCLUDED FROM TRAINING) = ')
        print(x)
        print(len(val_fold))
        print('First training fold and length = ')
        print(init_value)
        print(len(training_folds))
        for y in range(0,5):
            if y != x:
                if init_value != y:
                    print('concating - ')
                    print(y)
                    training_folds = np.concatenate((training_folds, raw_data[y]))
                    print(len(training_folds))
    
        print('Val Fold length and samples:')
        print(len(val_fold))
        print(val_fold)
        
        print('Training Folds legnth and samples:')
        print(len(training_folds))
        print(training_folds)

        train_dataset = ImplicitDatasetWithExplicitConversion(training_folds, raw_data['max_user'], raw_data['max_item'], name='Train')
        val_dataset = ImplicitDatasetWithExplicitConversion(val_fold, raw_data['max_user'], raw_data['max_item'], name='Val')

        sampler = PairwiseSamplerWithExplicitConversion(dataset=train_dataset, batch_size=batch_size, num_process=3)
        model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
        train_dataset=train_dataset, model=model, sampler=sampler)

        model_trainer.train(num_itr=int(10010), display_itr=display_itr, eval_datasets=[val_dataset],
                            evaluators=[auc_evaluator, recall_evaluator, precision_evaluator, ndcg_evaluator])
        
        print("Save at iteration")
        print(x)
        model.save("./model", 53)
        print("Saved")