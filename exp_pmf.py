import os
import sys

import numpy as np
from openrec import ImplicitModelTrainer, ItrMLPModelTrainer
from openrec.recommenders import BPR, CDL, CML, NCML, PMF, ItrMLP
from openrec.utils import Dataset, ImplicitDataset
from openrec.utils.evaluators import AUC, Recall, Precision, NDCG
from openrec.utils.samplers import PairwiseSampler, PointwiseSampler

from config import config
from ImplicitDatasetWithExplicitConversion import ImplicitDatasetWithExplicitConversion
from PairwiseSamplerWithExplicitConversion import PairwiseSamplerWithExplicitConversion

sys.path.append(os.getcwd())


if __name__ == "__main__":
    raw_data = dict()
    raw_data['max_user'] = 10657
    raw_data['max_item'] = 4251
    csv = np.genfromtxt('move_merge_large_timebased.csv', delimiter=",", dtype=[int,int,float,float,float,int,int,int,int,str,int], names=True, encoding='utf8')
    #csv = np.genfromtxt('movies_medium.csv', delimiter=",", dtype='int,int,float,bool,float,float', names=True, encoding='ansi')
    #csv = np.genfromtxt('Movies_ratings_small_merged_reduced.csv', delimiter=",", dtype='int,int,float,float,int,int,str,str,float,int,int,str,bool', names=True, encoding='ansi')
    
    #Permute all the data, then subsection it off - using temp AND THEN numpy
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
    
    print(len(raw_data['train_data']))
    print(raw_data['train_data'])
    print(len(raw_data['val_data']))
    print(raw_data['val_data'])
    print(len(raw_data['test_data']))
    print(raw_data['test_data'])

    batch_size = 1000
    test_batch_size = 100
    display_itr = 10000

    train_dataset = ImplicitDatasetWithExplicitConversion(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
    val_dataset = ImplicitDatasetWithExplicitConversion(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
    test_dataset = ImplicitDatasetWithExplicitConversion(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

    model = CML(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
                    dim_embed=20, opt='Adam', sess_config=config)
    sampler = PairwiseSamplerWithExplicitConversion(dataset=train_dataset, batch_size=batch_size, num_process=3)
    model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
        train_dataset=train_dataset, model=model, sampler=sampler)

    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    precision_evaluator = Precision(precision_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ndcg_evaluator = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    model_trainer.train(num_itr=int(30000), display_itr=display_itr, eval_datasets=[val_dataset],
                        evaluators=[auc_evaluator, recall_evaluator, precision_evaluator, ndcg_evaluator])

    test_dat = dict()

    test_dat['user_id_input'] = []
    test_dat['item_id_input'] = []

    test_dat['user_id_input'].append(1)
    test_dat['item_id_input'].append(1)
    
    print("Serving shit")
    scores = model.serve(test_dat)

    arr = []

    print(scores[0][1371])
    print(scores[0][1405])
    print(scores[0][2105])
    average = 0
    num = 0
    max = 0
    for inx, score in enumerate(scores[0]):
        if (score > max): max = score
        num += 1
        average += score
        arr.append([inx, score])
    print('average score: ', end='')
    print(average/num)
    print('max: ', end='')
    print(max)
    # arr.sort(key=lambda x: x[1])

    # for value in arr:
    #     print(value)
