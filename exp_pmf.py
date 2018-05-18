import os
import sys
sys.path.append(os.getcwd())

from openrec import ItrMLPModelTrainer, ImplicitModelTrainer
from openrec.utils import ImplicitDataset, Dataset
from openrec.recommenders import PMF, NCML, ItrMLP, BPR, CML, CDL
from openrec.utils.evaluators import AUC, Recall, MSE
from openrec.utils.samplers import PointwiseSampler, PairwiseSampler
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
    raw_data['max_user'] = 672
    raw_data['max_item'] = 4500
    csv = np.genfromtxt('movies_medium.csv', delimiter=",", dtype='int,int,float,bool,float,float', names=True, encoding='ansi')
    #csv = np.genfromtxt('Movies_ratings_small_merged_reduced.csv', delimiter=",", dtype='int,int,float,float,int,int,str,str,float,int,int,str,bool', names=True, encoding='ansi')
    raw_data['train_data'] = csv
    raw_data['val_data'] = csv
    raw_data['test_data'] = csv
    print(csv)

    batch_size = 5000
    test_batch_size = 500
    display_itr = 100

    train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
    
    val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
    test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')
    test_dataset.shuffle() # need this to set to Dataset#data field
    val_dataset.shuffle()
   
    model = CML(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
                    dim_embed=20, opt='Adam')
    sampler = PairwiseSampler(dataset=train_dataset, batch_size=batch_size, num_process=2)
    model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
        train_dataset=train_dataset, model=model, sampler=sampler)

    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    model_trainer.train(num_itr=int(300), display_itr=display_itr, eval_datasets=[val_dataset],
                        evaluators=[auc_evaluator, recall_evaluator])

    test_dat = dict()

    test_dat['user_id_input'] = []
    test_dat['item_id_input'] = []

    test_dat['user_id_input'].append(1)
    test_dat['item_id_input'].append(1)
    # test_dat['label'] = []

    # # test_dat['user_id_input'].append(9)

    # for pair in raw_data['test_data']:
    #     test_dat['user_id_input'].append(pair[0])
    #     test_dat['item_id_input'].append(pair[1])
    
    print("Serving shit")
    scores = model.serve(test_dat)
    # stuff = np.array([])

    # dtype = [('index', int), ('score', float)]

    # final_scores = np.array([], dtype=dtype)

    # for inx, score in enumerate(scores[0]):
    #     print('index: ', end='')
    #     print(inx)
    #     print('score', end='')
    #     print(score)
    #     final_scores = np.append(final_scores, np.array([inx, score], dtype=final_scores.dtype))

    # sorted = np.sort(final_scores, order='score')

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


    