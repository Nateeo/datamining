import random

import numpy as np
import time
from openrec import ImplicitModelTrainer
from openrec.recommenders import BPR
from openrec.utils import ImplicitDataset
from openrec.utils.evaluators import AUC
from openrec.utils.samplers import PairwiseSampler
import pandas as pd

if __name__ == "__main__":
    # We'll need to do a lot more wrangling/cleaning up - but this seems to AT LEAST WORK WITH DIFF DATA.
    max_users = 10000
    max_items = 200000
    csv = np.genfromtxt('Movies_ratings_small_merged_larger2.csv', delimiter=",", dtype=(int,int,float,int,int,float,int,int,float), names=True, encoding=None)
    print(csv)

    train_dataset = ImplicitDataset(raw_data=csv, 
                            max_user=max_users, 
                            max_item=max_items, name='Train')
    val_dataset = ImplicitDataset(raw_data=csv, 
                        max_user=max_users,
                        max_item=max_items, name='Val')
    test_dataset = ImplicitDataset(raw_data=csv, 
                        max_user=max_users,
                        max_item=max_items, name='Test')

    bpr_model = BPR(batch_size=1000, 
                    max_user=train_dataset.max_user(), 
                    max_item=train_dataset.max_item(), 
                    dim_embed=20, 
                    opt='Adam')


    print("before sampler")
    sampler = PairwiseSampler(batch_size=1000, dataset=train_dataset)
    print("after sampler")

    auc_evaluator = AUC()
    print("after evaluator")

    model_trainer = ImplicitModelTrainer(batch_size=1000, 
                                test_batch_size=100, 
                                train_dataset=train_dataset, 
                                model=bpr_model, 
                                sampler=sampler)
    print("after implicit")

    model_trainer.train(num_itr=10, 
                        display_itr=10, 
                        eval_datasets=[val_dataset, test_dataset],
                        evaluators=[auc_evaluator])
