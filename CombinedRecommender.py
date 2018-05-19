from openrec.recommenders import Recommender
from openrec.recommenders import BPR

import tensorflow as tf


class CombinedRecommender(Recommender):
    def _build_user_inputs(self, train=True):
        print('calling _build_user_inputs')
        if train:
            self._add_input(name='user_id', dtype='int32',
                            shape=[self._batch_size])
        else:
            self._add_input(name='user_id', dtype='int32',
                            shape=[None], train=False)

    def _build_item_inputs(self, train=True):
        print('calling _build_item_inputs')
        if train:
            self._add_input(name='item_id', dtype='int32',
                            shape=[self._batch_size])
        else:
            self._add_input(name='item_id', dtype='none', train=False)

    # this is for contextual data
    def _build_extra_inputs(self, train=True):
        print('calling _build_extra_inputs')
        pass
        # if train:
        #     self._add_input(name='lables', dtype='float32', shape=[self._batch_size])

    '''
    Define input mappings
    '''

    def _input_mappings(self, batch_data, train):
        print('calling _input_mappings')
        if train:
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('item_id'): batch_data['item_id_input'],
                    self._get_input('labels'): batch_data['labels']}
        else:
            return {self._get_input('user_id', train=False): batch_data['user_id_input']}

    '''
    Define modules using `add_module` function.
    '''

    def _build_user_extractions(self, train=True):
        print('calling _build_user_extractions')
        pass

    def _build_item_extractions(self, train=True):
        print('calling _build_item_extractions')
        pass

    def _build_default_interactions(self, train=True):
        print('calling _default_interactions')
        pass

    def _build_serving_graph(self):
        big_bpr = BPR(batch_size=self._batch_size, max_user=self._max_user,
                      max_item=self._max_item, dim_embed=20)
        Recommender.load(big_bpr, "model-1")
        print('calling _build_serving_graph')

        tf.reset_default_graph()

        big_bpr2 = BPR(batch_size=self._batch_size, max_user=self._max_user,
                       max_item=self._max_item, dim_embed=20)
        Recommender.load(big_bpr2, "model-1")

        tf.reset_default_graph()

        big_bpr3 = BPR(batch_size=self._batch_size, max_user=self._max_user,
                       max_item=self._max_item, dim_embed=20)
        Recommender.load(big_bpr2, "model-1")

        self._rec1 = big_bpr
        self._rec2 = big_bpr2
        self._rec3 = big_bpr3
        pass

    def _build_training_graph(self):
        print('calling _build_training_graph')
        pass

    def serve(self, batch_data):
        print('calling serve')
        return self._rec1.serve(batch_data)

    # default train method, need to override with ensemble loss calculation
    def train(self, batch_data):
        """Train the model with an input batch_data.

        Parameters
        ----------
        batch_data: dict
            A batch of training data.
        """

        _, loss = self._sess.run([self._train_op, self._loss],
                                 feed_dict=self._input_mappings(batch_data, train=False))
        return loss
