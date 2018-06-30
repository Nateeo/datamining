import tensorflow as tf
from openrec.recommenders import BPR
from openrec.modules.interactions import PairwiseEuDist
from openrec.modules.extractions import LatentFactor


class CMLTime(BPR):

    def _build_post_training_ops(self):
        unique_user_id, _ = tf.unique(self._get_input('user_id'))
        unique_item_id, _ = tf.unique(tf.concat(
            [self._get_input('p_item_id'), self._get_input('n_item_id')], axis=0))
        return [self._get_module('user_vec').censor_l2_norm_op(censor_id_list=unique_user_id),
                self._get_module('p_item_vec').censor_l2_norm_op(censor_id_list=unique_item_id)]

    def _build_interactions(self, train=True):

        if train:
            self._add_module('interaction',
                             PairwiseEuDist(user=self._get_module('user_vec').get_outputs()[0],
                                            p_item=self._get_module(
                                 'p_item_vec').get_outputs()[0],
                                 n_item=self._get_module(
                                                'n_item_vec').get_outputs()[0],
                                 p_item_bias=self._get_module(
                                 'p_item_bias').get_outputs()[0],
                                 n_item_bias=self._get_module(
                                                'n_item_bias').get_outputs()[0],
                                 scope='PairwiseEuDist', reuse=False, train=True),
                             train=True)
        else:
            self._add_module('interaction',
                             PairwiseEuDist(user=self._get_module('user_vec', train=train).get_outputs()[0],
                                            item=self._get_module(
                                 'item_vec', train=train).get_outputs()[0],
                                 item_bias=self._get_module(
                                 'item_bias', train=train).get_outputs()[0],
                                 scope='PairwiseEuDist', reuse=True, train=False),
                             train=False)

    def _input_mappings(self, batch_data, train):

        if train:
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('p_item_id'): batch_data['p_item_id_input'],
                    self._get_input('n_item_id'): batch_data['n_item_id_input'],
                    self._get_input('p_item_timestamp'): batch_data['p_item_timestamp_input'],
                    self._get_input('n_item_timestamp'): batch_data['n_item_timestamp_input']}
        else:
            return {self._get_input('user_id', train=train): batch_data['user_id_input']}

    def _build_extra_inputs(self, train=True):

        if train:
            self._add_input(name='p_item_timestamp',
                            dtype='int', shape=[self._batch_size])
            self._add_input(name='n_item_timestamp',
                            dtype='int', shape=[self._batch_size])

    def _build_extra_extractions(self, train=True):

        if train:
            self._add_module('p_extra_vec', LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('p_item_timestamp', train=train),
                            shape=[self._max_item, self._dim_embed], scope='item', reuse=not train), train=train)
            self._add_module('n_extra_vec', LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('n_item_timestamp', train=train),
                            shape=[self._max_item, self._dim_embed], scope='item', reuse=not train), train=train)



